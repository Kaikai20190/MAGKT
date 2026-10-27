"""Memory activation encoding layer (Section 3.4 of the paper).

RetrievalMemory (memory.type = retrieval, used in all reported experiments)
  Row t of a window prepares the prediction of the response to q_{t+1} and retrieves from the learner's
  interactions s <= t of the same window with K heads (memory.heads):

    lag ............................. l(t, s) = t + 1 - s  (>= 1)
                                      l(t, s) = (tau_{t+1} - tau_s) / time_scale + 1   (memory.decay_time = time)
    same item ....................... delta(t, s) = 1[q_s = q_{t+1}] or 1[c_s = c_{t+1} != 0]
    associative prior ............... pi(t, s) = u * delta + w * S[q_s, q_{t+1}] * (1 - delta)
    score of head k ................. a_k(t, s) = <q_{t+1} W_Q, h_{q_s} W_M>_k / sqrt(E/K)
                                                   - d_k * ln l(t, s) + ln(1 + pi(t, s))
    retrieval weights ............... alpha_k(t, .) = masked softmax over s <= t
    memory content .................. v_s = (h_{q_s} + e_{r_s}) W_v + b_v
    memory representation ........... x^d_t = LN(v_t) + LN([c_1; ...; c_K] W_o + b_o),  c_k = sum_s alpha_k v_s^(k)
    success / failure activations ... A+_t = ln(1 + sum_s delta * r_s * l^-d),  A-_t = ln(1 + sum_s delta * (1 - r_s) * l^-d)

  d_k, d, u and w are positive (softplus) and shared by all learners; the d_k are initialised at values evenly
  spaced on a log scale between 0.1 and 1 (memory.learn_d = false keeps them fixed), d at memory.d_init.
  memory.indiv_d makes the head decay rates depend on the learner's history (ablation 'indiv_d').
  memory.individual = false gives the shared, history-free memory of the ablation 'no_individual':
    x^d_t = LN((S^T V + V)[q_t]),  V = (h + (e_0 + e_1)/2) W_v + b_v.

ActivationMemory (memory.type = activation)
  An activation-weighted aggregation layer with an activation gate, kept as an alternative formulation; its
  options (memory.target, output, act_offset, act_gate, gate_monotone, chunk, history) are not used in the
  reported experiments. RetrievalMemory reuses its parameter handling.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y):
    return math.log(math.expm1(y))


class ActivationMemory(nn.Module):
    def __init__(self, dim, S, mcfg):
        super().__init__()
        self.dim = dim
        # S: [Q, Q] transition affinity, stored sparse; entries are looked up by key q_s*Q + q_t
        if not S.is_sparse:
            S = S.to_sparse()
        S = S.coalesce()
        self.Q = S.shape[0]
        idx = S.indices()
        self.register_buffer("S_keys", idx[0] * self.Q + idx[1], persistent=False)   # sorted
        self.register_buffer("S_vals", S.values(), persistent=False)
        self.register_buffer("S_T", S.t().coalesce(), persistent=False)
        self.individual = mcfg["individual"]
        self.answer_aware = mcfg["answer_aware"]
        self.forgetting = mcfg["forgetting"]
        self.spreading = mcfg["spreading"]
        self.act_gate = mcfg["act_gate"]
        self.gate_monotone = mcfg["gate_monotone"]
        self.decay_time = mcfg["decay_time"]
        self.time_scale = float(mcfg["time_scale"])
        self.eps = float(mcfg["eps"])
        self.target = mcfg.get("target", "current")          # current | next
        self.chunk = mcfg.get("chunk", "question")           # question | concept
        self.output = mcfg.get("output", "legacy")           # legacy: LN(g*c) | residual: LN(v_t) + g*LN_r(c)
        off = mcfg.get("act_offset")
        self.act_offset = self.eps if off is None else float(off)
        if self.output == "residual":
            self.norm_base = nn.LayerNorm(dim)

        self.raw_d = nn.Parameter(torch.tensor(_inv_softplus(mcfg["d_init"])),
                                  requires_grad=bool(mcfg["learn_d"]))
        self.raw_w = nn.Parameter(torch.tensor(_inv_softplus(mcfg["w_init"])),
                                  requires_grad=bool(mcfg["learn_w"]))

        self.ans_emb = nn.Embedding(2, dim)
        self.value_proj = nn.Linear(dim, dim)
        # gate: same parameter count as nn.Linear(1, dim)
        self.gate_u = nn.Parameter(torch.empty(dim).uniform_(-1.0, 1.0))
        self.gate_b = nn.Parameter(torch.empty(dim).uniform_(-1.0, 1.0))
        if self.gate_monotone:
            self.gate_u.data = torch.full((dim,), _inv_softplus(0.5))
        self.drop = nn.Dropout(mcfg["dropout"])
        self.norm = nn.LayerNorm(dim)

    def lookup_S(self, rows, cols):
        """S[rows, cols] (broadcast) from the sparse representation."""
        key = (rows * self.Q + cols)
        flat = key.reshape(-1)
        if self.S_keys.numel() == 0:
            return torch.zeros(key.shape, device=key.device)
        pos = torch.searchsorted(self.S_keys, flat).clamp(max=self.S_keys.numel() - 1)
        found = self.S_keys[pos] == flat
        val = torch.where(found, self.S_vals[pos], torch.zeros_like(self.S_vals[pos]))
        return val.reshape(key.shape)

    # ------------------------------------------------------------ params
    def d_value(self):
        return F.softplus(self.raw_d) if self.forgetting else self.raw_d.new_zeros(())

    def w_value(self):
        return F.softplus(self.raw_w) if self.spreading else self.raw_w.new_zeros(())

    def gate_weight(self):
        return F.softplus(self.gate_u) if self.gate_monotone else self.gate_u

    # ----------------------------------------------------------- forward
    def _forward_global(self, qid, h_struct):
        valid = qid >= 0
        safe = qid.clamp(min=0)
        if self.answer_aware:
            avg_ans = 0.5 * (self.ans_emb.weight[0] + self.ans_emb.weight[1])
            V = self.value_proj(h_struct + avg_ans.unsqueeze(0))
        else:
            V = self.value_proj(h_struct)
        ctx = (torch.sparse.mm(self.S_T, V) + V)[safe]
        x_d = self.norm(ctx) * valid.unsqueeze(-1).float()
        nan = torch.full(qid.shape, float("nan"), device=qid.device)
        return x_d, {"activation": nan, "gate_mean": nan}

    def forward(self, qid, correct, h_struct, times=None, hist=None, kc=None):
        """qid, correct: [B, T] with -1 padding; h_struct: [Q, E]; times: [B, T] seconds or None.
        hist: optional (hq, hr, ht) [B, H] -- interactions of the same learner BEFORE the window
        (memory.history > 0); they are added to the memory history s (keys) but are not predicted."""
        if not self.individual:
            return self._forward_global(qid, h_struct)

        B, T = qid.shape
        dev = qid.device
        valid = qid >= 0
        safe = qid.clamp(min=0)

        # keys (history positions s) = [prefix ; window]; queries (rows t) = window
        if hist is not None:
            hq, hr, ht, hc = hist
            kq = torch.cat([hq, qid], dim=1)
            kr = torch.cat([hr, correct], dim=1)
            kt = torch.cat([ht, times], dim=1) if times is not None else None
            kk = torch.cat([hc, kc], dim=1) if kc is not None else None
        else:
            kq, kr, kt, kk = qid, correct, times, kc
        H = kq.shape[1] - T
        kvalid = kq >= 0
        ksafe = kq.clamp(min=0)
        kans = kr.clamp(min=0)

        nxt = self.target == "next"
        if nxt:                                    # retrieval target of row t = q_{t+1}
            tq = torch.cat([safe[:, 1:], safe[:, -1:]], dim=1)
            tvalid = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
        else:                                      # target = the question just answered
            tq, tvalid = safe, valid

        # retention phi(t, s): [B or 1, T, H+T]
        if self.decay_time == "time":
            if times is None:
                raise ValueError("decay_time='time' requires timestamps (set data.t_line)")
            t_ret = torch.cat([times[:, 1:], times[:, -1:]], dim=1) if nxt else times
            delta = (t_ret.unsqueeze(2) - kt.unsqueeze(1)) / self.time_scale
        else:
            rows = torch.arange(T, device=dev, dtype=torch.float32) + H     # key-axis position of row t
            cols = torch.arange(H + T, device=dev, dtype=torch.float32)
            delta = (rows.view(T, 1) - cols.view(1, H + T)).unsqueeze(0)    # (t - s); +1 below
        recency = torch.pow(delta.clamp(min=0) + 1.0, -self.d_value())

        # association c(t, s): same chunk as the target + w * S[q_s, target_t]
        # chunk = question or knowledge concept (memory.chunk = concept; questions without a
        # concept fall back to question identity)
        same_q = tq.unsqueeze(2) == ksafe.unsqueeze(1)                     # [B, T(t), H+T(s)]
        if self.chunk == "concept":
            if kc is None:
                raise ValueError("memory.chunk = concept needs per-interaction concepts (data.c_line)")
            kks = kk.clamp(min=0)
            tc = torch.cat([kc[:, 1:], kc[:, -1:]], dim=1).clamp(min=0) if nxt else kc.clamp(min=0)
            same_c = (tc.unsqueeze(2) == kks.unsqueeze(1)) & (tc.unsqueeze(2) > 0)
            same = (same_q | same_c).float()
        else:
            same = same_q.float()
        compat = same
        if self.spreading:
            spread = self.lookup_S(ksafe.unsqueeze(1), tq.unsqueeze(2))      # [b,t,s] = S[q_s, target_t]
            compat = compat + self.w_value() * spread * (1.0 - same)

        causal = (torch.arange(H + T, device=dev).view(1, H + T)
                  <= (torch.arange(T, device=dev) + H).view(T, 1))           # s <= t
        mask = causal.unsqueeze(0) & kvalid.unsqueeze(1)
        if nxt:
            mask = mask & tvalid.unsqueeze(2)
        a = recency * compat * mask.float()                               # [B, T, H+T]
        total = a.sum(dim=-1, keepdim=True)
        activation = torch.log(total + self.act_offset)                   # [B, T, 1]

        val = h_struct[ksafe]
        if self.answer_aware:
            val = val + self.ans_emb(kans)
        v = self.value_proj(val)                                          # [B, H+T, E]
        alpha = self.drop(a / (total + self.eps))
        ctx = torch.einsum("bts,bse->bte", alpha, v)

        gate_mean = torch.full((B, T), float("nan"), device=dev)
        g = None
        if self.act_gate:
            g = torch.sigmoid(activation * self.gate_weight() + self.gate_b)  # [B, T, E]
            gate_mean = g.mean(-1)
        if self.output == "residual":
            mem = self.norm(ctx)
            if g is not None:
                mem = mem * g
            x_d = (self.norm_base(v[:, H:]) + mem) * valid.unsqueeze(-1).float()
        else:
            if g is not None:
                ctx = ctx * g
            x_d = self.norm(ctx) * valid.unsqueeze(-1).float()
        if nxt:
            activation = activation * tvalid.unsqueeze(-1).float()
        return x_d, {"activation": activation.squeeze(-1), "gate_mean": gate_mean}


class RetrievalMemory(ActivationMemory):
    """Retrieval memory of MAGKT (memory.type = retrieval).

    Row t predicts the response to q_{t+1}; the memory retrieves from the learner's past interactions s <= t
    (window, plus memory.history interactions before it) with multi-head attention whose logits contain
    the ACT-R power-law retention in log form:

      score_h(t, s) = <W_q u_{q_{t+1}}, W_k u_{q_s}>_h / sqrt(d_k)            content match
                      - d_h * ln(lag(t, s))                                  retention lag^(-d_h)
                      + ln(1 + u * 1[same chunk] + w * S[q_s, q_{t+1}])       ACT-R associative prior
      lag(t, s) = t + 1 - s  (step)   or   (tau_{t+1} - tau_s) / time_scale + 1   (decay_time = time)
      softmax over s of exp(score) = strength / sum of strengths, so each head is an ACT-R retrieval with
      its own decay rate d_h = softplus(.) > 0 (initialised at different values: short / long memory).
      memory.indiv_d: d_{h,t} = softplus(d_h_raw + w_h . m_t), m_t = causal mean of the learner's past
      interaction values (learner- and stage-specific forgetting rate).
      same chunk = same question, or same knowledge concept when concepts are available.

      x^d_t = LN(v_t) + LN(W_o [retrieval_1 .. retrieval_H])                 current interaction + retrieval

    It also returns the ACT-R base-level activation of the target's chunk (used by the recall head):
      A_t = ln(1 + sum_{s<=t, same chunk} lag(t, s)^(-d_c)),   d_c = softplus(.) (memory.forgetting)
    """

    def __init__(self, dim, S, mcfg):
        super().__init__(dim, S, mcfg)
        self.heads = int(mcfg.get("heads", 4))
        if dim % self.heads:
            raise ValueError("emb_dim must be divisible by memory.heads")
        self.dk = dim // self.heads
        self.prior = bool(mcfg.get("retrieval_prior", True))
        self.indiv_d = bool(mcfg.get("indiv_d", False))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.norm_base = nn.LayerNorm(dim)
        init = torch.tensor([_inv_softplus(x) for x in torch.logspace(-1, 0, self.heads).tolist()])
        self.raw_dh = nn.Parameter(init, requires_grad=bool(mcfg["learn_d"]))
        self.raw_u = nn.Parameter(torch.tensor(_inv_softplus(1.0)))
        if self.indiv_d:
            self.d_mod = nn.Linear(dim, self.heads)
            nn.init.zeros_(self.d_mod.weight)
            nn.init.zeros_(self.d_mod.bias)

    def head_decay(self):
        return F.softplus(self.raw_dh) if self.forgetting else self.raw_dh.new_zeros(self.heads)

    def forward(self, qid, correct, h_struct, times=None, hist=None, kc=None):
        if not self.individual:
            return self._forward_global(qid, h_struct)
        B, T = qid.shape
        dev = qid.device
        valid = qid >= 0
        safe = qid.clamp(min=0)
        if hist is not None:
            hq, hr, ht, hc = hist
            kq, kr = torch.cat([hq, qid], 1), torch.cat([hr, correct], 1)
            kt = torch.cat([ht, times], 1) if times is not None else None
            kk = torch.cat([hc, kc], 1) if kc is not None else None
        else:
            kq, kr, kt, kk = qid, correct, times, kc
        K = kq.shape[1]
        H = K - T
        kvalid = kq >= 0
        ksafe, kans = kq.clamp(min=0), kr.clamp(min=0)
        tq = torch.cat([safe[:, 1:], safe[:, -1:]], 1)                     # target of row t = q_{t+1}
        tvalid = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], 1)

        # lag to the retrieval moment, >= 1
        if self.decay_time == "time":
            if times is None:
                raise ValueError("decay_time='time' requires timestamps (set data.t_line)")
            t_ret = torch.cat([times[:, 1:], times[:, -1:]], 1)
            lag = (t_ret.unsqueeze(2) - kt.unsqueeze(1)).clamp(min=0) / self.time_scale + 1.0
        else:
            rows = torch.arange(T, device=dev, dtype=torch.float32) + H + 1.0
            cols = torch.arange(K, device=dev, dtype=torch.float32)
            lag = (rows.view(T, 1) - cols.view(1, K)).clamp(min=1.0).unsqueeze(0)
        loglag = torch.log(lag)                                            # [B or 1, T, K]

        same = tq.unsqueeze(2) == ksafe.unsqueeze(1)                        # [B, T, K]
        if kc is not None:
            tc = torch.cat([kc[:, 1:], kc[:, -1:]], 1).clamp(min=0)
            kks = kk.clamp(min=0)
            same = same | ((tc.unsqueeze(2) == kks.unsqueeze(1)) & (tc.unsqueeze(2) > 0))
        samef = same.float()
        causal = (torch.arange(K, device=dev).view(1, K) <= (torch.arange(T, device=dev) + H).view(T, 1))
        mask = causal.unsqueeze(0) & kvalid.unsqueeze(1)                    # [B, T, K]

        # values / keys / queries
        hk = h_struct[ksafe]
        val = hk + self.ans_emb(kans) if self.answer_aware else hk
        v = self.value_proj(val)                                            # [B, K, E]
        Qh = self.q_proj(h_struct[tq]).view(B, T, self.heads, self.dk)
        Kh = self.k_proj(hk).view(B, K, self.heads, self.dk)
        Vh = v.view(B, K, self.heads, self.dk)
        scores = torch.einsum("bthd,bshd->bhts", Qh, Kh) / math.sqrt(self.dk)
        d = self.head_decay().view(1, self.heads, 1, 1)
        if self.indiv_d:
            m = (v * kvalid.unsqueeze(-1).float()).cumsum(1)
            cnt = kvalid.float().cumsum(1).clamp(min=1.0).unsqueeze(-1)
            m_rows = (m / cnt)[:, H:]                                        # causal mean up to row t
            mod = self.d_mod(m_rows)                                        # [B, T, heads]
            d = F.softplus(self.raw_dh.view(1, 1, -1) + mod).permute(0, 2, 1).unsqueeze(-1) \
                if self.forgetting else d
        scores = scores - d * loglag.unsqueeze(1)
        if self.prior:
            pri = F.softplus(self.raw_u) * samef
            if self.spreading:
                pri = pri + self.w_value() * self.lookup_S(ksafe.unsqueeze(1), tq.unsqueeze(2)) * (1.0 - samef)
            scores = scores + torch.log1p(pri).unsqueeze(1)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = self.drop(torch.softmax(scores, dim=-1))
        retr = torch.einsum("bhts,bshd->bthd", attn, Vh).reshape(B, T, self.dim)
        x_d = (self.norm_base(v[:, H:]) + self.norm(self.o_proj(retr))) * valid.unsqueeze(-1).float()

        # ACT-R base-level activation of the target's chunk (for the recall head)
        a = torch.exp(-self.d_value() * loglag) * samef * mask.float()
        activation = torch.log1p(a.sum(-1)) * tvalid.float()                # [B, T]
        # success / failure split (PFA-style): A+ from correct, A- from incorrect past practice of the chunk
        kc_ok = kans.float().unsqueeze(1)                                   # [B, 1, K]
        act_pos = torch.log1p((a * kc_ok).sum(-1)) * tvalid.float()
        act_neg = torch.log1p((a * (1.0 - kc_ok)).sum(-1)) * tvalid.float()
        return x_d, {"activation": activation, "act_pos": act_pos, "act_neg": act_neg,
                     "gate_mean": torch.full((B, T), float("nan"), device=dev)}
