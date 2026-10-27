"""MAGKT model (Section 3 of the paper). Row-vector convention: x W.

  response-state node ................ n_t = q_t + Q * (1 - r_t)        (correct -> q, incorrect -> q + Q)
  structural branch .................. Z_hat = StructEncoder(Z, G);  x^h_t = Z_hat[n_t]          (Sec. 3.3)
  question embeddings for memory ..... Z_q = Z[0:Q]  (embeddings of the correct-response nodes)
  directed GCN ....................... h = [GCN_in(Z_q, A_in) || GCN_out(Z_q, A_out)]            (Sec. 3.4.1)
  memory branch ...................... x^d, A+, A- = RetrievalMemory(q, r, h, c)                  (Sec. 3.4)
  concept-response embedding ......... x^* += K[c_t, r_t]                       (model.kc_input, Sec. 3.5)
  cross-graph gate ................... g = sigmoid(x^h U_h + x^d U_d + b);  x^f = g*x^h + (1-g)*x^d  (Sec. 3.5)
  shared LSTM ........................ O^* = LSTM(x^*), * in {h, d, f}                              (Sec. 3.6.1)
  recall term ........................ rho_t = gamma_c A+ + rho_c A- - tau_c,  c = c_{t+1}
                                       (beta+ = gamma_c, beta- = rho_c in the paper)                   (Sec. 3.6.2)
  output ............................. ensemble: z = (z^h + z^d + z^f)/3 ;  fusion: z = z^f

Only the logit of the next question is needed, so forward() returns, for every position t, z_{t, q_{t+1}}
(shape [B, T]; the last position is unused), without materialising the [B, T, Q] output.

Output layer (model.head_type):
  dot  : z = O_t . (P[q_{t+1}] + P^c[c_{t+1}]) + b[q_{t+1}] + b^c[c_{t+1}]   (concept term with model.kc_head)
  mlp  : z = MLP([O_t ; Z_hat[q_{t+1}] + K^q[c_{t+1}]])                       (concept term with model.kc_head)
  rasch: z = O_t . (P^c[c] + mu_q * V^c[c]) + b^c[c] + beta_q,  c = c_{t+1}, q = q_{t+1}
         (alternative considered during the selection of settings; not used in the reported experiments)
The concept c_{t+1} of the next question is known at prediction time, like q_{t+1} itself.

Branch ablations: without the memory branch x^d := x^h, without the structural branch x^h := x^d
(three paths remain).

memory.type = activation selects ActivationMemory, an activation-weighted aggregation layer kept as an
alternative; it is not used in the reported experiments.
"""
import torch
import torch.nn as nn

from .encoders import GCN, StructEncoderLegacy, StructEncoderSAGE
from .memory import ActivationMemory, RetrievalMemory


class CrossGraphGate(nn.Module):
    def __init__(self, dim, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.U_h = nn.Linear(dim, dim)
        self.U_d = nn.Linear(dim, dim)

    def forward(self, xh, xd):
        if not self.enabled:
            return 0.5 * (xh + xd)
        gamma = torch.sigmoid(self.U_h(xh) + self.U_d(xd))
        return gamma * xh + (1.0 - gamma) * xd


class MAGKT(nn.Module):
    def __init__(self, cfg, G=None, A_out=None, A_in=None, S=None):
        super().__init__()
        m = cfg["model"]
        self.Q = cfg["data"]["num_questions"]
        E, Hd = m["emb_dim"], m["hidden_dim"]
        self.use_static = m["use_static"]
        self.use_memory = m["use_memory"]
        self.head_mode = m["head_mode"]

        self.embedding = nn.Embedding(2 * self.Q, E)
        if self.use_static:
            enc = StructEncoderSAGE if m["struct_encoder"] == "sage" else StructEncoderLegacy
            self.struct_enc = enc(E, dropout=m["struct_dropout"])
            G_L, G_R = G if isinstance(G, (tuple, list)) else (G, None)
            self.register_buffer("G_L", G_L, persistent=False)
            self.register_buffer("G_R", G_R, persistent=False)
        if self.use_memory:
            self.gcn_out = GCN(E, E, E // 2)
            self.gcn_in = GCN(E, E, E // 2)
            self.register_buffer("A_out", A_out, persistent=False)
            self.register_buffer("A_in", A_in, persistent=False)
            mtype = cfg["memory"].get("type", "activation")
            self.memory = (RetrievalMemory if mtype == "retrieval" else ActivationMemory)(E, S, cfg["memory"])
            self.recall_head = bool(cfg["memory"].get("recall_head", False))
            if self.recall_head:
                if mtype != "retrieval" or not cfg["memory"]["individual"]:
                    raise ValueError("memory.recall_head needs memory.type = retrieval and memory.individual = true")
                C = int(cfg["data"].get("num_concepts") or 0)
                self.tau_c = nn.Embedding(C + 1, 1)          # threshold tau_c per concept (index 0 = no concept)
                nn.init.zeros_(self.tau_c.weight)
                self.raw_s = nn.Parameter(torch.tensor(0.5413))   # softplus(0.5413) = 1: activation noise s
                self.recall_split = bool(cfg["memory"].get("recall_split", False))
                if self.recall_split:
                    # PFA-style: z += gamma_c * A+ + rho_c * A- - tau_c  (per concept, initialised at 0)
                    self.gamma_c = nn.Embedding(C + 1, 1)
                    self.rho_c = nn.Embedding(C + 1, 1)
                    nn.init.zeros_(self.gamma_c.weight)
                    nn.init.zeros_(self.rho_c.weight)
            self.act_head = bool(cfg["memory"].get("act_head", False))
            if self.act_head:
                if cfg["memory"].get("target", "current") != "next" or not cfg["memory"]["individual"]:
                    raise ValueError("memory.act_head needs memory.target = next and memory.individual = true")
                self.act_beta = nn.Parameter(torch.zeros(3))      # one scalar per path, z += beta * A
        if not self.use_memory:
            self.act_head = False
            self.recall_head = False
        if not (self.use_static or self.use_memory):
            raise ValueError("at least one branch must be enabled")

        self.cross_gate = CrossGraphGate(E, enabled=m["cross_gate"])
        self.lstm = nn.LSTM(E, Hd, m["lstm_layers"], batch_first=True)
        if m.get("head_type", "dot") == "dot":
            self.P_h = nn.Linear(Hd, self.Q)
            self.P_d = nn.Linear(Hd, self.Q)
            self.P_f = nn.Linear(Hd, self.Q)
        self.in_drop = nn.Dropout(m.get("input_dropout", 0.0))
        self.id_dropout = float(m.get("id_dropout", 0.0))
        self.head_type = m.get("head_type", "dot")
        if self.head_type == "mlp":
            hd = float(m.get("head_dropout", 0.2))
            self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(Hd + E, Hd), nn.ReLU(), nn.Dropout(hd),
                                                    nn.Linear(Hd, 1)) for _ in range(3)])

        self.kc_input = bool(m.get("kc_input", False))
        self.kc_head = bool(m.get("kc_head", False))
        self.C = int(cfg["data"].get("num_concepts") or 0)
        if (self.kc_input or self.kc_head) and self.C <= 0:
            raise ValueError("kc_input / kc_head need per-interaction concepts: set data.c_line in the dataset yaml")
        if self.kc_input:
            self.kc_emb = nn.Embedding(2 * self.C + 1, E, padding_idx=0)
        if self.head_type == "rasch":
            if self.C <= 0:
                raise ValueError("head_type = rasch needs per-interaction concepts (data.c_line)")
            self.R_w = nn.ModuleList([nn.Embedding(self.C + 1, Hd) for _ in range(3)])
            self.R_v = nn.ModuleList([nn.Embedding(self.C + 1, Hd) for _ in range(3)])
            self.R_b = nn.ModuleList([nn.Embedding(self.C + 1, 1) for _ in range(3)])
            self.R_mu = nn.ModuleList([nn.Embedding(self.Q, 1) for _ in range(3)])
            self.R_beta = nn.ModuleList([nn.Embedding(self.Q, 1) for _ in range(3)])
            for e in list(self.R_v) + list(self.R_b) + list(self.R_mu) + list(self.R_beta):
                nn.init.zeros_(e.weight)
            for e in self.R_w:
                nn.init.normal_(e.weight, std=0.05)
        if self.head_type == "mlp" and self.kc_head:
            self.kc_query = nn.Embedding(self.C + 1, E, padding_idx=0)
        if self.kc_head and self.head_type == "dot":
            self.PC_w = nn.ModuleList([nn.Embedding(self.C + 1, Hd, padding_idx=0) for _ in range(3)])
            self.PC_b = nn.ModuleList([nn.Embedding(self.C + 1, 1, padding_idx=0) for _ in range(3)])
            for e in list(self.PC_w) + list(self.PC_b):
                nn.init.zeros_(e.weight)

    # ---------------------------------------------------------------
    def node_embedding(self):
        """Free node embeddings; with id_dropout, whole rows are zeroed during training so that
        the model also learns to predict from the shared (concept / structural) information."""
        Z = self.embedding.weight
        if self.training and self.id_dropout > 0:
            keep = (torch.rand(Z.shape[0], 1, device=Z.device) >= self.id_dropout).to(Z.dtype)
            Z = Z * keep
        return Z

    def struct_node_repr(self, Z=None):
        """Z_hat: [2Q, E] structural node representations."""
        Z = self.embedding.weight if Z is None else Z
        G = self.G_L if self.G_R is None else (self.G_L, self.G_R)
        return self.struct_enc(Z, G)

    def question_struct_embedding(self, Z=None):
        Zq = (self.embedding.weight if Z is None else Z)[: self.Q]
        return torch.cat([self.gcn_in(Zq, self.A_in), self.gcn_out(Zq, self.A_out)], dim=-1)

    def id_parameters(self):
        """Question-specific parameters (one row per question / node)."""
        ps = [self.embedding.weight]
        for name in ("P_h", "P_d", "P_f"):
            if hasattr(self, name):
                ps += [getattr(self, name).weight, getattr(self, name).bias]
        if self.head_type == "rasch":
            ps += [e.weight for e in list(self.R_mu) + list(self.R_beta)]
        return ps

    def diagnostics(self):
        if not self.use_memory:
            return {"d": float("nan"), "w": float("nan")}
        out = {"d": float(self.memory.d_value()), "w": float(self.memory.w_value())}
        if self.act_head:
            out["act_beta"] = float(self.act_beta.mean())
        if isinstance(self.memory, RetrievalMemory):
            out["d_heads"] = float(self.memory.head_decay().mean())
        if self.recall_head and not self.recall_split:
            out["recall_s"] = float(torch.nn.functional.softplus(self.raw_s))
        if self.recall_head and self.recall_split:
            out["gamma"] = float(self.gamma_c.weight.mean())
            out["rho"] = float(self.rho_c.weight.mean())
        return out

    @staticmethod
    def _head(P, O, idx):
        """(O P)[b, t, idx[b, t]] without computing all Q logits."""
        return (O * P.weight[idx]).sum(-1) + P.bias[idx]

    def forward(self, q, r, times=None, kc=None, hist=None):
        valid = (q >= 0).unsqueeze(-1).float()
        xh = xd = None
        aux = {}
        Zhat = None
        Z = self.node_embedding()
        if self.use_static:
            node = (q.clamp(min=0) + self.Q * (1 - r.clamp(min=0)))
            Zhat = self.struct_node_repr(Z)
            xh = Zhat[node] * valid
        if self.use_memory:
            xd, aux = self.memory(q, r, self.question_struct_embedding(Z), times, hist=hist, kc=kc)
        if self.kc_input:
            c = kc.clamp(min=0)
            kidx = torch.where(c > 0, c + self.C * (1 - r.clamp(min=0)), torch.zeros_like(c))
            ek = self.kc_emb(kidx) * valid
            xh = None if xh is None else xh + ek
            xd = None if xd is None else xd + ek
        if xh is None:
            xh = xd
        if xd is None:
            xd = xh
        xh, xd = self.in_drop(xh), self.in_drop(xd)
        xf = self.cross_gate(xh, xd)

        nxt = torch.cat([q[:, 1:], q[:, -1:]], dim=1).clamp(min=0)     # q_{t+1} (last position: dummy)
        cn = torch.cat([kc[:, 1:], kc[:, -1:]], dim=1).clamp(min=0) if kc is not None else None  # c_{t+1}
        Oh, Od, Of = self.lstm(xh)[0], self.lstm(xd)[0], self.lstm(xf)[0]
        if self.head_type == "rasch":
            zs = []
            for i, O in enumerate((Oh, Od, Of)):
                w = self.R_w[i](cn) + self.R_mu[i](nxt) * self.R_v[i](cn)
                zs.append((O * w).sum(-1) + self.R_b[i](cn).squeeze(-1) + self.R_beta[i](nxt).squeeze(-1))
            zh, zd, zf = zs
        elif self.head_type == "mlp":
            base = Zhat[: self.Q] if Zhat is not None else Z[: self.Q]
            u = base[nxt]
            if self.kc_head:
                u = u + self.kc_query(cn)
            zh, zd, zf = [self.mlp[i](torch.cat([O, u], dim=-1)).squeeze(-1)
                          for i, O in enumerate((Oh, Od, Of))]
        else:
            zh = self._head(self.P_h, Oh, nxt)
            zd = self._head(self.P_d, Od, nxt)
            zf = self._head(self.P_f, Of, nxt)
            if self.kc_head:
                zh = zh + (Oh * self.PC_w[0](cn)).sum(-1) + self.PC_b[0](cn).squeeze(-1)
                zd = zd + (Od * self.PC_w[1](cn)).sum(-1) + self.PC_b[1](cn).squeeze(-1)
                zf = zf + (Of * self.PC_w[2](cn)).sum(-1) + self.PC_b[2](cn).squeeze(-1)
        if self.act_head:                     # ACT-R: recall probability increases with the target's activation
            A = aux["activation"]
            zh, zd, zf = zh + self.act_beta[0] * A, zd + self.act_beta[1] * A, zf + self.act_beta[2] * A
        if self.recall_head:                  # ACT-R recall probability: sigma((A - tau) / s)
            cidx = cn if cn is not None else torch.zeros_like(nxt)
            if self.recall_split:
                rec = (self.gamma_c(cidx).squeeze(-1) * aux["act_pos"] + self.rho_c(cidx).squeeze(-1) * aux["act_neg"]
                       - self.tau_c(cidx).squeeze(-1))
            else:
                rec = (aux["activation"] - self.tau_c(cidx).squeeze(-1)) / torch.nn.functional.softplus(self.raw_s)
            zh, zd, zf = zh + rec, zd + rec, zf + rec
        final = (zh + zd + zf) / 3.0 if self.head_mode == "ensemble" else zf
        return {"h": zh, "d": zd, "f": zf, "final": final, "aux": aux}
