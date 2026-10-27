"""Data reading, learner-level train/validation split, and windowing.

Sequence file format: each learner is a block of
`n_lines` lines -- sequence length, comma-separated question ids, (optional
extra lines), comma-separated responses (1 = correct, 0 = incorrect).

Protocol (Section 4.2 of the paper):
  * The train/test files are split by LEARNER (no learner appears in both).
  * A fixed fraction of TRAINING learners (val_ratio, split_seed) is held out
    for validation; early stopping and checkpoint selection use only it.
  * Each learner sequence is cut into non-overlapping windows of `max_len`
    steps; the last window is padded with -1. Windows are modelled
    independently: memory history and LSTM state are reset at each window.
"""
import os

import numpy as np

from .config import data_path


def read_sequences(path, n_lines, q_line, a_line, t_line=None, qid_offset=1, c_line=None):
    """Return a list of dicts {'q','r','t','c'} with 0-based question ids.
    'c' (optional) is the concept id of each interaction, 1-based, 0 = none."""
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if len(lines) % n_lines != 0:
        raise ValueError(f"{path}: {len(lines)} non-empty lines is not a multiple of {n_lines}")

    def parse(s, dtype):
        return np.array([x for x in s.strip(",").split(",") if x != ""], dtype=float).astype(dtype)

    seqs = []
    for i in range(0, len(lines), n_lines):
        block = lines[i:i + n_lines]
        length = int(float(block[0].strip(",")))
        q = parse(block[q_line], np.int64) - qid_offset
        r = (parse(block[a_line], np.float64) > 0).astype(np.int64)
        L = min(length, len(q), len(r))
        seq = {"q": q[:L], "r": r[:L], "t": None, "c": None}
        if t_line is not None:
            seq["t"] = parse(block[t_line], np.float64)[:L]
        if c_line is not None:
            seq["c"] = np.maximum(parse(block[c_line], np.int64)[:L], 0)
        seqs.append(seq)
    return seqs


def drop_invalid(seqs, Q, mode="error", name=""):
    """Handle question ids outside [0, Q). mode: error | drop (removes those interactions)."""
    n_bad = 0
    out = []
    for s in seqs:
        ok = (s["q"] >= 0) & (s["q"] < Q)
        if ok.all():
            out.append(s); continue
        n_bad += int((~ok).sum())
        if mode != "drop":
            continue
        t = None if s["t"] is None else s["t"][ok]
        c = None if s.get("c") is None else s["c"][ok]
        out.append(dict(s, q=s["q"][ok], r=s["r"][ok], t=t, c=c))
    if n_bad and mode != "drop":
        raise ValueError(f"{name}: {n_bad} interactions have question ids outside [0, {Q}). "
                         f"Check qid_offset / invalid_qid in the dataset yaml.")
    if n_bad:
        total = sum(len(s["q"]) for s in seqs)
        print(f"[{name}] removed {n_bad} of {total} interactions ({100 * n_bad / total:.3f}%) "
              f"with question ids outside [0, {Q})")
    return out


def raw_qid_report(paths, n_lines, q_line, Q):
    """Raw (un-shifted) question-id statistics over the given files."""
    ids = []
    for p in paths:
        for s in read_sequences(p, n_lines, q_line, q_line, None, qid_offset=0):
            ids.append(s["q"])
    ids = np.concatenate(ids)
    u, c = np.unique(ids, return_counts=True)
    cnt = dict(zip(u.tolist(), c.tolist()))
    rep = {"min": int(ids.min()), "max": int(ids.max()), "n_distinct": int(len(u)), "n_total": int(len(ids)),
           "count_0": int(cnt.get(0, 0)), "count_Q": int(cnt.get(Q, 0)), "Q": Q}
    if rep["min"] >= 1 and rep["max"] <= Q:
        rep["verdict"], rep["advice"] = "ok", "question ids start at 1; qid_offset: 1 is correct"
    elif rep["min"] == 0 and rep["max"] <= Q - 1:
        rep["verdict"] = "zero_based"
        rep["advice"] = ("question ids start at 0 (range 0..Q-1): set qid_offset: 0 in the dataset yaml; "
                         "with qid_offset: 1, id 0 would be mapped to question Q-1 with the response states swapped")
    elif rep["min"] == 0 and rep["max"] == Q:
        rep["verdict"] = "one_based_with_zeros"
        rep["advice"] = (f"question ids are 1..Q plus {rep['count_0']} records with id 0 "
                         f"({100 * rep['count_0'] / rep['n_total']:.3f}%), which is not a valid id: "
                         "set invalid_qid: drop in the dataset yaml to remove them")
    else:
        rep["verdict"] = "mismatch"
        rep["advice"] = "the range of question ids does not match num_questions; check num_questions and the rows of H"
    return rep


def split_learners(seqs, val_ratio, split_seed):
    rng = np.random.RandomState(split_seed)
    idx = rng.permutation(len(seqs))
    n_val = int(round(val_ratio * len(seqs)))
    val_idx = set(idx[:n_val].tolist())
    train = [s for i, s in enumerate(seqs) if i not in val_idx]
    val = [s for i, s in enumerate(seqs) if i in val_idx]
    return train, val


def windowize(seqs, max_len, history=0):
    """Cut sequences into windows of max_len; pad with -1 (q, r) and 0 (t, c).

    history > 0 (memory.history): for every window also return the `history` interactions of the same
    learner that precede the window ("hq", "hr", "ht"; right-aligned, -1 / 0 padding on the left). They are
    used only as memory history (the LSTM still starts at the window), so that the memory activation can
    see practice outside the current window. Times are relative to the window start (prefix < 0)."""
    qs, rs, ts, cs = [], [], [], []
    hqs, hrs, hts, hcs = [], [], [], []
    has_t = any(s["t"] is not None for s in seqs)
    has_c = any(s.get("c") is not None for s in seqs)
    H = int(history or 0)
    for s in seqs:
        L = len(s["q"])
        for start in range(0, L, max_len):
            q = np.full(max_len, -1, dtype=np.int64)
            r = np.full(max_len, -1, dtype=np.int64)
            t = np.zeros(max_len, dtype=np.float64)
            c = np.zeros(max_len, dtype=np.int64)
            seg = slice(start, min(start + max_len, L))
            n = seg.stop - seg.start
            q[:n] = s["q"][seg]
            r[:n] = s["r"][seg]
            if has_t:
                t[:n] = s["t"][seg] - s["t"][seg.start]   # relative time inside the window
            if has_c:
                c[:n] = s["c"][seg]
            qs.append(q); rs.append(r); ts.append(t); cs.append(c)
            if H > 0:
                hq = np.full(H, -1, dtype=np.int64)
                hr = np.full(H, -1, dtype=np.int64)
                ht = np.zeros(H, dtype=np.float64)
                hc = np.zeros(H, dtype=np.int64)
                lo = max(0, start - H)
                m = start - lo
                if m > 0:
                    hq[H - m:] = s["q"][lo:start]
                    hr[H - m:] = s["r"][lo:start]
                    if has_t:
                        ht[H - m:] = s["t"][lo:start] - s["t"][start]
                    if has_c:
                        hc[H - m:] = s["c"][lo:start]
                hqs.append(hq); hrs.append(hr); hts.append(ht); hcs.append(hc)
    out = {"q": np.stack(qs) if qs else np.zeros((0, max_len), np.int64),
           "r": np.stack(rs) if rs else np.zeros((0, max_len), np.int64)}
    if has_t:
        out["t"] = np.stack(ts)
    if has_c:
        out["c"] = np.stack(cs)
    if H > 0:
        out["hq"] = np.stack(hqs) if hqs else np.zeros((0, H), np.int64)
        out["hr"] = np.stack(hrs) if hrs else np.zeros((0, H), np.int64)
        out["ht"] = np.stack(hts) if hts else np.zeros((0, H))
        out["hc"] = np.stack(hcs) if hcs else np.zeros((0, H), np.int64)
    return out


def load_splits(cfg):
    """Return dict with raw learner lists and windowed arrays for train/val/test."""
    d = cfg["data"]
    kw = dict(n_lines=d["n_lines"], q_line=d["q_line"], a_line=d["a_line"],
              t_line=d["t_line"], qid_offset=d["qid_offset"], c_line=d.get("c_line"))
    Q = d["num_questions"]
    mode = d.get("invalid_qid", "error")
    train_all = drop_invalid(read_sequences(data_path(cfg, d["train_file"]), **kw), Q, mode, "train file")
    test = drop_invalid(read_sequences(data_path(cfg, d["test_file"]), **kw), Q, mode, "test file")
    train, val = split_learners(train_all, d["val_ratio"], d["split_seed"])
    return {
        "learners": {"train": train, "val": val, "test": test},
        "windows": {k: windowize(v, d["max_len"], (cfg.get("memory") or {}).get("history", 0)) for k, v in
                    (("train", train), ("val", val), ("test", test))},
    }


def describe(splits):
    """Statistics for Table 1 / the data section (computed from the files actually used)."""
    out = {}
    for k, seqs in splits["learners"].items():
        n_int = int(sum(len(s["q"]) for s in seqs))
        qs = np.concatenate([s["q"] for s in seqs]) if seqs else np.array([])
        rs = np.concatenate([s["r"] for s in seqs]) if seqs else np.array([])
        out[k] = {"learners": len(seqs), "interactions": n_int,
                  "distinct_questions": int(len(np.unique(qs))),
                  "positive_rate": float(rs.mean()) if len(rs) else float("nan"),
                  "windows": int(len(splits["windows"][k]["q"]))}
    return out


# ---------------------------------------------------------------- torch part
def make_loader(windows, batch_size, shuffle, num_workers=0, drop_last=False):
    """Batches are (q, r, t, c) [+ (hq, hr, ht) with memory.history]; t and c are zeros without such lines."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    q = torch.as_tensor(windows["q"])
    t = torch.as_tensor(windows["t"], dtype=torch.float32) if "t" in windows else torch.zeros(q.shape)
    c = torch.as_tensor(windows["c"]) if "c" in windows else torch.zeros(q.shape, dtype=torch.long)
    tensors = [q, torch.as_tensor(windows["r"]), t, c]
    if "hq" in windows:
        tensors += [torch.as_tensor(windows["hq"]), torch.as_tensor(windows["hr"]),
                    torch.as_tensor(windows["ht"], dtype=torch.float32), torch.as_tensor(windows["hc"])]
    ds = TensorDataset(*tensors)
    drop_last = drop_last and len(ds) > batch_size   # training: drop the last incomplete batch
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=drop_last)
