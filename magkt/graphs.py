"""Structural priors (Sec. 3.2.2 and 3.2.3). Everything is kept sparse so that datasets
with many questions (e.g. ASSIST2009 with ~17k problems) fit in memory.

* Question-concept hypergraph H (2Q x M): row q = correct response to question q,
  row Q+q = incorrect response. With the DGEKT construction every concept c is split
  into a "correct" hyperedge (column c) and an "incorrect" hyperedge (column C+c).
  The operator G = Dv^-1/2 H W De^-1 H^T Dv^-1/2 is never formed: it is applied as
  G Z = L (R Z) with L = Dv^-1/2 H W De^-1 (2Q x M) and R = H^T Dv^-1/2 (M x 2Q).
* Global question transition graph: counted ONLY on training learners (validation
  learners excluded), sparsified per row (top-k), self loops added, row normalised.
  The out-edge matrix is also the transition affinity S used for memory diffusion.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import data_path


# ------------------------------------------------------------- hypergraph
def load_incidence(cfg):
    d = cfg["data"]
    H = pd.read_csv(data_path(cfg, d["hypergraph_file"]), header=None).values.astype(np.float64)
    Q = d["num_questions"]
    if H.shape[0] != 2 * Q:
        raise ValueError(f"H has {H.shape[0]} rows, expected 2Q = {2 * Q}")
    return H


def _safe_pow(x, p):
    out = np.zeros_like(x, dtype=np.float64)
    nz = x > 0
    out[nz] = np.power(x[nz], p)
    return out


def hypergraph_factors(H, norm="hgnn", edge_weight=None):
    """Return (L, R) with G = L @ R.

    hgnn   : G = Dv^-1/2 H W De^-1 H^T Dv^-1/2   (Eq. 4)
    legacy : G = Dv^-1/2 H W H^T Dv^-1/2         (without De^-1; aggregation-only ablation)
    Nodes with no hyperedge get a zero row.
    """
    Hs = sp.csr_matrix(H)
    w = np.ones(H.shape[1]) if edge_weight is None else np.asarray(edge_weight, dtype=np.float64)
    dv = np.asarray(Hs.multiply(w[None, :]).sum(axis=1)).ravel()
    de = np.asarray(Hs.sum(axis=0)).ravel()
    Dv = sp.diags(_safe_pow(dv, -0.5))
    if norm == "hgnn":
        mid = sp.diags(w * _safe_pow(de, -1.0))
    elif norm == "legacy":
        mid = sp.diags(w)
    else:
        raise ValueError(norm)
    L = sp.csr_matrix(Dv @ Hs @ mid)
    R = sp.csr_matrix(Hs.T @ Dv)
    return L, R


def hypergraph_operator(H, norm="hgnn", edge_weight=None):
    """Explicit G (only for small H, e.g. tests)."""
    L, R = hypergraph_factors(H, norm, edge_weight)
    return sp.csr_matrix(L @ R)


def concept_rows(H, Q):
    """Binary question x concept-column matrix (correct-response rows)."""
    return sp.csr_matrix((H[:Q] > 0).astype(np.float32))


def share_pairs(Hq, i, j):
    """Boolean array: do questions i[k] and j[k] share at least one concept column?"""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    if len(i) == 0:
        return np.zeros(0, dtype=bool)
    out = np.empty(len(i), dtype=bool)
    step = 500000
    for s in range(0, len(i), step):
        a, b = Hq[i[s:s + step]], Hq[j[s:s + step]]
        out[s:s + step] = np.asarray(a.multiply(b).sum(axis=1)).ravel() > 0
    return out


def share_counts(Hq):
    """For every question, the number of OTHER questions sharing >= 1 concept with it."""
    Q = Hq.shape[0]
    Hc = Hq.tocsc()
    members = [Hc.indices[Hc.indptr[c]:Hc.indptr[c + 1]] for c in range(Hc.shape[1])]
    out = np.zeros(Q, dtype=np.int64)
    cache = {}
    for qi in range(Q):
        cols = tuple(Hq.indices[Hq.indptr[qi]:Hq.indptr[qi + 1]].tolist())
        if not cols:
            continue
        if cols not in cache:
            mask = np.zeros(Q, dtype=bool)
            for c in cols:
                mask[members[c]] = True
            cache[cols] = int(mask.sum())
        out[qi] = cache[cols] - 1
    return out


# ------------------------------------------------------- transition graph
def count_transitions(learners, Q):
    """Sparse CSR matrix F[i, j] = number of times question j directly follows question i."""
    src, dst = [], []
    for s in learners:
        q = np.asarray(s["q"])
        if len(q) > 1:
            src.append(q[:-1]); dst.append(q[1:])
    if not src:
        return sp.csr_matrix((Q, Q), dtype=np.float64)
    src = np.concatenate(src); dst = np.concatenate(dst)
    F = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(Q, Q)).tocsr()
    F.sum_duplicates()
    return F


def drop_diagonal(M):
    M = sp.coo_matrix(M)
    keep = M.row != M.col
    return sp.csr_matrix((M.data[keep], (M.row[keep], M.col[keep])), shape=M.shape)


def resolve_topk(rule, Q):
    if rule == "auto":           # Q >= 2000 -> 50, 500 < Q < 2000 -> 30, otherwise no sparsification
        if Q > 500:
            return 50 if Q >= 2000 else 30
        return 0
    return int(rule)


def topk_rows(M, k):
    """Keep the k largest entries of every row (ties broken by column index)."""
    M = sp.csr_matrix(M)
    if k <= 0:
        return M.copy()
    rows, cols, vals = [], [], []
    for i in range(M.shape[0]):
        a, b = M.indptr[i], M.indptr[i + 1]
        idx, dat = M.indices[a:b], M.data[a:b]
        if len(idx) > k:
            order = np.argsort(idx, kind="stable")            # column order, like np.nonzero
            idx, dat = idx[order], dat[order]
            keep = np.argsort(dat, kind="stable")[-k:]
            idx, dat = idx[keep], dat[keep]
        rows.append(np.full(len(idx), i)); cols.append(idx); vals.append(dat)
    rows = np.concatenate(rows) if rows else np.zeros(0, int)
    return sp.csr_matrix((np.concatenate(vals), (rows, np.concatenate(cols))), shape=M.shape)


def concept_graph_topk(H, Q, k, chunk=2000):
    """Question-question graph by number of shared concepts, top-k per row, built in row chunks."""
    Hq = concept_rows(H, Q)
    parts = []
    for s in range(0, Q, chunk):
        C = (Hq[s:s + chunk] @ Hq.T).tocoo()
        keep = C.col != C.row + s
        C = sp.csr_matrix((C.data[keep], (C.row[keep], C.col[keep])), shape=C.shape)
        parts.append(topk_rows(C, k))
    return sp.vstack(parts).tocsr()


def row_normalize(M):
    M = sp.csr_matrix(M, dtype=np.float64)
    inv = _safe_pow(np.asarray(M.sum(axis=1)).ravel(), -1.0)
    return sp.csr_matrix(sp.diags(inv) @ M)


def transition_graphs(cfg, train_learners, H=None):
    """Return (A_out, A_in, S, info) as scipy CSR float32 matrices of shape (Q, Q).

    A = D^-1 (A_topk + I) (Eq. 6). A_in is built by sparsifying the transposed counts
    (in-neighbours ranked by count). S = A_out.
    """
    Q = cfg["data"]["num_questions"]
    src = cfg["graph"]["transition_source"]
    k = resolve_topk(cfg["graph"]["transition_topk"], Q)
    if src == "train":
        F = count_transitions(train_learners, Q)
        Fo, Fi = topk_rows(F, k), topk_rows(F.T.tocsr(), k)
    elif src == "identity":
        F = sp.csr_matrix((Q, Q))
        Fo = Fi = F
    elif src == "concept":
        if H is None:
            raise ValueError("concept graph needs H")
        Fo = concept_graph_topk(H, Q, k)
        Fi = Fo                                   # symmetric, so in- and out-neighbours coincide
        F = Fo
    else:
        raise ValueError(src)
    I = sp.identity(Q, format="csr")
    A_out = row_normalize(Fo + I).astype(np.float32)
    A_in = row_normalize(Fi + I).astype(np.float32)
    return A_out, A_in, A_out, {"topk": k, "source": src, "nonzero_counts": int(F.nnz),
                                "edges_kept": int(Fo.nnz)}


def lookup(M, i, j):
    """Values M[i[k], j[k]] of a scipy sparse matrix as a dense array."""
    M = sp.csr_matrix(M)
    return np.asarray(M[np.asarray(i), np.asarray(j)]).ravel()


# ------------------------------------------------------------ torch utils
def to_torch_sparse(M):
    import torch
    M = sp.coo_matrix(M).astype(np.float32)
    idx = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    return torch.sparse_coo_tensor(idx, torch.from_numpy(M.data), M.shape).coalesce()
