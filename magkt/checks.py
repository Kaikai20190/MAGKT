"""Consistency checks of the processed data (used by step1_prepare_data.py).

h_column_blocks     the incorrect-response rows of H are the correct-response rows shifted by C columns
check_concept_line  every (question, concept) pair of the sequence files is an entry of H
Results are printed and saved to results/checks/.
"""
import json
import os

import numpy as np

from . import graphs as Gr
from .config import RESULTS_DIR, build_config, data_path

CHECK_DIR = os.path.join(RESULTS_DIR, "checks")


def _save_json(obj, name):
    os.makedirs(CHECK_DIR, exist_ok=True)
    path = os.path.join(CHECK_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def h_column_blocks(dataset):
    """Do the correct-response rows (0..Q-1) and the incorrect-response rows (Q..2Q-1) use separate hyperedge
    columns? Tests whether the second half equals the first half with the columns shifted by a constant k
    (hyperedge = (concept, response), 2C columns)."""
    cfg = build_config(dataset)
    Q = cfg["data"]["num_questions"]
    H = Gr.load_incidence(cfg)
    A, B = H[:Q] > 0, H[Q:] > 0
    M = H.shape[1]
    ca, cb = set(np.nonzero(A.any(0))[0]), set(np.nonzero(B.any(0))[0])
    best_k, best_f = 0, -1.0
    for k in range(-M + 1, M):
        f = float(np.mean(np.all(np.roll(A, k, axis=1) == B, axis=1)))
        if f > best_f:
            best_k, best_f = k, f
    res = {"dataset": dataset, "Q": Q, "M": M,
           "columns_used_by_correct_rows": len(ca), "columns_used_by_incorrect_rows": len(cb),
           "columns_shared": len(ca & cb),
           "correct_cols_range": [int(min(ca)), int(max(ca))] if ca else None,
           "incorrect_cols_range": [int(min(cb)), int(max(cb))] if cb else None,
           "best_column_shift": best_k, "rows_matching_at_best_shift": best_f}
    print(f"\n[{dataset}] H has {M} columns")
    print(f"  correct-response rows use {len(ca)} columns {res['correct_cols_range']}, incorrect-response rows "
          f"use {len(cb)} columns {res['incorrect_cols_range']}, shared: {len(ca & cb)}")
    print(f"  share of rows equal after shifting the correct-response rows by k = {best_k}: {best_f:.3f}")
    if best_f > 0.95 and best_k != 0:
        print(f"  => incorrect-response rows = correct-response rows shifted by {best_k} columns: hyperedges are "
              f"(concept, response) pairs with {len(ca)} concepts; the two node types share no hyperedge")
    elif len(ca & cb) == 0:
        print("  => disjoint columns, but not a simple shift")
    else:
        print("  => not a (concept, response) structure")
    _save_json(res, f"H_columns_{dataset}.json")
    return res


def check_concept_line(dataset, concept_line=2):
    """For the files written by step1_prepare_data.py: every interaction (q, c) with a concept must satisfy
    H[q, c] = 1 (correct-response hyperedge) and H[Q + q, C + c] = 1 (incorrect-response hyperedge), which
    verifies that the question ids and H come from the same mapping."""
    cfg = build_config(dataset)
    d = cfg["data"]
    Q = d["num_questions"]
    H = Gr.load_incidence(cfg) > 0
    C = H.shape[1] // 2
    res = {"dataset": dataset}
    for part in ("train_file", "test_file"):
        with open(data_path(cfg, d[part]), "r", encoding="utf-8-sig") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        qs, cs = [], []
        for i in range(0, len(lines), d["n_lines"]):
            qs.append(np.array(lines[i + d["q_line"]].split(","), dtype=np.int64))
            cs.append(np.array(lines[i + concept_line].split(","), dtype=np.int64))
        q = np.concatenate(qs) - d["qid_offset"]
        c = np.concatenate(cs)
        has = c > 0
        ok_c = H[q[has], c[has] - 1]
        ok_i = H[Q + q[has], C + c[has] - 1]
        res[part] = {"interactions": int(len(q)), "with_concept": int(has.sum()),
                     "match_correct_rows": float(ok_c.mean()) if has.any() else float("nan"),
                     "match_incorrect_rows": float(ok_i.mean()) if has.any() else float("nan"),
                     "qid_in_range": bool(q.min() >= 0 and q.max() < Q)}
    good = all(res[p]["qid_in_range"] and res[p]["match_correct_rows"] == 1.0 and res[p]["match_incorrect_rows"] == 1.0
               for p in ("train_file", "test_file"))
    res["ok"] = bool(good)
    tr = res["train_file"]
    print(f"[{dataset}] question/concept ids versus H: training file {tr['interactions']} interactions "
          f"({tr['with_concept']} with a concept), agreement correct rows {tr['match_correct_rows']:.3f} / "
          f"incorrect rows {tr['match_incorrect_rows']:.3f}; test file {res['test_file']['match_correct_rows']:.3f} "
          f"=> {'passed' if good else 'FAILED'}")
    _save_json(res, f"concept_line_check_{dataset}.json")
    return res
