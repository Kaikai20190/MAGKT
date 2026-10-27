"""End-to-end check on a tiny synthetic dataset (step2_smoke_test.py).

Trains every preset for 2 epochs on a temporary synthetic dataset and checks
that training runs, that no future information leaks into past outputs, and
that the structural encoder keeps correct/incorrect nodes apart (the
aggregation-only encoder of the 'legacy_struct' ablation does not, because the
synthetic hypergraph gives both nodes the same memberships). Outputs go to
results/_smoke.
"""
import os
import shutil
import tempfile

import numpy as np

import torch

from .config import RESULTS_DIR, build_config, load_presets
from .pipeline import build
from .trainer import fit

Q, M = 30, 6


def make_data(root):
    rng = np.random.RandomState(0)
    os.makedirs(os.path.join(root, "syn"), exist_ok=True)
    os.makedirs(os.path.join(root, "H"), exist_ok=True)
    concept = rng.randint(0, M, size=Q)
    H = np.zeros((2 * Q, M), dtype=int)
    H[np.arange(Q), concept] = 1
    H[np.arange(Q) + Q, concept] = 1
    H[0, (concept[0] + 1) % M] = H[Q, (concept[0] + 1) % M] = 1     # one multi-concept question
    np.savetxt(os.path.join(root, "H", "syn.csv"), H, fmt="%d", delimiter=",")
    skill = rng.normal(size=M)
    for name, n in (("train", 120), ("test", 40)):
        with open(os.path.join(root, "syn", f"syn_{name}.txt"), "w") as f:
            for _ in range(n):
                L = rng.randint(2, 130)
                q = rng.randint(1, Q + 1, size=L)
                theta = rng.normal()
                p = 1 / (1 + np.exp(-(theta + skill[concept[q - 1]] + 0.02 * np.arange(L))))
                r = (rng.uniform(size=L) < p).astype(int)
                t = np.cumsum(rng.exponential(600, size=L)).astype(int)
                c = concept[q - 1] + 1
                f.write(f"{L}\n" + ",".join(map(str, q)) + "\n" + ",".join(map(str, t)) + "\n"
                        + ",".join(map(str, c)) + "\n" + ",".join(map(str, r)) + "\n")


def cfg_for(preset, data_root, extra_model=None, extra_train=None):
    cfg = build_config("assist2017", preset)
    cfg["data"].update({"root": data_root, "name": "syn", "train_file": "syn/syn_train.txt",
                        "test_file": "syn/syn_test.txt", "hypergraph_file": "H/syn.csv",
                        "num_questions": Q, "n_lines": 5, "q_line": 1, "t_line": 2, "c_line": 3,
                        "a_line": 4, "qid_offset": 1, "invalid_qid": "error"})
    cfg["model"].update({"emb_dim": 32, "hidden_dim": 16, "kc_input": False, "kc_head": False,
                         "input_dropout": 0.0, "head_type": "dot", "id_dropout": 0.0})
    cfg["data"]["max_len"] = 50
    cfg["model"].update(extra_model or {})
    cfg["train"].update(extra_train or {})
    cfg["train"].update({"epochs": 2, "batch_size": 16, "patience": 5})
    return cfg


def check_causality(model, loaders):
    """Changing the future must not change past outputs."""
    model.eval()
    batch = next(iter(loaders["test"]))
    q, r = batch[0][:4].clone(), batch[1][:4].clone()
    dev = next(model.parameters()).device
    t = batch[2][:4].to(dev)
    c = batch[3][:4].to(dev)
    with torch.no_grad():
        a = model(q.to(dev), r.to(dev), t, c)["final"]
        q2, r2 = q.clone(), r.clone()
        k = 10
        valid = q2[:, k:] >= 0
        q2[:, k:][valid] = (q2[:, k:][valid] + 1) % Q
        r2[:, k:][valid] = 1 - r2[:, k:][valid]
        c2 = c.clone()
        c2[:, k:] = (c2[:, k:] % 3) + 1
        b = model(q2.to(dev), r2.to(dev), t, c2)["final"]
    assert torch.allclose(a[:, :k - 1], b[:, :k - 1], atol=1e-5), "future information leaks into past outputs"


def smoke_test(presets=None):
    data_root = tempfile.mkdtemp(prefix="magkt_syn_")
    out_root = os.path.join(RESULTS_DIR, "_smoke")
    make_data(data_root)
    try:
        runs = [(p, p, None, None) for p in (presets or list(load_presets()))]
        runs.append(("full+kc", "full", {"kc_input": True, "kc_head": True, "input_dropout": 0.2},
                     {"weight_decay": 1e-5}))
        runs.append(("no_memory+kc", "no_memory", {"kc_input": True, "kc_head": True}, None))
        runs.append(("full+kc+mlp", "full", {"kc_input": True, "kc_head": True, "head_type": "mlp"},
                     {"grad_clip": 1.0}))
        runs.append(("no_static+mlp", "no_static", {"head_type": "mlp"}, None))
        runs.append(("full+kc+rasch", "full", {"kc_input": True, "head_type": "rasch"},
                     {"id_weight_decay": 1e-3}))
        runs.append(("full+kc+idreg", "full", {"kc_input": True, "kc_head": True, "id_dropout": 0.3},
                     {"id_weight_decay": 1e-3, "lr_plateau": True, "plateau_patience": 0}))
        for label, preset, em, et in runs:
            cfg = cfg_for(preset, data_root, em, et)
            out = os.path.join(out_root, label)
            os.makedirs(out, exist_ok=True)
            model, loaders, splits, priors = build(cfg)
            tm = fit(model, loaders, cfg, out, log=lambda m: None)
            check_causality(model, loaders)
            if cfg["model"]["use_static"]:
                with torch.no_grad():
                    Z = model.struct_node_repr()
                gap = (Z[:Q] - Z[Q:]).norm(dim=-1).min().item()
                if cfg["model"]["struct_encoder"] == "sage":
                    assert gap > 0, "correct/incorrect nodes collapsed"
                else:
                    assert gap < 1e-4, "aggregation-only encoder expected to map these nodes to the same representation"
            print(f"[ok] {label:22s} test AUC {tm['auc']:.4f}  d {tm['d']:.3f}  w {tm['w']:.3f}")
    finally:
        shutil.rmtree(data_root, ignore_errors=True)
    print("\nsmoke test passed")

