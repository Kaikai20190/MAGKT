# -*- coding: utf-8 -*-
"""Recompute the reported test metrics from the released predictions.

predictions/<dataset>/<model>/seed<k>.npz contains, for every evaluated test interaction, the observed response
y (1 = correct) and the predicted probability p of a correct response, produced by the checkpoint with the best
validation AUC. This script recomputes AUC, ACC, RMSE, F1, precision, recall and ECE with magkt/metrics.py
(ACC and F1 at threshold 0.5, correct response as the positive class) and reports, for each model and
dataset, the mean and standard deviation over the seeds. For every run it also checks F1 <= 2 ACC / (1 + ACC),
which holds for any binary classifier evaluated on the same samples.

Output: printed to the console and saved to predictions/recomputed_metrics.csv (one row per run) and
predictions/recomputed_summary.csv (mean and standard deviation).
"""
import glob
import os
import sys


def _project_root():
    """The MAGKT folder (contains configs/ and magkt/), found from this file's location."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(d, "configs")) and os.path.isdir(os.path.join(d, "magkt")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("Project root not found: place this file in the MAGKT folder (next to configs/ and magkt/)")


ROOT = _project_root()
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from magkt.metrics import compute_metrics, f1_upper_bound  # noqa: E402

# ============================ settings ============================
PRED_DIR = os.path.join(ROOT, "predictions")
DATASETS = ["assist2009", "kddcup2010", "static2011", "assist2017"]
METRICS = ["auc", "acc", "rmse", "f1", "precision", "recall", "ece"]
# ==================================================================


def collect():
    rows = []
    for ds in DATASETS:
        for mdir in sorted(glob.glob(os.path.join(PRED_DIR, ds, "*"))):
            if not os.path.isdir(mdir):
                continue
            for f in sorted(glob.glob(os.path.join(mdir, "seed*.npz"))):
                z = np.load(f)
                m = compute_metrics(z["y"], z["p"])
                m.update(dataset=ds, model=os.path.basename(mdir),
                         seed=int(os.path.basename(f)[4:-4]),
                         f1_bound=f1_upper_bound(m["acc"]))
                rows.append(m)
    if not rows:
        raise FileNotFoundError(f"no prediction files found in {PRED_DIR}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    runs = collect()
    runs = runs[["dataset", "model", "seed", "n", "positive_rate"] + METRICS + ["f1_bound"]]
    ok = (runs["f1"] <= runs["f1_bound"] + 1e-9).all()
    summ = runs.groupby(["dataset", "model"], sort=False)[METRICS].agg(["mean", "std"])
    summ.columns = [f"{k}_{s}" for k, s in summ.columns]
    summ.insert(0, "runs", runs.groupby(["dataset", "model"], sort=False)["seed"].count())

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    for ds in DATASETS:
        if ds not in summ.index.get_level_values(0):
            continue
        s = summ.loc[ds]
        print(f"\n===== {ds} (mean and standard deviation over seeds) =====")
        tab = pd.DataFrame({"runs": s["runs"]})
        for k in METRICS:
            tab[k.upper()] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(s[f"{k}_mean"], s[f"{k}_std"])]
        print(tab.to_string())
    print(f"\nF1 <= 2 ACC / (1 + ACC) for all {len(runs)} runs: {'yes' if ok else 'NO'}")

    runs.to_csv(os.path.join(PRED_DIR, "recomputed_metrics.csv"), index=False)
    summ.reset_index().to_csv(os.path.join(PRED_DIR, "recomputed_summary.csv"), index=False)
    print("saved:", os.path.join(PRED_DIR, "recomputed_metrics.csv"), "and recomputed_summary.csv")
