"""Collection of the test metrics of all runs (used by step7_summarize.py).

aggregate(...)  results/summary_long.csv: one row per (model, dataset, seed)
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from .config import RESULTS_DIR

RUN_FIELDS = ["auc", "acc", "rmse", "f1", "ece", "d", "w", "best_epoch"]


def model_name(preset):
    if preset.startswith("BL_"):                 # baseline results (results/<dataset>/BL_<model>/)
        return preset[3:]
    return "MAGKT" if preset == "full" else f"MAGKT-{preset}"


def aggregate(results_dir=RESULTS_DIR, presets=None, include_baselines=True):
    """Read results/<dataset>/<preset>/seed<k>/test_metrics.json and write results/summary_long.csv.

    presets: MAGKT presets to include (None = all); baselines (BL_<model>) are included if include_baselines.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "seed*", "test_metrics.json"))):
        parts = os.path.normpath(path).split(os.sep)
        ds, preset, seed = parts[-4], parts[-3], int(parts[-2].replace("seed", ""))
        if ds.startswith("_"):
            continue
        if preset.startswith("BL_"):
            if not include_baselines:
                continue
        elif presets is not None and preset not in presets:
            continue
        with open(path) as f:
            m = json.load(f)
        row = {"model": model_name(preset), "dataset": ds, "preset": preset, "seed": seed}
        row.update({k: m.get(k, np.nan) for k in RUN_FIELDS})
        if row["f1"] > 2 * row["acc"] / (1 + row["acc"]) + 1e-9:
            print(f"!! F1 and ACC are incompatible: {path}")
        rows.append(row)
    if not rows:
        print("No finished runs found.")
        return None
    long = pd.DataFrame(rows)
    out = os.path.join(results_dir, "summary_long.csv")
    long.to_csv(out, index=False)
    print(f"saved: {out} ({len(long)} runs)")
    return long
