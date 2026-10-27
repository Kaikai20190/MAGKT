# -*- coding: utf-8 -*-
"""Step 7: collect the test metrics of all runs into results/summary_long.csv.

One row per (model, dataset, seed) with AUC, ACC, RMSE, F1, ECE, the learned d and w and the selected epoch.
Only the models reported in the paper are collected: MAGKT ('full'), its ablation variants and the baselines
(results/<dataset>/BL_<model>/, if present).
"""
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

from magkt.experiments import CORE_ABLATIONS, EXTRA_ABLATIONS  # noqa: E402
from magkt.report import aggregate  # noqa: E402

# ============================ settings ============================
PRESETS = ["full"] + CORE_ABLATIONS + EXTRA_ABLATIONS + ["time_decay"]
INCLUDE_BASELINES = True
# ==================================================================

if __name__ == "__main__":
    long = aggregate(presets=PRESETS, include_baselines=INCLUDE_BASELINES)
    if long is not None:
        n = long.groupby(["dataset", "model"]).size().unstack(0).fillna(0).astype(int)
        print("\nNumber of runs per model and dataset:")
        print(n.to_string())
