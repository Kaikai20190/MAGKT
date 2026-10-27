# -*- coding: utf-8 -*-
"""Step 6: lag measured in elapsed time instead of in interactions (variant 'time_decay', Table 5).

The lag between a past interaction s and the retrieval for q_{t+1} is (tau_{t+1} - tau_s) / 60 + 1, i.e. measured
in minutes (configs/presets.yaml). Only datasets with timestamps are used (KDDCup2010 and Statics2011 as produced
by step1_prepare_data.py); the reference with the interaction-based lag is the 'full' run of step 4.
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

from magkt.config import build_config  # noqa: E402
from magkt.experiments import run_grid  # noqa: E402

# ============================ settings ============================
TIME_DATASETS = ["kddcup2010", "static2011"]
SEEDS = [0, 1, 2, 3, 4]
# ==================================================================

if __name__ == "__main__":
    ok = []
    for ds in TIME_DATASETS:
        if build_config(ds)["data"]["t_line"] is None:
            print(f"[skipped] {ds}: the dataset configuration has no timestamp line (t_line)")
        else:
            ok.append(ds)
    run_grid(ok, ["time_decay"], seeds=SEEDS)
