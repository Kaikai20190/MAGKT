# -*- coding: utf-8 -*-
"""Step 4: main experiments with MAGKT (Table 3).

Protocol: early stopping on the validation AUC (patience 10); the checkpoint with the best validation AUC is
evaluated once on the test learners; five seeds per dataset. The dataset-specific settings selected by
step3_tune.py are read from configs/datasets/<dataset>.yaml (Table 2).

Results: results/<dataset>/full/seed<k>/. Finished runs are skipped, so the script can be restarted after an
interruption. With QUICK_TRY = True, one seed is trained for two epochs on every dataset (results_quick/) to
check that everything runs.
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

from magkt.experiments import run_grid  # noqa: E402

# ============================ settings ============================
DATASETS = ["assist2009", "kddcup2010", "static2011", "assist2017"]
PRESETS = ["full"]
SEEDS = [0, 1, 2, 3, 4]
QUICK_TRY = False
# ==================================================================


def show_settings(datasets):
    """Print the tuned settings of every dataset; warn if a dataset has not been tuned."""
    import yaml
    print("Dataset-specific settings (configs/datasets/<dataset>.yaml):")
    for ds in datasets:
        with open(os.path.join(ROOT, "configs", "datasets", f"{ds}.yaml"), encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        tuned = {**{f"model.{k}": v for k, v in (y.get("model") or {}).items()},
                 **{f"train.{k}": v for k, v in (y.get("train") or {}).items()}}
        if "max_len" in (y.get("data") or {}):
            tuned["data.max_len"] = y["data"]["max_len"]
        if tuned:
            print(f"  {ds:12s} " + ", ".join(f"{k}={v}" for k, v in tuned.items()))
        else:
            print(f"  {ds:12s} [warning] no tuned settings, the defaults are used; run step3_tune.py first")


if __name__ == "__main__":
    show_settings(DATASETS)
    if QUICK_TRY:
        run_grid(DATASETS, ["full"], seeds=[0], overrides=["train.epochs=2"],
                 results_dir=os.path.join(ROOT, "results_quick"))
    else:
        run_grid(DATASETS, PRESETS, seeds=SEEDS)
