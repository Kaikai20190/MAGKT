# -*- coding: utf-8 -*-
"""Step 5: ablation variants (Tables 5 and 6, Fig. 4).

The variants are defined in configs/presets.yaml (the elapsed-time variant 'time_decay' is run by
step6_time_decay.py):
  no_memory           without the memory branch               no_static        without the structural branch
  fusion_only         single prediction path (fused input)    ensemble_loss_only  loss on the averaged prediction only
  no_cross_gate       equal-weight fusion                      legacy_struct    aggregation-only structural encoder
  no_individual       shared, history-free memory (NI)         no_answer        no response in the memory content
  no_forgetting       all decay rates set to zero              fixed_d          decay rates fixed at their initial values
  indiv_d             decay rates conditioned on the learner's history
  no_spreading        transition prior removed (w = 0)         no_prior         both associative priors removed (u = w = 0)
  no_recall           recall term removed                      recall_unsplit   recall term with a single activation
  graph_identity      no transition edges                      graph_concept    concept-based graph instead of transitions
Results: results/<dataset>/<variant>/seed<k>/. Finished runs are skipped.
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

from magkt.experiments import CORE_ABLATIONS, EXTRA_ABLATIONS, run_grid  # noqa: E402

# ============================ settings ============================
DATASETS = ["assist2009", "kddcup2010", "static2011", "assist2017"]
VARIANTS = CORE_ABLATIONS + EXTRA_ABLATIONS
SEEDS = [0, 1, 2, 3, 4]
# ==================================================================

if __name__ == "__main__":
    run_grid(DATASETS, VARIANTS, seeds=SEEDS)
