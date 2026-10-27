# -*- coding: utf-8 -*-
"""Step 2: smoke test of MAGKT on a small synthetic dataset (about 1-3 minutes).

Every preset of configs/presets.yaml is trained for two epochs on a temporary synthetic dataset, and the test
checks that training, validation and testing run end to end, that no future information reaches earlier
outputs (causality), and that the structural encoder keeps the correct-response and incorrect-response nodes
of a question apart (the aggregation-only encoder of the ablation 'legacy_struct' does not).
The real data are not used.
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

from magkt.smoke import smoke_test  # noqa: E402

if __name__ == "__main__":
    smoke_test()
