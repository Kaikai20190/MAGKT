# -*- coding: utf-8 -*-
"""Step 3: selection of the dataset-specific settings on the validation learners (before step 4).

For every dataset, a single run with seed 0 is trained per candidate, and candidates are compared by their
validation AUC; the test learners are not used. The best setting is written to configs/datasets/<dataset>.yaml
(Table 2). The search proceeds in stages, each starting from the best setting of the previous stage:
  stage 1  output layer (dot product, Eq. 23, or MLP, Eq. 23b) combined with the segment length (50 / 100 / 200);
           concept-response embeddings and the concept output term are always used (model.kc_input, kc_head)
  stage 2  learning rate, weight decay, input dropout, embedding dimension, gradient clipping
  stage 3  dropout of question embeddings, a separate weight decay on question-specific parameters,
           halving of the learning rate when the validation AUC stops improving
  stage 4  (STAGE4_DATASETS only) further values of the question-specific weight decay and an output layer
           with concept-level output vectors and two scalar parameters per question ('rasch')
Finished candidates are skipped; the summary is saved to results_tune/tuning_summary.csv.
"""
import os
import sys

import yaml


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

from magkt.experiments import run_one  # noqa: E402

# ============================ settings ============================
DATASETS = ["assist2009", "kddcup2010", "static2011", "assist2017"]
KC = ["model.kc_input=true", "model.kc_head=true", "train.weight_decay=1.0e-5"]
STAGE1 = {
    "dot_L50": KC + ["model.head_type=dot", "data.max_len=50"],
    "mlp_L50": KC + ["model.head_type=mlp", "data.max_len=50"],
    "dot_L200": KC + ["model.head_type=dot", "data.max_len=200"],
    "mlp_L100": KC + ["model.head_type=mlp", "data.max_len=100"],
    "mlp_L200": KC + ["model.head_type=mlp", "data.max_len=200"],
}
STAGE2 = {                      # applied on top of the best stage-1 setting
    "lr5e-4": ["train.lr=5.0e-4"],
    "lr2e-3": ["train.lr=2.0e-3"],
    "wd1e-4": ["train.weight_decay=1.0e-4"],
    "drop0.2": ["model.input_dropout=0.2"],
    "emb128": ["model.emb_dim=128"],
    "clip1": ["train.grad_clip=1.0"],
}
STAGE3 = {                      # regularising question-specific parameters + LR schedule, on top of stage 2
    "iddrop0.2": ["model.id_dropout=0.2"],
    "iddrop0.5": ["model.id_dropout=0.5"],
    "idwd1e-3": ["train.id_weight_decay=1.0e-3"],
    "idwd1e-2": ["train.id_weight_decay=1.0e-2"],
    "plateau": ["train.lr_plateau=true"],
    "iddrop0.2+plateau": ["model.id_dropout=0.2", "train.lr_plateau=true"],
}
STAGE4 = {                      # question-specific capacity, on top of stage 3 (alternative heads replace head_type)
    "idwd3e-4": ["train.id_weight_decay=3.0e-4"],
    "idwd3e-3": ["train.id_weight_decay=3.0e-3"],
    "rasch": ["model.head_type=rasch", "train.id_weight_decay=null"],
    "rasch+idwd1e-3": ["model.head_type=rasch", "train.id_weight_decay=1.0e-3"],
    "rasch+idwd1e-2": ["model.head_type=rasch", "train.id_weight_decay=1.0e-2"],
    "mlp+idwd1e-3": ["model.head_type=mlp", "train.id_weight_decay=1.0e-3"],
}
STAGE4_DATASETS = ["assist2009"]
RUN_STAGE2 = True
RUN_STAGE3 = True
RUN_STAGE4 = True
SEED = 0
WRITE_BEST = True
# ================================================================

TUNE_DIR = os.path.join(ROOT, "results_tune")


def ds_yaml(ds):
    return os.path.join(ROOT, "configs", "datasets", f"{ds}.yaml")


def _load(ds):
    with open(ds_yaml(ds), encoding="utf-8") as f:
        text = f.read()
    header = "".join(l for l in text.splitlines(True) if l.startswith("#") and "selected on the validation" not in l)
    return yaml.safe_load(text) or {}, header


def _dump(ds, cfg, header, extra=""):
    with open(ds_yaml(ds), "w", encoding="utf-8") as f:
        f.write(header + extra)
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def ensure_concept_line(ds):
    """Datasets in the 4/5-line format carry the concept of every interaction on line 2."""
    cfg, header = _load(ds)
    d = cfg.setdefault("data", {})
    if d.get("c_line") is None and int(d.get("n_lines", 3)) >= 4:
        d["c_line"] = 2
        _dump(ds, cfg, header)
        print(f"[{ds}] c_line: 2 set in the configuration (line with the concept of every interaction)")
    return d.get("c_line") is not None


def reset_tuned(ds):
    """Remove model/train/max_len settings written by an earlier tuning run, so candidates start clean."""
    cfg, header = _load(ds)
    cfg.pop("model", None)
    cfg.pop("train", None)
    cfg.get("data", {}).pop("max_len", None)
    _dump(ds, cfg, header)


def write_best(ds, overrides):
    cfg, header = _load(ds)
    for item in overrides:
        k, v = item.split("=", 1)
        sec, key = k.split(".")
        cfg.setdefault(sec, {})[key] = yaml.safe_load(v)
    _dump(ds, cfg, header, "# model/train/max_len: selected on the validation learners by step3_tune.py\n")


def merge(base, extra):
    keys = {x.split("=", 1)[0] for x in extra}
    return [x for x in base if x.split("=", 1)[0] not in keys] + list(extra)


if __name__ == "__main__":
    import pandas as pd

    rows = []
    for ds in DATASETS:
        has_c = ensure_concept_line(ds)
        if not has_c:
            print(f"[{ds}] no concept line in the data, skipped"); continue
        with open(ds_yaml(ds), encoding="utf-8") as f:
            yaml_backup = f.read()
        reset_tuned(ds)
        holder = {"best": None}
        try:

            def trial(name, ov, stage):
                print("\n" + "=" * 70 + f"\n{ds} | stage {stage} | {name}\n{ov}\n" + "=" * 70)
                tm = run_one(ds, "full", SEED, overrides=list(ov), results_dir=os.path.join(TUNE_DIR, name))
                rows.append({"dataset": ds, "stage": stage, "candidate": name, "val_auc": tm["best_val_auc"],
                             "best_epoch": tm["best_epoch"], "overrides": " ".join(ov)})
                return tm["best_val_auc"]

            holder["best"] = best = None
            for name, ov in STAGE1.items():
                v = trial(name, ov, 1)
                if best is None or v > best[1]:
                    best = (name, v, ov)
                    holder["best"] = best
            print(f"\n[{ds}] best of stage 1: {best[0]} (validation AUC {best[1]:.4f})")
            if RUN_STAGE2:
                base_name, base_ov = best[0], best[2]
                for name, extra in STAGE2.items():
                    ov = merge(base_ov, extra)
                    v = trial(f"{base_name}+{name}", ov, 2)
                    if v > best[1]:
                        best = (f"{base_name}+{name}", v, ov)
                        holder["best"] = best
            if RUN_STAGE3:
                base_name, base_ov = best[0], best[2]
                for name, extra in STAGE3.items():
                    ov = merge(base_ov, extra)
                    v = trial(f"{base_name}+{name}", ov, 3)
                    if v > best[1]:
                        best = (f"{base_name}+{name}", v, ov)
                        holder["best"] = best
            if RUN_STAGE4 and ds in STAGE4_DATASETS:
                base_name, base_ov = best[0], best[2]
                for name, extra in STAGE4.items():
                    ov = merge(base_ov, extra)
                    v = trial(f"{base_name}+{name}", ov, 4)
                    if v > best[1]:
                        best = (f"{base_name}+{name}", v, ov)
                        holder["best"] = best
            if WRITE_BEST:
                write_best(ds, best[2])
                holder["written"] = True
            print(f"\n[{ds}] selected: {best[0]} (validation AUC {best[1]:.4f})" + (", written to the configuration" if WRITE_BEST else ""))


        finally:
            if holder["best"] is not None and not holder.get("written") and WRITE_BEST:
                write_best(ds, holder["best"][2])  # interrupted: keep the best setting found so far
                print(f"[{ds}] interrupted; the best setting so far was written: {holder['best'][0]}")
            elif holder["best"] is None:          # interrupted before any trial finished: restore
                with open(ds_yaml(ds), "w", encoding="utf-8") as f:
                    f.write(yaml_backup)
        df = pd.DataFrame(rows)
        os.makedirs(TUNE_DIR, exist_ok=True)
        df.to_csv(os.path.join(TUNE_DIR, "tuning_summary.csv"), index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70 + "\nValidation AUC (used for selection only)\n" + "=" * 70)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[["dataset", "stage", "candidate", "val_auc", "best_epoch"]].round(4).to_string(index=False))
    print("\nNext: step4_main_experiments.py")