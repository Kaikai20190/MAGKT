"""Run experiments from Python (used by the step*.py files in the project root).

run_one(...)   one (dataset, preset, seed) run
run_grid(...)  every combination; finished runs are skipped, so an interrupted
               grid can simply be started again.

Outputs in results/<dataset>/<preset>/seed<k>/:
    config.yaml  data_stats.json  history.csv  best.pt  test_metrics.json  test_predictions.npz  train.log
"""
import gc
import json
import os
import time
import traceback

from .config import RESULTS_DIR, build_config, save_config

# Standard ablation sets (see configs/presets.yaml)
CORE_ABLATIONS = ["no_memory", "no_individual", "no_static", "no_forgetting", "fixed_d", "no_spreading",
                  "no_recall", "fusion_only"]
EXTRA_ABLATIONS = ["no_answer", "no_prior", "recall_unsplit", "indiv_d", "no_cross_gate", "ensemble_loss_only",
                   "graph_identity", "graph_concept", "legacy_struct"]


# Configuration keys introduced after some runs had been stored, with the value those runs implicitly used.
# A run whose config.yaml lacks such a key was run with this value.
_ADDED_KEYS = {
    ("memory", "type"): "activation",
    ("memory", "recall_head"): False,
    ("memory", "recall_split"): False,
}


def _same_config(old, new):
    """Equal configs. A key missing from the old run counts as equal if the new value is what the old run
    implicitly used: the value in _ADDED_KEYS for keys whose default changed later, otherwise the default."""
    if old is None:
        return False
    from .config import CONFIG_DIR, load_yaml
    default = load_yaml(os.path.join(CONFIG_DIR, "default.yaml"))

    def walk(o, n, d, path):
        for k in set(o) | set(n):
            if k not in n:
                return False
            if k not in o:
                key = path + (k,)
                implicit = _ADDED_KEYS[key] if key in _ADDED_KEYS else (
                    d[k] if isinstance(d, dict) and k in d else object())
                if implicit != n[k]:
                    return False
                continue
            if isinstance(o[k], dict) and isinstance(n[k], dict):
                if not walk(o[k], n[k], d.get(k, {}) if isinstance(d, dict) else {}, path + (k,)):
                    return False
            elif o[k] != n[k]:
                return False
        return True

    return walk(old, new, default, ())


def run_dir(dataset, preset, seed, results_dir=RESULTS_DIR):
    return os.path.join(results_dir, dataset, preset, f"seed{seed}")


def run_one(dataset, preset="full", seed=0, overrides=None, results_dir=RESULTS_DIR, force=False):
    """Train with validation-based early stopping, then evaluate once on the test set.

    overrides: list of "section.key=value" strings, e.g. ["train.epochs=5", "data.t_line=2"].
    Returns the test-metric dict.
    """
    import torch

    from .data import describe
    from .pipeline import build
    from .trainer import fit

    out_dir = run_dir(dataset, preset, seed, results_dir)
    done = os.path.join(out_dir, "test_metrics.json")
    cfg = build_config(dataset, preset, overrides)
    cfg["seed"] = seed
    if os.path.exists(done) and not force:
        from .config import load_yaml
        old = load_yaml(os.path.join(out_dir, "config.yaml")) if os.path.exists(
            os.path.join(out_dir, "config.yaml")) else None
        if old is not None:
            old.get("data", {}).pop("num_concepts", None)
        import yaml
        new = yaml.safe_load(yaml.safe_dump(cfg))
        new["data"].pop("num_concepts", None)
        if _same_config(old, new):
            print(f"[skipped] finished: {out_dir}")
            with open(done) as f:
                return json.load(f)
        rel = os.path.relpath(out_dir, results_dir)
        backup = os.path.join(results_dir + "_old", rel + "_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        os.rename(out_dir, backup)
        print(f"[configuration changed] previous result moved to {backup}; running again")
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config.yaml"))

    with open(os.path.join(out_dir, "train.log"), "w", encoding="utf-8") as log_f:
        def log(msg):
            print(msg, flush=True)
            log_f.write(msg + "\n"); log_f.flush()

        model, loaders, splits, priors = build(cfg)
        stats = describe(splits)
        stats["graph"] = priors["info"]
        stats["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
        stats["device"] = str(next(model.parameters()).device)
        with open(os.path.join(out_dir, "data_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        log(f"===== {dataset} | {preset} | seed {seed} | device {stats['device']} =====")
        log(json.dumps(stats, ensure_ascii=False))
        t0 = time.time()
        tm = fit(model, loaders, cfg, out_dir, log=log)
        log(f"time {(time.time() - t0) / 60:.1f} min")

    del model, loaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tm


def run_grid(datasets, presets, seeds=(0, 1, 2, 3, 4), overrides=None, results_dir=RESULTS_DIR,
             stop_on_error=False):
    """Run every (dataset, preset, seed). Failed runs are reported and skipped unless stop_on_error."""
    total = len(datasets) * len(presets) * len(seeds)
    k, failed = 0, []
    for ds in datasets:
        for pr in presets:
            for sd in seeds:
                k += 1
                print(f"\n>>> [{k}/{total}] {ds} | {pr} | seed {sd}", flush=True)
                try:
                    run_one(ds, pr, sd, overrides=overrides, results_dir=results_dir)
                except Exception:
                    if stop_on_error:
                        raise
                    traceback.print_exc()
                    failed.append((ds, pr, sd))
    print(f"\nfinished {total - len(failed)}/{total} runs.")
    if failed:
        print("failed runs (run the script again after fixing; finished runs are skipped):")
        for f in failed:
            print("   ", f)
    return failed
