"""Configuration loading.

A run is configured by merging, in order:
  configs/default.yaml  ->  configs/datasets/<name>.yaml  ->  a preset from
  configs/presets.yaml  ->  command-line overrides  (--set key.sub=value)
"""
import copy
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "configs")
RESULTS_DIR = os.path.join(ROOT, "results")


def data_path(cfg, rel):
    """Resolve a data file: relative `data.root` is taken relative to the project root,
    so scripts work whatever the working directory is (e.g. in PyCharm)."""
    root = cfg["data"]["root"]
    if not os.path.isabs(root):
        root = os.path.join(ROOT, root)
    return os.path.join(root, rel)


def _deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def _set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"Unknown config section '{k}' in '{dotted}'")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted}'")
    node[keys[-1]] = value


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_presets():
    return load_yaml(os.path.join(CONFIG_DIR, "presets.yaml"))


def build_config(dataset, preset="full", overrides=None):
    cfg = load_yaml(os.path.join(CONFIG_DIR, "default.yaml"))
    _deep_update(cfg, load_yaml(os.path.join(CONFIG_DIR, "datasets", f"{dataset}.yaml")))
    presets = load_presets()
    if preset not in presets:
        raise KeyError(f"Unknown preset '{preset}'. Available: {sorted(presets)}")
    for dotted, value in (presets[preset] or {}).items():
        _set_dotted(cfg, dotted, value)
    for item in overrides or []:
        dotted, raw = item.split("=", 1)
        _set_dotted(cfg, dotted, yaml.safe_load(raw))
    cfg["preset"] = preset
    return cfg


def save_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
