"""Assemble data, structural priors and model from a config (shared by all scripts)."""
import os
import random

import numpy as np
import torch

from . import data as D
from . import graphs as Gr
from .model import MAGKT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_priors(cfg, train_learners):
    """Return torch sparse tensors: G as factors (L, R) with G = L R, A_out, A_in, S (Q x Q), and info."""
    H = Gr.load_incidence(cfg)
    L, R = Gr.hypergraph_factors(H, norm=cfg["graph"]["hyper_norm"])
    A_out, A_in, S, info = Gr.transition_graphs(cfg, train_learners, H)
    return {
        "H": H,
        "G": (Gr.to_torch_sparse(L), Gr.to_torch_sparse(R)),
        "A_out": Gr.to_torch_sparse(A_out),
        "A_in": Gr.to_torch_sparse(A_in),
        "S": Gr.to_torch_sparse(S),
        "info": info,
    }


def build(cfg, seed=None):
    """Return (model, loaders, splits, priors). The data split does not depend on `seed`."""
    seed = cfg["seed"] if seed is None else seed
    splits = D.load_splits(cfg)
    if cfg["data"].get("c_line") is not None:
        cmax = max((int(s["c"].max()) for part in splits["learners"].values() for s in part
                    if s.get("c") is not None and len(s["c"])), default=0)
        cfg["data"]["num_concepts"] = cmax
    if cfg["memory"]["decay_time"] == "time" and cfg["data"]["t_line"] is None:
        raise ValueError("decay_time = time needs timestamps (data.t_line)")
    priors = build_priors(cfg, splits["learners"]["train"])
    set_seed(seed)
    device = get_device(cfg["train"]["device"])
    model = MAGKT(cfg, G=priors["G"], A_out=priors["A_out"], A_in=priors["A_in"], S=priors["S"]).to(device)
    tc = cfg["train"]
    w = splits["windows"]
    loaders = {
        "train": D.make_loader(w["train"], tc["batch_size"], shuffle=True,
                               num_workers=tc["num_workers"], drop_last=True),
        "val": D.make_loader(w["val"], tc["batch_size"], shuffle=False, num_workers=tc["num_workers"]),
        "test": D.make_loader(w["test"], tc["batch_size"], shuffle=False, num_workers=tc["num_workers"]),
    }
    return model, loaders, splits, priors


def load_run(run_dir, device="auto"):
    """Rebuild a trained model from a results folder (config.yaml + best.pt)."""
    from .config import load_yaml
    cfg = load_yaml(os.path.join(run_dir, "config.yaml"))
    cfg["train"]["device"] = device
    model, loaders, splits, priors = build(cfg)
    dev = next(model.parameters()).device
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=dev))
    model.eval()
    return cfg, model, loaders, splits, priors
