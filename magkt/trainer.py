"""Training loop with validation-based model selection.

Label alignment: the output at step t predicts the response to q_{t+1}; the model returns z_{t, q_{t+1}}
directly.

Loss: for each path, the BCE is averaged over the valid positions of each window and summed over the windows
of the batch; train.loss_mode = multi averages the four losses (h, d, f, ensemble).

Windows containing a single interaction have no prediction target and are skipped in training and evaluation.
"""
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_metrics


def next_step(logits, q, r):
    nq = q[:, 1:]
    mask = nq >= 0
    lg = logits[:, :-1]                      # model already returns z_{t, q_{t+1}}
    p = torch.sigmoid(lg)
    y = r[:, 1:].clamp(min=0).float()
    return p, y, mask


def path_loss(logits, q, r):
    p, y, mask = next_step(logits, q, r)
    bce = F.binary_cross_entropy(p, y, reduction="none")
    m = mask.float()
    cnt = m.sum(1)
    per_window = (bce * m).sum(1) / cnt.clamp(min=1.0)
    return per_window[cnt > 0].sum(), p[mask], y[mask]


def compute_loss(out, q, r, loss_mode):
    loss_final, p, y = path_loss(out["final"], q, r)
    if loss_mode == "final":
        return loss_final, p, y
    lh = path_loss(out["h"], q, r)[0]
    ld = path_loss(out["d"], q, r)[0]
    lf = path_loss(out["f"], q, r)[0]
    return (lh + ld + lf + loss_final) / 4.0, p, y


def _unpack(batch, device):
    q, r = batch[0].to(device), batch[1].to(device)
    t = batch[2].to(device) if len(batch) > 2 else None
    c = batch[3].to(device) if len(batch) > 3 else None
    return q, r, t, c


def _hist(batch, device):
    """Memory history preceding the window (memory.history > 0), else None."""
    return tuple(x.to(device) for x in batch[4:8]) if len(batch) > 4 else None


def _forward(model, batch, device):
    q, r, t, c = _unpack(batch, device)
    h = _hist(batch, device)
    out = model(q, r, t, c) if h is None else model(q, r, t, c, hist=h)
    return out, q, r


def train_epoch(model, loader, optimizer, device, loss_mode, grad_clip=0.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        out, q, r = _forward(model, batch, device)
        loss, _, _ = compute_loss(out, q, r, loss_mode)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item()); n += 1
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device, collect_aux=False):
    """Return y, p (flattened over valid positions) and, optionally, per-position extras."""
    model.eval()
    ys, ps, extras = [], [], {"window": [], "pos": [], "q_next": [], "q_cur": [],
                              "activation": [], "gate_mean": []}
    offset = 0
    for batch in loader:
        out, q, r = _forward(model, batch, device)
        p, y, mask = next_step(out["final"], q, r)
        ps.append(p[mask].cpu().numpy()); ys.append(y[mask].cpu().numpy())
        if collect_aux:
            b_idx, t_idx = torch.nonzero(mask, as_tuple=True)
            extras["window"].append((b_idx + offset).cpu().numpy())
            extras["pos"].append(t_idx.cpu().numpy())
            extras["q_next"].append(q[:, 1:][mask].cpu().numpy())
            extras["q_cur"].append(q[:, :-1][mask].cpu().numpy())
            for k in ("activation", "gate_mean"):
                v = out["aux"].get(k)
                extras[k].append(v[:, :-1][mask].cpu().numpy() if v is not None
                                 else np.full(int(mask.sum()), np.nan))
        offset += q.shape[0]
    y = np.concatenate(ys); p = np.concatenate(ps)
    if collect_aux:
        extras = {k: np.concatenate(v) for k, v in extras.items()}
        return y, p, extras
    return y, p


def fit(model, loaders, cfg, out_dir, log=print):
    tc = cfg["train"]
    device = next(model.parameters()).device
    wd = float(tc.get("weight_decay", 0.0))
    id_wd = tc.get("id_weight_decay")
    id_ps = [p for p in model.id_parameters() if p.requires_grad] if hasattr(model, "id_parameters") else []
    id_set = {id(p) for p in id_ps}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in id_set]
    groups = [{"params": rest, "weight_decay": wd}]
    if id_ps:
        groups.append({"params": id_ps, "weight_decay": wd if id_wd is None else float(id_wd)})
    opt = torch.optim.Adam(groups, lr=tc["lr"])
    sched = None
    if tc.get("lr_plateau", False):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                           patience=int(tc.get("plateau_patience", 2)))
    best_auc, best_epoch, bad = -1.0, 0, 0
    hist_path = os.path.join(out_dir, "history.csv")
    ckpt = os.path.join(out_dir, "best.pt")
    fields = ["epoch", "train_loss", "val_auc", "val_acc", "val_rmse", "val_f1", "d", "w", "seconds"]
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)

    for epoch in range(1, tc["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, loaders["train"], opt, device, tc["loss_mode"],
                              float(tc.get("grad_clip", 0.0)))
        y, p = predict(model, loaders["val"], device)
        vm = compute_metrics(y, p)
        diag = model.diagnostics()
        row = [epoch, tr_loss, vm["auc"], vm["acc"], vm["rmse"], vm["f1"], diag["d"], diag["w"], time.time() - t0]
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        log(f"epoch {epoch:3d} | loss {tr_loss:.4f} | val AUC {vm['auc']:.4f} ACC {vm['acc']:.4f} "
            f"| d {diag['d']:.4f} w {diag['w']:.4f}")
        if sched is not None:
            sched.step(vm["auc"])
        if vm["auc"] > best_auc:
            best_auc, best_epoch, bad = vm["auc"], epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= tc["patience"]:
                log(f"early stop at epoch {epoch} (best val epoch {best_epoch})")
                break

    # ---- single evaluation on the test set with the validation-selected checkpoint
    model.load_state_dict(torch.load(ckpt, map_location=device))
    y, p, extras = predict(model, loaders["test"], device, collect_aux=True)
    tm = compute_metrics(y, p)
    tm.update({"best_epoch": best_epoch, "best_val_auc": best_auc, **model.diagnostics()})
    np.savez_compressed(os.path.join(out_dir, "test_predictions.npz"), y=y, p=p, **extras)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(tm, f, indent=2)
    log("TEST " + " ".join(f"{k}={v:.4f}" for k, v in tm.items() if isinstance(v, float)))
    return tm
