"""Liveness training loop. Config-driven, checkpointed, and logged to JSON.

Deliberately not using an experiment-tracking service: a JSON run log records the same
information (config, per-epoch metrics, best epoch, duration, environment) with no
extra infrastructure to install inside a Kaggle session.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.evaluation.metrics import compute_metrics, select_threshold
from ml.liveness.dataset import LivenessDataset, SequenceAugment, clip_eval_batches
from ml.liveness.models import build_model


@dataclass
class RunLog:
    run_id: str
    config: dict
    device: str
    n_train: int
    n_val: int
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_acer: float = float("inf")
    duration_sec: float = 0.0
    notes: str = ""

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def score_clips(model, rows, crops_dir, mode, seq_len, device, max_seqs=4):
    """Produce one score per clip by averaging its views.

    Both architectures are reduced to the same unit of evaluation this way, which is
    what makes the baseline-vs-temporal comparison meaningful.
    """
    model.eval()
    scores, labels, attacks, ids = [], [], [], []
    for item in clip_eval_batches(rows, crops_dir, mode, seq_len, max_seqs):
        views = item["views"].to(device)
        logits = model(views)
        p = torch.sigmoid(logits).mean().item()
        scores.append(p)
        labels.append(item["label"])
        attacks.append(item["attack_type"])
        ids.append(item["clip_id"])
    return np.array(labels), np.array(scores), attacks, ids


def train(cfg: dict, splits: dict, crops_dir: str, out_dir: str, run_id: str) -> RunLog:
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mcfg, tcfg, dcfg = cfg["model"], cfg["training"], cfg["data"]
    mode = "frame" if mcfg["arch"] == "cnn" else "sequence"
    seq_len = dcfg.get("sequence_length", 8)

    aug = None
    acfg = cfg.get("augmentation", {})
    if acfg.get("enabled"):
        import inspect
        allowed = set(inspect.signature(SequenceAugment.__init__).parameters) - {"self"}
        params = {k: v for k, v in acfg.items() if k in allowed}
        ignored = set(acfg) - allowed - {"enabled"}
        if ignored:
            # Silently dropping an augmentation key would mean training without an
            # augmentation the config says is on — a difference that never surfaces
            # as an error but does change the results.
            print(f"WARNING: ignoring unknown augmentation keys {sorted(ignored)}")
        aug = SequenceAugment(**params)

    # A single-frame model sees one frame per clip per epoch, so with few clips an
    # epoch is tiny. Sampling several frames per clip per epoch evens out the number
    # of gradient steps between the two architectures, keeping the comparison fair.
    spc = 8 if mode == "frame" else 2

    train_ds = LivenessDataset(splits["train"], crops_dir, mode, seq_len, aug, spc)
    val_ds = LivenessDataset(splits["val"], crops_dir, mode, seq_len, None, 1)
    if train_ds.skipped:
        print(f"WARNING: {len(train_ds.skipped)} train clips had no cached crops")

    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,
                          num_workers=tcfg.get("num_workers", 2), drop_last=True,
                          pin_memory=(device == "cuda"))

    model = build_model(mcfg).to(device)

    # Class weighting for the 4:1 spoof:live imbalance. Without it the model can reach
    # high accuracy by predicting 'spoof' for everything, which is exactly the failure
    # the APCER/BPCER pair is designed to expose.
    n_live = sum(1 for r in splits["train"] if r["label"] == "live")
    n_spoof = len(splits["train"]) - n_live
    pos_weight = torch.tensor([n_spoof / max(n_live, 1)], device=device)
    print(f"class balance: {n_live} live / {n_spoof} spoof  -> pos_weight={pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                            weight_decay=tcfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tcfg["epochs"])
    use_amp = bool(tcfg.get("mixed_precision", True)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log = RunLog(run_id=run_id, config=cfg, device=device,
                 n_train=len(train_ds), n_val=len(val_ds))
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{run_id}_best.pt")
    patience = tcfg.get("early_stopping_patience", 5)
    since_best = 0
    t0 = time.time()

    for epoch in range(tcfg["epochs"]):
        model.train()
        losses = []
        for x, y, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        y_val, s_val, a_val, _ = score_clips(model, splits["val"], crops_dir, mode,
                                             seq_len, device)
        # Threshold is chosen on validation each epoch purely to report a comparable
        # val ACER. The threshold shipped to production is fixed once, after training.
        thr = select_threshold(y_val, s_val, criterion="min_acer")
        m = compute_metrics(y_val, s_val, thr, a_val)

        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_acer": m.acer, "val_apcer": m.apcer, "val_bpcer": m.bpcer,
               "val_auc": m.roc_auc, "threshold": thr,
               "lr": opt.param_groups[0]["lr"]}
        log.epochs.append(rec)
        print(f"epoch {epoch:>2}  loss {rec['train_loss']:.4f}  "
              f"val ACER {m.acer * 100:5.2f}%  AUC {m.roc_auc:.4f}")

        if m.acer < log.best_val_acer:
            log.best_val_acer, log.best_epoch, since_best = m.acer, epoch, 0
            torch.save({"model": model.state_dict(), "config": cfg,
                        "epoch": epoch, "val_acer": m.acer}, best_path)
        else:
            since_best += 1
            if since_best >= patience:
                print(f"early stopping at epoch {epoch} (no improvement in {patience})")
                break

    log.duration_sec = time.time() - t0
    torch.save({"model": model.state_dict(), "config": cfg},
               os.path.join(out_dir, f"{run_id}_last.pt"))
    log.save(os.path.join(out_dir, f"{run_id}_run.json"))
    print(f"\nbest epoch {log.best_epoch}  val ACER {log.best_val_acer * 100:.2f}%  "
          f"({log.duration_sec / 60:.1f} min)")
    return log
