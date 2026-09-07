"""Evaluate trained checkpoints against cached crops, off the training box.

Training produced checkpoints and cached face crops; evaluation needs nothing else.
Running it locally avoids repeating a 30-minute preprocessing pass on a remote GPU box
just to recompute metrics.

Usage:
    python scripts/evaluate_local.py --output-dir <kaggle_output> --out docs/results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

from ml.evaluation.evaluate import evaluate_all, format_report, save_report
from ml.liveness.models import build_model
from ml.preprocessing.splits import load_manifest, split_by_group, summarise_by
from ml.training.train import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True,
                    help="directory containing crops/ and models/ from the training run")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.output_dir
    crops_root = os.path.join(root, "crops")
    models_root = os.path.join(root, "models")
    if not os.path.isdir(crops_root):
        raise SystemExit(f"no crops/ under {root!r}")

    CROPS = {name: os.path.join(crops_root, name)
             for name in os.listdir(crops_root)
             if os.path.exists(os.path.join(crops_root, name, "manifest.csv"))}
    print("crop sets:", {k: v for k, v in CROPS.items()})

    primary_rows = [r for r in load_manifest(f"{CROPS['primary']}/manifest.csv")
                    if int(r["n_frames"]) > 0]

    # Identical split call to training: same seed, same grouping key, so the test
    # subjects are the same ones the model never saw.
    splits = split_by_group(primary_rows, train_frac=0.6, val_frac=0.2,
                            seed=args.seed, group_key="subject")
    print(summarise_by(splits, group_key="subject"))
    print("test subjects:", sorted({r["subject"] for r in splits["test"]}))

    external = {}
    for name in ("real_vs_fake", "printout_masks", "lighting"):
        if name in CROPS:
            rows = [r for r in load_manifest(f"{CROPS[name]}/manifest.csv")
                    if int(r["n_frames"]) > 0]
            external[name] = rows
            print(f"external {name}: {len(rows)} clips")

    device = pick_device()
    print(f"device: {device}\n")

    crops_dirs = {"primary": CROPS["primary"], **{k: CROPS[k] for k in external}}
    os.makedirs(args.out, exist_ok=True)
    reports = {}

    for ckpt_path in sorted(glob.glob(os.path.join(models_root, "*_best.pt"))):
        run_id = os.path.basename(ckpt_path).replace("_best.pt", "")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = build_model(cfg["model"]).to(device)
        model.load_state_dict(ckpt["model"])
        mode = "frame" if cfg["model"]["arch"] == "cnn" else "sequence"

        print("=" * 72)
        print(f"{run_id}  (arch={cfg['model']['arch']}, best epoch {ckpt.get('epoch')}, "
              f"val ACER {ckpt.get('val_acer', float('nan')) * 100:.2f}%)")
        print("=" * 72)

        rep = evaluate_all(model, splits, external, crops_dirs, mode,
                           cfg["data"].get("sequence_length", args.seq_len), device)
        rep["arch"] = cfg["model"]["arch"]
        rep["best_epoch"] = ckpt.get("epoch")
        rep["val_acer_at_checkpoint"] = ckpt.get("val_acer")
        reports[run_id] = rep
        save_report(rep, os.path.join(args.out, f"{run_id}.json"))
        print(format_report(rep))
        print()

    with open(os.path.join(args.out, "all_reports.json"), "w") as fh:
        json.dump(reports, fh, indent=2)
    print(f"wrote {args.out}/all_reports.json")


if __name__ == "__main__":
    sys.exit(main())
