"""Export a trained checkpoint to ONNX for CPU serving.

The deployed container runs onnxruntime, not torch: ~50 MB instead of ~2.5 GB, and
faster on CPU. Export is therefore a required step, not an optimisation.
"""
from __future__ import annotations

import numpy as np
import torch

from ml.liveness.models import build_model


def export(ckpt_path: str, out_path: str, seq_len: int = 8, image_size: int = 112,
           opset: int = 17) -> str:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    is_seq = cfg["model"]["arch"] != "cnn"
    dummy = (torch.randn(1, seq_len, 3, image_size, image_size) if is_seq
             else torch.randn(1, 3, image_size, image_size))
    dynamic = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(model, dummy, out_path, opset_version=opset,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes=dynamic)

    # Verify rather than assume: a silently wrong export produces a model that runs
    # and returns different numbers, which is worse than one that fails to load.
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    diff = float(np.max(np.abs(onnx_out.reshape(-1) - torch_out.reshape(-1))))
    if diff > 1e-3:
        raise RuntimeError(f"ONNX output diverges from torch by {diff:.6f}")
    print(f"exported {out_path}  (max |onnx - torch| = {diff:.2e})")
    return out_path
