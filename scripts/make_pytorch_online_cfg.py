#!/usr/bin/env python3
"""Generate a PyTorch config with online_mode enabled — i.e. streaming without TensorRT.

WHY THIS IS NEEDED. Upstream ships three configs and only ONE has
online_mode=True:

    v0.4_hubert_cfg_pytorch.pkl        online_mode = False
    v0.4_hubert_cfg_trt.pkl           online_mode = False
    v0.4_hubert_cfg_trt_online.pkl    online_mode = True   <- TensorRT only

So as shipped, streaming appears to require TensorRT. It does not. online_mode
is a plain flag in default_kwargs, the chunked entrypoint (StreamSDK.run_chunk)
is backend-agnostic Python, and the streaming audio encoder it needs
(aux_models/hubert_streaming_fix_kv.onnx) is already present in ditto_pytorch.
Flipping the flag on the PyTorch config is sufficient — measured working on an
RTX 5090 with no TensorRT installed at all.

The guard that matters is in the SDK itself:
    assert self.wav2feat.support_streaming or not self.online_mode
so a config whose audio encoder cannot stream fails loudly rather than
producing silently wrong output.

Usage:
    python scripts/make_pytorch_online_cfg.py \
        --cfg checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl
Writes <name>_online.pkl beside the input.
"""
import argparse
import copy
import pickle
from pathlib import Path


def set_first(obj, key, value):
    """Set the first occurrence of `key` anywhere in a nested dict. Returns True if found."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == key:
                obj[k] = value
                return True
            if set_first(v, key, value):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="path to v0.4_hubert_cfg_pytorch.pkl")
    ap.add_argument("--out", default=None, help="output path (default: <cfg stem>_online.pkl)")
    a = ap.parse_args()

    src = Path(a.cfg)
    cfg = pickle.load(src.open("rb"))
    if not set_first(cfg, "online_mode", True):
        raise SystemExit(
            "online_mode key not found in this config — upstream may have "
            "restructured it. Do NOT guess: inspect the pickle and re-anchor."
        )
    dst = Path(a.out) if a.out else src.with_name(src.stem + "_online.pkl")
    pickle.dump(cfg, dst.open("wb"))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
