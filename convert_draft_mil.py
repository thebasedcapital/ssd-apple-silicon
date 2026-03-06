#!/usr/bin/env python3
"""
Convert Qwen2.5-0.5B to CoreML via MIL builder (no torch tracing).

Loads weights from the MLX model directory and constructs the MIL graph
directly. This avoids torch.jit.trace issues with unsupported ops.
"""

import os
import json
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
import glob


def load_mlx_weights(model_dir):
    """Load weights from MLX safetensors format."""
    import safetensors
    from safetensors.numpy import load_file

    weights = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        w = load_file(f)
        weights.update(w)
    return weights


def build_qwen2_single_layer_test():
    """
    Build a minimal CoreML model: just embeddings + 1 transformer layer + lm_head.
    Tests ANE execution before building full model.
    """
    # Load MLX model weights
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dirs = glob.glob(os.path.join(cache_dir, "models--mlx-community--Qwen2.5-0.5B-Instruct-4bit", "snapshots", "*"))
    if not model_dirs:
        print("Model not found. Run: mlx_lm.load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')")
        return

    model_dir = model_dirs[0]
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Model config: {config.get('hidden_size')=}, {config.get('num_hidden_layers')=}")
    print(f"Vocab size: {config.get('vocab_size')}")

    # For a breakthrough demo: build a trivial CoreML model that runs on ANE
    # Just embedding lookup + linear projection (lm_head)
    # This proves ANE works, then we can extend to full transformer

    vocab_size = config.get("vocab_size", 151936)
    hidden_size = config.get("hidden_size", 896)

    print(f"Building CoreML model: embed({vocab_size}x{hidden_size}) + lm_head({hidden_size}x{vocab_size})")

    # Load actual weights
    print("Loading weights...")
    weights = load_mlx_weights(model_dir)

    # Print available weight keys
    print(f"Found {len(weights)} weight tensors")
    embed_keys = [k for k in weights if "embed" in k]
    head_keys = [k for k in weights if "lm_head" in k or "head" in k]
    print(f"Embed keys: {embed_keys[:5]}")
    print(f"Head keys: {head_keys[:5]}")

    # Check if weights are quantized
    for k in sorted(weights.keys())[:10]:
        print(f"  {k}: {weights[k].shape} {weights[k].dtype}")


if __name__ == "__main__":
    build_qwen2_single_layer_test()
