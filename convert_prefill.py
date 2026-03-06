#!/usr/bin/env python3
"""
Convert Qwen2.5-0.5B to stateless CoreML that OUTPUTS KV cache.
This enables ANE batch prefill with KV handoff to MLX.

Run with python3.12 (patched coremltools).
"""

import os, torch, numpy as np, coremltools as ct
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_DIR = os.path.expanduser("~/models/qwen2.5-0.5b-f16")
OUT = os.path.expanduser("~/models/qwen2.5-0.5b-prefill/prefill.mlpackage")

config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
NUM_LAYERS = config.num_hidden_layers  # 24
NUM_KV_HEADS = config.num_key_value_heads  # 2
HEAD_DIM = config.hidden_size // config.num_attention_heads  # 64

print(f"layers={NUM_LAYERS} kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM}")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
)
model.eval()


class PrefillWrapper(torch.nn.Module):
    """Returns logits + flattened KV cache for all layers."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids):
        out = self.m(input_ids, use_cache=True)
        logits = out.logits[:, -1:, :]  # [1, 1, vocab]

        # Concatenate all K caches into one tensor, all V into another
        # Each layer's K/V: [batch, kv_heads, seq_len, head_dim]
        ks = []
        vs = []
        for k, v in out.past_key_values:
            ks.append(k)
            vs.append(v)

        # Stack: [num_layers, batch, kv_heads, seq_len, head_dim]
        all_k = torch.stack(ks, dim=0)
        all_v = torch.stack(vs, dim=0)

        return logits, all_k, all_v


wrapper = PrefillWrapper(model)
wrapper.eval()

SEQ_LEN = 8  # Fixed prefill length for tracing
print(f"Tracing with seq_len={SEQ_LEN}...")
dummy = torch.zeros(1, SEQ_LEN, dtype=torch.long)

with torch.no_grad():
    traced = torch.jit.trace(wrapper, dummy)

    # Verify trace output shapes
    test_out = traced(dummy)
    print(f"logits: {test_out[0].shape}")
    print(f"all_k:  {test_out[1].shape}")
    print(f"all_v:  {test_out[2].shape}")

print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType("input_ids", shape=(1, SEQ_LEN), dtype=np.int32)],
    outputs=[
        ct.TensorType("logits"),
        ct.TensorType("all_k"),
        ct.TensorType("all_v"),
    ],
    minimum_deployment_target=ct.target.macOS15,
    compute_units=ct.ComputeUnit.ALL,
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
mlmodel.save(OUT)
print(f"Saved to {OUT}")

# Test
print("Testing...")
test_input = {"input_ids": np.array([[42, 100, 200, 300, 400, 500, 600, 700]], dtype=np.int32)}
result = mlmodel.predict(test_input)
print(f"logits: {result['logits'].shape}")
print(f"all_k:  {result['all_k'].shape}")
print(f"all_v:  {result['all_v'].shape}")
print(f"argmax: {int(np.argmax(result['logits']))}")
print("Done!")
