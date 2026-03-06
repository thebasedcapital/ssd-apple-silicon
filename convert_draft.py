#!/usr/bin/env python3
"""Convert Qwen2.5-0.5B to CoreML for ANE. Run with python3.12."""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoModelForCausalLM

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "draft_ane.mlpackage"

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
model.eval()

# Simple wrapper: single token in, logits out (no KV cache for MVP)
class SimpleDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids, use_cache=False)
        return out.logits[:, -1:, :]  # Last token logits only

wrapper = SimpleDecoder(model)
wrapper.eval()

# Trace with variable-length input
print("Tracing...")
example = torch.randint(0, 1000, (1, 8))
with torch.no_grad():
    traced = torch.jit.trace(wrapper, example)

print("Converting to CoreML (ANE target)...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(
        name="input_ids",
        shape=ct.Shape(shape=(1, ct.RangeDim(1, 512))),
        dtype=np.int32,
    )],
    outputs=[ct.TensorType(name="logits")],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
)

mlmodel.save(OUT)
print(f"Saved to {OUT}")

# Quick test
print("Testing...")
test = {"input_ids": np.array([[42, 100, 200]], dtype=np.int32)}
result = mlmodel.predict(test)
print(f"Logits shape: {result['logits'].shape}")
print("Done!")
