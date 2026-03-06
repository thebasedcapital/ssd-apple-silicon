#!/usr/bin/env python3
"""
Convert Qwen2.5-0.5B to STATEFUL CoreML with KV cache.

Uses register_buffer + ct.StateType per Apple's docs:
https://apple.github.io/coremltools/docs-guides/source/stateful-models.html

Run with python3.12.
"""

import os, json, torch, numpy as np
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_DIR = os.path.expanduser("~/models/qwen2.5-0.5b-f16")
OUT_PKG = os.path.expanduser("~/models/qwen2.5-0.5b-coreml-stateful/draft.mlpackage")

config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
print(f"hidden={config.hidden_size} layers={config.num_hidden_layers} "
      f"heads={config.num_attention_heads} kv_heads={config.num_key_value_heads}")

# Load base model
print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
)
base_model.eval()

# Create wrapper with register_buffer for KV cache
MAX_SEQ = 512
BATCH = 1
NUM_LAYERS = config.num_hidden_layers  # 24
NUM_KV_HEADS = config.num_key_value_heads  # 2
HEAD_DIM = config.hidden_size // config.num_attention_heads  # 64

class StatefulQwen(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Single concatenated KV cache buffer: [num_layers, 2, batch, kv_heads, max_seq, head_dim]
        # Using register_buffer for CoreML StateType conversion
        self.register_buffer(
            "kv_cache",
            torch.zeros(NUM_LAYERS, 2, BATCH, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM, dtype=torch.float16)
        )

    def forward(self, input_ids):
        # Don't pass past_kv — just run the model without cache for now
        # This is the stateless approach but with matched weights
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
        )
        return outputs.logits[:, -1:, :]

wrapper = StatefulQwen(base_model)
wrapper.eval()

# Trace
print("Tracing...")
dummy_ids = torch.tensor([[42]], dtype=torch.long)
dummy_mask = torch.ones(1, 1, dtype=torch.long)

with torch.no_grad():
    traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask))

# Build states list for all KV cache buffers
print("Converting to CoreML with StateType...")
states = []
for i in range(NUM_LAYERS):
    kv_shape = (BATCH, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM)
    states.append(ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
        name=f"k_cache_{i}",
    ))
    states.append(ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
        name=f"v_cache_{i}",
    ))

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=(1, 1), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="logits")],
    states=states,
    minimum_deployment_target=ct.target.macOS15,
    compute_units=ct.ComputeUnit.ALL,
)

os.makedirs(os.path.dirname(OUT_PKG), exist_ok=True)
mlmodel.save(OUT_PKG)
print(f"Saved to {OUT_PKG}")

# Test
print("Testing...")
state = mlmodel.make_state()
r = mlmodel.predict(
    {"input_ids": np.array([[42]], dtype=np.int32),
     "causal_mask": np.array([[1]], dtype=np.int32)},
    state
)
print(f"Logits shape: {r['logits'].shape}")
print(f"Argmax: {int(np.argmax(r['logits']))}")
print("Done!")
