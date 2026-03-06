#!/usr/bin/env python3
"""Convert dequantized float16 Qwen2.5-0.5B to CoreML. Run with python3.12."""

import os, json
import torch
import numpy as np
import coremltools as ct
from safetensors.torch import load_file

MODEL_DIR = os.path.expanduser("~/models/qwen2.5-0.5b-f16")

# Load config
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    config = json.load(f)

print(f"Config: hidden={config['hidden_size']}, layers={config['num_hidden_layers']}, "
      f"heads={config['num_attention_heads']}, kv_heads={config['num_key_value_heads']}")

# Load weights
print("Loading safetensors weights...")
weights = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
print(f"Loaded {len(weights)} tensors")

# Build a minimal single-forward-pass model using torch
# For CoreML conversion, we need a traced torch model.
# Instead of using HuggingFace transformers (which has unsupported ops),
# build a minimal Qwen2 implementation from scratch.

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class QwenAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.num_kv_heads = config['num_key_value_heads']
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = torch.nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat(1, rep, 1, 1)
            v = v.repeat(1, rep, 1, 1)

        # Scaled dot-product attention (no RoPE for simplicity — affects output but tests pipeline)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Causal mask
        if L > 1:
            mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)

class QwenMLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = torch.nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        self.up_proj = torch.nn.Linear(config['hidden_size'], config['intermediate_size'], bias=False)
        self.down_proj = torch.nn.Linear(config['intermediate_size'], config['hidden_size'], bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config['hidden_size'], config.get('rms_norm_eps', 1e-6))
        self.self_attn = QwenAttention(config)
        self.post_attention_layernorm = RMSNorm(config['hidden_size'], config.get('rms_norm_eps', 1e-6))
        self.mlp = QwenMLP(config)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class QwenModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.layers = torch.nn.ModuleList([QwenBlock(config) for _ in range(config['num_hidden_layers'])])
        self.norm = RMSNorm(config['hidden_size'], config.get('rms_norm_eps', 1e-6))
        self.lm_head = torch.nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x[:, -1:, :])  # Only last token logits

# Build model and load weights
print("Building model...")
model = QwenModel(config)

# Map safetensors keys to our model
state_dict = {}
for k, v in weights.items():
    # Map: model.layers.N.self_attn.q_proj.weight -> layers.N.self_attn.q_proj.weight
    new_k = k.replace("model.", "", 1)
    if new_k in dict(model.named_parameters()):
        state_dict[new_k] = v.to(torch.float16)

print(f"Mapped {len(state_dict)}/{len(dict(model.named_parameters()))} parameters")

# Tie lm_head to embed_tokens (Qwen2.5 shares these weights)
if "lm_head.weight" not in state_dict and "embed_tokens.weight" in state_dict:
    state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]
    print("Tied lm_head.weight to embed_tokens.weight")

model.load_state_dict(state_dict, strict=False)
model.eval()
model.half()

# Trace
print("Tracing...")
example = torch.randint(0, 1000, (1, 1))
with torch.no_grad():
    traced = torch.jit.trace(model, example)

# Convert to CoreML
print("Converting to CoreML...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32)],
    outputs=[ct.TensorType(name="logits")],
    compute_units=ct.ComputeUnit.ALL,
    minimum_deployment_target=ct.target.macOS15,
)

out_path = os.path.expanduser("~/models/qwen2.5-0.5b-coreml-matched/draft.mlpackage")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
mlmodel.save(out_path)
print(f"Saved to {out_path}")

# Test
print("Testing...")
test = {"input_ids": np.array([[42]], dtype=np.int32)}
result = mlmodel.predict(test)
logits = result["logits"]
print(f"Logits shape: {logits.shape}")
print(f"Argmax: {np.argmax(logits)}")
print("Done!")
