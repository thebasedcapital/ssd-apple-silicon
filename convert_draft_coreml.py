#!/usr/bin/env python3
"""
Convert Qwen2.5-0.5B draft model to CoreML for ANE execution.

Run with Python 3.12 venv:
    source .venv312/bin/activate
    python convert_draft_coreml.py
"""

import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def convert_single_step():
    """
    Convert draft model for single-token generation (decode step).
    The model takes: input_ids [1, 1] + past KV cache
    Returns: logits [1, 1, vocab_size] + updated KV cache
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Trace with a single token input (decode step)
    # For KV cache, we need to handle it carefully
    dummy_input = torch.randint(0, 1000, (1, 1))

    # First, trace without KV cache (prefill mode - single token)
    print("Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (dummy_input,),
            strict=False,
        )

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits"),
        ],
        compute_units=ct.ComputeUnit.ALL,  # Allow ANE
        minimum_deployment_target=ct.target.macOS15,
    )

    output_path = "draft_model_ane.mlpackage"
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")

    # Test it
    print("Testing...")
    test_input = {"input_ids": np.array([[42]], dtype=np.int32)}
    result = mlmodel.predict(test_input)
    print(f"Output shape: {result['logits'].shape}")
    print("Done!")


def convert_with_kv_cache():
    """
    Convert with explicit KV cache inputs/outputs for autoregressive generation.
    This is the proper way — each decode step takes previous KV cache and returns updated cache.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model.eval()

    num_layers = model.config.num_hidden_layers  # 24
    num_kv_heads = model.config.num_key_value_heads  # 2
    head_dim = model.config.hidden_size // model.config.num_attention_heads  # 64

    print(f"Model: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head_dim")

    # For CoreML, we'll create a wrapper that takes flattened KV cache
    class DraftModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.num_layers = model.config.num_hidden_layers
            self.num_kv_heads = model.config.num_key_value_heads
            self.head_dim = model.config.hidden_size // model.config.num_attention_heads

        def forward(self, input_ids, past_key_values_flat):
            # Reshape flat KV cache into proper structure
            # past_key_values_flat: [num_layers * 2, batch, num_kv_heads, seq_len, head_dim]
            batch = input_ids.shape[0]
            past_kv = []
            for i in range(self.num_layers):
                k = past_key_values_flat[i * 2]
                v = past_key_values_flat[i * 2 + 1]
                past_kv.append((k, v))

            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )

            # Flatten updated KV cache
            new_kv_flat = []
            for k, v in outputs.past_key_values:
                new_kv_flat.append(k)
                new_kv_flat.append(v)

            return outputs.logits, torch.stack(new_kv_flat)

    wrapper = DraftModelWrapper(model)
    wrapper.eval()

    # Create dummy inputs
    seq_len = 16  # Initial context length
    dummy_input_ids = torch.randint(0, 1000, (1, 1))
    dummy_kv = torch.zeros(num_layers * 2, 1, num_kv_heads, seq_len, head_dim, dtype=torch.float16)

    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy_input_ids, dummy_kv),
            strict=False,
        )

    print("Converting to CoreML...")
    # Use flexible shapes for KV cache seq_len dimension
    kv_shape = ct.Shape(
        shape=(num_layers * 2, 1, num_kv_heads, ct.RangeDim(1, 2048), head_dim)
    )

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="past_kv", shape=kv_shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="logits"),
            ct.TensorType(name="new_past_kv"),
        ],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )

    output_path = "draft_model_kv_ane.mlpackage"
    mlmodel.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import sys
    if "--kv" in sys.argv:
        convert_with_kv_cache()
    else:
        convert_single_step()
