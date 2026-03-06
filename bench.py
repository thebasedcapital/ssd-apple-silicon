#!/usr/bin/env python3
"""
Benchmark: Standard Spec Decoding vs SSD on Apple Silicon.

Usage:
    python3 bench.py [--target MODEL] [--draft MODEL] [--max-tokens N] [--num-draft K]
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.generate import speculative_generate_step
from ssd_engine import ssd_generate_step, SSDStats


PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function that implements binary search.",
    "What are the key differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
]


def bench_vanilla(model, tokenizer, prompt, max_tokens):
    """Baseline: no speculative decoding."""
    start = time.perf_counter()
    tokens = 0
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        tokens += 1
    elapsed = time.perf_counter() - start
    return tokens, elapsed


def bench_standard_spec(model, draft_model, tokenizer, prompt, max_tokens, num_draft):
    """Standard speculative decoding (MLX built-in)."""
    tokens_ids = tokenizer.encode(prompt)
    prompt_arr = mx.array(tokens_ids, mx.uint32)

    start = time.perf_counter()
    tokens = 0
    accepted_draft = 0
    for tok, lp, from_draft in speculative_generate_step(
        prompt_arr, model, draft_model,
        num_draft_tokens=num_draft,
        max_tokens=max_tokens,
    ):
        tokens += 1
        if from_draft:
            accepted_draft += 1
        if tok == tokenizer.eos_token_id:
            break
    elapsed = time.perf_counter() - start
    return tokens, elapsed, accepted_draft


def bench_ssd(model, draft_model, tokenizer, prompt, max_tokens, num_draft, fan_out=4):
    """SSD: speculative speculative decoding."""
    tokens_ids = tokenizer.encode(prompt)
    prompt_arr = mx.array(tokens_ids, mx.uint32)

    start = time.perf_counter()
    tokens = 0
    accepted_draft = 0
    for tok, lp, from_draft in ssd_generate_step(
        prompt_arr, model, draft_model,
        num_draft_tokens=num_draft,
        max_tokens=max_tokens,
        fan_out_base=fan_out,
    ):
        tokens += 1
        if from_draft:
            accepted_draft += 1
        if tok == tokenizer.eos_token_id:
            break
    elapsed = time.perf_counter() - start

    stats = getattr(ssd_generate_step, '_last_stats', SSDStats())
    return tokens, elapsed, accepted_draft, stats


def main():
    parser = argparse.ArgumentParser(description="SSD Benchmark")
    parser.add_argument("--target", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--draft", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-draft", type=int, default=4)
    parser.add_argument("--fan-out", type=int, default=4)
    parser.add_argument("--prompts", type=int, default=2, help="Number of prompts to test")
    parser.add_argument("--mode", choices=["all", "vanilla", "spec", "ssd"], default="all")
    args = parser.parse_args()

    print(f"Loading target: {args.target}")
    target_model, tokenizer = load(args.target)

    draft_model = None
    if args.mode in ("all", "spec", "ssd"):
        print(f"Loading draft:  {args.draft}")
        draft_model, _ = load(args.draft)

    print(f"\nConfig: max_tokens={args.max_tokens}, num_draft={args.num_draft}, fan_out={args.fan_out}")
    print("=" * 80)

    prompts = PROMPTS[:args.prompts]

    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt[:60]}...")
        print("-" * 60)

        # Warmup
        if i == 0:
            print("  Warming up...")
            for _ in stream_generate(target_model, tokenizer, "Hi", max_tokens=5):
                pass

        # Vanilla
        if args.mode in ("all", "vanilla"):
            toks, elapsed = bench_vanilla(target_model, tokenizer, prompt, args.max_tokens)
            tps = toks / elapsed
            print(f"  Vanilla:  {toks:4d} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")

        # Standard spec decode
        if args.mode in ("all", "spec") and draft_model:
            toks, elapsed, accepted = bench_standard_spec(
                target_model, draft_model, tokenizer, prompt,
                args.max_tokens, args.num_draft
            )
            tps = toks / elapsed
            acc_rate = accepted / max(toks, 1) * 100
            print(f"  Spec:     {toks:4d} tokens in {elapsed:.2f}s = {tps:.1f} tok/s  (draft accepted: {acc_rate:.0f}%)")

        # SSD
        if args.mode in ("all", "ssd") and draft_model:
            toks, elapsed, accepted, stats = bench_ssd(
                target_model, draft_model, tokenizer, prompt,
                args.max_tokens, args.num_draft, args.fan_out
            )
            tps = toks / elapsed
            acc_rate = accepted / max(toks, 1) * 100
            hit_rate = stats.cache_hit_rate * 100 if hasattr(stats, 'cache_hit_rate') else 0
            print(f"  SSD:      {toks:4d} tokens in {elapsed:.2f}s = {tps:.1f} tok/s  (draft accepted: {acc_rate:.0f}%)")

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
