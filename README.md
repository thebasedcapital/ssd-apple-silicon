# SSD on Apple Silicon: Can We Beat Speculative Decoding?

**TL;DR: No.** MLX speculative decoding at **66.9 tok/s** is the practical ceiling on M4 16GB. We tried every approach — SSD, ANE parallelism, disaggregated inference, Eagle-3, CoreML bridging — nothing beats plain MLX spec decode. Here's everything we learned.

## The Question

[Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251) (Kumar, Dao, May 2026) achieves 2x over standard spec decode on H100 GPUs by running draft and target models on separate hardware in parallel. Can we do the same on Apple Silicon using the Neural Engine (ANE) + GPU?

## Hardware

- Apple M4, 16GB unified memory, 10-core GPU, 16-core ANE
- Memory bandwidth: ~100 GB/s
- Models: Qwen2.5-3B-Instruct (4-bit target), Qwen2.5-0.5B-Instruct (4-bit draft)

## Results

### Final Benchmark (4-prompt average, 128 tokens)

| Method | tok/s | vs Vanilla |
|--------|-------|-----------|
| Vanilla MLX 3B | 47.0 | 1.00x |
| **MLX Spec Decode (d=2)** | **66.9** | **1.42x** |
| Eagle-3 (Qwen3-1.7B) | 54.5 | 1.10x* |
| Disaggregated ANE+GPU | 42.3 | 0.90x |
| CoreML 3B decode | 1.8 | 0.04x |

*Different model (Qwen3 vs Qwen2.5), not directly comparable.

### Why 66.9 tok/s Is the Ceiling

We're consuming **93% of memory bandwidth** (~93 of 100 GB/s). LLM token generation is memory-bandwidth bound, not compute bound. No software optimization can exceed the hardware limit.

## What We Proved (Technical Contributions)

### 1. ANE + GPU True Parallelism
ANE and GPU execute concurrently on M4 with **73% overlap efficiency** and **1.32x speedup** on paired workloads.

```
Sequential (ANE then GPU): 22.1 ms
Parallel   (ANE || GPU):   16.8 ms
Speedup: 1.32x
```

**Binary:** `ssd_ane_gpu_test.swift`

### 2. CoreML ↔ MLX KV Cache Handoff
CoreML outputs KV cache as numpy arrays, which inject into MLX's cache in **0.2ms** with exact token match.

```python
# CoreML prefill → numpy → MLX cache injection
for i in range(num_layers):
    mlx_cache[i].state = (mx.array(k_data[i]), mx.array(v_data[i]))
    mlx_cache[i].offset = seq_len
# Cost: 0.2ms, tokens match perfectly
```

### 3. Same-Model SSD with Stochastic Acceptance
Using the same 0.5B model as both draft and target with stochastic sampling (T=0.8), SSD achieves **1.41x over sequential spec decode** with 67% acceptance.

**Binary:** `ssd_stochastic.swift`

### 4. CoreML Token-Matched Conversion
Patched a coremltools 9.0 bug (`ops.py:3048`) to convert MLX float16 weights to CoreML with **exact argmax match**.

### 5. CoreML Stateless Prefill with KV Output
Converted Qwen2.5 to a CoreML model that outputs KV cache tensors for disaggregated inference.

```
CoreML 0.5B prefill: 643 tok/s (batch)
CoreML 3B prefill:   270 tok/s (batch)
```

## Why Each Approach Failed

### SSD (Speculative Speculative Decoding)
**Problem:** Requires draft and target on separate hardware. On single GPU, pre-speculation always adds latency.
- CPU draft: 12.5x slower than GPU (14 vs 175 tok/s)
- ANE draft: fast (137 tok/s) but CoreML tokens don't match MLX
- Same GPU: can't overlap (sequential execution)

### ANE Draft + GPU Target
**Problem:** CoreML and MLX produce different tokens for the same weights.
- Different quantization paths → 0% greedy agreement
- Stochastic acceptance between 0.5B/3B: only 1-2%
- Even matched-weight conversion only works for stateless (no KV cache)

### Disaggregated Inference (ANE Prefill + GPU Decode)
**Problem:** CoreML prefill is slower than MLX GPU for short prompts.
- CoreML 3B prefill: 148ms for 40 tokens
- MLX GPU prefill: ~50ms for 40 tokens
- ANE advantage only appears at 500+ token prompts

### Eagle-3
**Problem:** Pre-trained heads for Qwen2.5-3B don't exist. Community Qwen3 heads have low acceptance (39%).
- Built llama.cpp from PR #18039 (draft, not merged)
- Best result: 54.5 tok/s (1.10x over vanilla 49.8)
- With properly trained heads (77%+ acceptance), could reach 2-3x

## When This Research Becomes Relevant

- **M4 Ultra 192GB**: 70B target at ~5 tok/s → ANE draft overlap is massive win
- **M5 chip**: +28% bandwidth + Neural Accelerators → projected ~85 tok/s
- **Longer contexts**: ANE batch prefill advantage grows with prompt length
- **Better Eagle heads**: Properly trained heads achieve 77-81% acceptance → 2x+

## Files

### Swift (Apple-native)
| File | Purpose |
|------|---------|
| `ssd_ane_gpu_test.swift` | **Key result:** proves ANE+GPU parallel execution |
| `ssd_stochastic.swift` | Same-model SSD with stochastic acceptance (1.41x) |
| `ssd_overlap_test.swift` | ANE+CPU overlap measurement |
| `ssd_parallel.swift` | CoreML-only parallel demo |
| `ssd_e2e.swift` | End-to-end with different-sized models |
| `ssd_final.swift` | E2E with stateful + stateless models |
| `ane_bench.swift` | CoreML ANE speed benchmark |
| `ane_draft_server.swift` | CoreML draft token server (stdin/stdout) |
| `bench_disagg.swift` | CoreML prefill KV-output benchmark |
| `bench_e2e_disagg.swift` | KV export for MLX handoff |
| `bench_3b_prefill.swift` | 3B CoreML prefill speed |

### Python (MLX + conversion)
| File | Purpose |
|------|---------|
| `bench.py` | MLX vanilla vs spec decode benchmark |
| `ssd_engine.py` | MLX SSD engine (multiple iterations) |
| `convert_f16.py` | Token-matched CoreML conversion (patched coremltools) |
| `convert_prefill.py` | Stateless CoreML with KV output |
| `ssd_real_bench.py` | Real before/after comparison |

### Rust
| File | Purpose |
|------|---------|
| `ssd-rust/` | Orchestrator prototype (ANE server + timing analysis) |

## Reproducing

```bash
# Install MLX
pip install mlx-lm

# Baseline: MLX speculative decoding
KMP_DUPLICATE_LIB_OK=TRUE python3 bench.py --max-tokens 128 --num-draft 2

# ANE+GPU overlap proof (Swift)
swiftc -O -parse-as-library -framework CoreML -framework Foundation ssd_ane_gpu_test.swift -o test
./test

# Same-model SSD (Swift)
swiftc -O -parse-as-library -framework CoreML -framework Foundation ssd_stochastic.swift -o ssd
./ssd
```

## coremltools Bug Fix

**File:** `coremltools/converters/mil/frontend/torch/ops.py:3048`

The `_int` op handler crashes when `x.val` is an ndarray instead of a scalar:

```python
# Before (crashes):
res = mb.const(val=dtype(x.val), name=node.name)

# After (fixed):
val = x.val.item() if isinstance(x.val, np.ndarray) else x.val
res = mb.const(val=dtype(val), name=node.name)
```

## References

- [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251) (Kumar, Dao, May 2026)
- [EAGLE-3](https://github.com/SafeAILab/EAGLE) (SafeAI Lab)
- [Apple ReDrafter](https://machinelearning.apple.com/research/recurrent-drafter)
- [Apple Mirror-SD](https://machinelearning.apple.com/research/mirror)
- [Meta LayerSkip](https://github.com/facebookresearch/LayerSkip)
- [SqueezeBits Yetter](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx)

## License

MIT
