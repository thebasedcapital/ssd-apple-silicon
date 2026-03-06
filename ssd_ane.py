#!/usr/bin/env python3
"""
SSD with ANE draft + GPU target on Apple Silicon.

Architecture:
  - Target model: MLX on GPU (Metal)
  - Draft model: CoreML on ANE (Neural Engine)
  - True hardware parallelism via concurrent dispatch

The ANE draft runs SIMULTANEOUSLY with GPU verification.
Unified memory means zero-copy data sharing.
"""

import os
import time
import functools
import threading
from typing import Optional, List, Tuple, Callable, Any, Generator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache
import numpy as np

# CoreML for ANE draft
try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False


@dataclass
class SSDStats:
    total_tokens: int = 0
    draft_accepted: int = 0
    cache_hits: int = 0
    cache_lookups: int = 0
    rounds: int = 0
    ane_draft_time: float = 0.0
    gpu_verify_time: float = 0.0
    wall_time: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / max(self.cache_lookups, 1)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens / max(self.wall_time, 1e-9)

    @property
    def acceptance_rate(self) -> float:
        return self.draft_accepted / max(self.total_tokens, 1)

    @property
    def overlap_ratio(self) -> float:
        """How much of draft time overlaps with verify time."""
        if self.gpu_verify_time == 0:
            return 0
        return min(1.0, self.ane_draft_time / self.gpu_verify_time)


class ANEDraft:
    """CoreML draft model running on ANE."""

    def __init__(self, mlpackage_path: str):
        if not HAS_COREML:
            raise RuntimeError("coremltools not available")

        print(f"Loading CoreML draft from {mlpackage_path}...")
        self.model = ct.models.MLModel(
            mlpackage_path,
            compute_units=ct.ComputeUnit.CPU_AND_NE,  # Prefer ANE
        )
        print("CoreML draft loaded on ANE")

    def generate_tokens(self, start_token: int, num_tokens: int) -> List[int]:
        """Generate num_tokens starting from start_token. Returns token ids."""
        tokens = []
        current = start_token

        for _ in range(num_tokens):
            input_data = {"input_ids": np.array([[current]], dtype=np.int32)}
            result = self.model.predict(input_data)
            logits = result["logits"][0, -1, :]  # [vocab_size]
            next_token = int(np.argmax(logits))
            tokens.append(next_token)
            current = next_token

        return tokens


class ANEDraftThread:
    """Runs ANE draft in a background thread for true parallelism."""

    def __init__(self, ane_draft: ANEDraft):
        self.ane = ane_draft
        self._result = None
        self._thread = None
        self._start_token = None

    def start_draft(self, start_token: int, num_tokens: int):
        """Start drafting on ANE in background thread."""
        self._result = None
        self._start_token = start_token

        def _run():
            self._result = self.ane.generate_tokens(start_token, num_tokens)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def get_result(self, timeout: float = 1.0) -> Optional[Tuple[int, List[int]]]:
        """Wait for draft result. Returns (start_token, draft_tokens) or None."""
        if self._thread is None:
            return None
        self._thread.join(timeout=timeout)
        if self._result is not None:
            return (self._start_token, self._result)
        return None


def maybe_quantize_kv_cache(cache_list, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is not None:
        for c in cache_list:
            if hasattr(c, 'offset') and c.offset >= quantized_kv_start:
                if hasattr(c, 'quantize'):
                    c.quantize(kv_bits, kv_group_size)


generation_stream = mx.default_stream(mx.default_device())


def ssd_ane_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,  # MLX draft (fallback)
    *,
    ane_draft: Optional[ANEDraftThread] = None,  # ANE draft (preferred)
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable] = None,
    logits_processors: Optional[List[Callable]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """
    SSD with ANE draft model for true GPU+ANE parallelism.

    Timeline:
      Round T: [GPU: verify draft_T] || [ANE: pre-draft for round T+1]
      Round T+1: if ANE prediction correct -> use ANE tokens, skip GPU draft

    The ANE runs in a background thread. While GPU verifies, ANE generates
    the next round's draft tokens. On cache hit, we skip the MLX draft step.
    """
    stats = SSDStats()
    use_ane = ane_draft is not None

    y = prompt.astype(mx.uint32)

    if prompt_cache is None:
        model_cache = cache.make_prompt_cache(model)
        draft_cache = cache.make_prompt_cache(draft_model)
    else:
        model_cache = prompt_cache[: len(model.layers)]
        draft_cache = prompt_cache[len(model.layers) :]

    sampler = sampler or (lambda x: mx.argmax(x, axis=-1))

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    def _sample(logits):
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _prefill(m, c, y):
        while y.size > prefill_step_size:
            m(y[:prefill_step_size][None], cache=c)
            quantize_cache_fn(c)
            mx.eval([ci.state for ci in c])
            y = y[prefill_step_size:]
            mx.clear_cache()
        return y

    def _rewind_cache(num_draft, num_accept):
        cache.trim_prompt_cache(model_cache, num_draft - num_accept)
        cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate_mlx(y, num_draft):
        """Fallback: draft on GPU via MLX."""
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            logits = draft_model(y[None], cache=draft_cache)
            logits = logits[:, -1:, :]
            quantize_cache_fn(draft_cache)
            y, _ = _sample(logits.squeeze(0))
            mx.async_eval(y)
            ys.append(y)
        return mx.concatenate(ys)

    # Prefill
    draft_y = _prefill(draft_model, draft_cache, y)
    y = _prefill(model, model_cache, y)

    ntoks = 0
    num_draft = 0
    n = 0
    start_time = time.perf_counter()

    # ANE pre-speculation state
    ane_pending = False

    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            stats.rounds += 1

            # === CHECK ANE PRE-SPECULATION ===
            ane_hit = False
            if use_ane and ane_pending:
                t0 = time.perf_counter()
                ane_result = ane_draft.get_result(timeout=0.5)
                stats.ane_draft_time += time.perf_counter() - t0

                if ane_result is not None:
                    ane_start, ane_tokens = ane_result
                    current_start = draft_y[-1].item() if draft_y.size > 0 else None
                    stats.cache_lookups += 1

                    if ane_start == current_start and len(ane_tokens) >= num_draft:
                        # HIT! Use ANE draft tokens
                        draft_tokens = mx.array(ane_tokens[:num_draft], mx.uint32)
                        ane_hit = True
                        stats.cache_hits += 1

                        # Advance MLX draft cache through these tokens
                        for i in range(num_draft):
                            dt = mx.array([ane_tokens[i]], mx.uint32)
                            draft_model(dt[None], cache=draft_cache)
                            quantize_cache_fn(draft_cache)

                ane_pending = False

            # === DRAFT (MLX fallback if ANE missed) ===
            if not ane_hit:
                draft_tokens = _draft_generate_mlx(draft_y, num_draft)

            # === START ANE PRE-SPECULATION FOR NEXT ROUND ===
            # Kick off ANE draft while GPU verifies.
            # We predict the next round starts from the last draft token.
            if use_ane and draft_tokens.size > 0:
                mx.eval(draft_tokens)
                predicted_start = draft_tokens[-1].item()
                ane_draft.start_draft(predicted_start, num_draft)
                ane_pending = True

            # === VERIFY on GPU ===
            t0 = time.perf_counter()
            y = mx.concatenate([y, draft_tokens])
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -(num_draft + 1):, :]
            quantize_cache_fn(model_cache)
            tokens, logprobs = _sample(logits.squeeze(0))
            mx.eval(tokens, draft_tokens)
            stats.gpu_verify_time += time.perf_counter() - t0

            draft_tokens_list = draft_tokens.tolist()
            tokens_list = tokens.tolist()

            # === ACCEPT/REJECT ===
            n = 0
            while n < num_draft:
                tn, dtn = tokens_list[n], draft_tokens_list[n]
                if tn != dtn:
                    break
                n += 1
                ntoks += 1
                stats.total_tokens += 1
                stats.draft_accepted += 1
                yield tn, logprobs[n - 1], True
                if ntoks == max_tokens:
                    break

            if ntoks < max_tokens:
                ntoks += 1
                stats.total_tokens += 1
                yield tokens_list[n], logprobs[n], False

            if ntoks == max_tokens:
                break

            accepted_token = tokens_list[n]
            y = mx.array([accepted_token], mx.uint32)
            draft_y = y

            if n == num_draft:
                draft_y = mx.concatenate(
                    [mx.array(draft_tokens_list[-1:], mx.uint32), draft_y]
                )

            _rewind_cache(num_draft, n)

            # If ANE prediction was wrong (not all accepted), invalidate
            if use_ane and ane_pending:
                # Check if our prediction (last draft token) matches actual next start
                if n < num_draft:
                    # Rejected early — ANE predicted wrong start token
                    # ANE is still running but result will be discarded next round
                    pass

    finally:
        _rewind_cache(num_draft, n)
        stats.wall_time = time.perf_counter() - start_time

    ssd_ane_generate_step._last_stats = stats
