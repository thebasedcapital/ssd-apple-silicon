"""
Speculative Speculative Decoding (SSD) for Apple Silicon via MLX.

Based on "Speculative Speculative Decoding" (Kumar, Dao, May, 2026)

Single-GPU strategy: after verification, kick off the NEXT round's draft
generation using mx.async_eval BEFORE yielding tokens. The draft forward
passes are queued on GPU and execute during Python's yield overhead.

This is a pipelining optimization: overlap draft compute with Python overhead.
"""

import time
import functools
from typing import Optional, List, Tuple, Callable, Any, Generator
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache


@dataclass
class SSDStats:
    total_tokens: int = 0
    draft_accepted: int = 0
    cache_hits: int = 0
    cache_lookups: int = 0
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


def maybe_quantize_kv_cache(cache_list, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is not None:
        for c in cache_list:
            if hasattr(c, 'offset') and c.offset >= quantized_kv_start:
                if hasattr(c, 'quantize'):
                    c.quantize(kv_bits, kv_group_size)


def ssd_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    *,
    num_draft_tokens: int = 4,
    max_tokens: int = 256,
    sampler: Optional[Callable] = None,
    logits_processors: Optional[List[Callable]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    fan_out_base: int = 1,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """
    Pipelined spec decode: start next draft BEFORE yielding current tokens.

    Standard:  draft -> verify -> eval -> yield -> [idle] -> draft -> ...
    Pipelined: draft -> verify -> eval -> [start next draft] -> yield -> [draft completing] -> verify -> ...

    The async draft runs on GPU during Python yield overhead.
    """
    stats = SSDStats()

    y = prompt.astype(mx.uint32)
    prev_tokens = None

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

    def _draft_generate(y, num_draft):
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

    # Pipeline state: pre-drafted tokens from previous round
    pipeline_draft = None  # Pre-generated draft tokens (async, may not be eval'd yet)

    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)

            # === DRAFT ===
            if pipeline_draft is not None:
                # We already started drafting at end of last round.
                # Just eval and use the results.
                draft_tokens = pipeline_draft
                mx.eval(draft_tokens)
                pipeline_draft = None
                stats.cache_hits += 1
                stats.cache_lookups += 1
            else:
                draft_tokens = _draft_generate(draft_y, num_draft)
                stats.cache_lookups += 1

            # === VERIFY ===
            if prev_tokens is not None:
                prev_tokens = prev_tokens[: prev_tokens.size - y.size - num_draft + 1]
            y = mx.concatenate([y, draft_tokens])
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -(num_draft + 1):, :]
            quantize_cache_fn(model_cache)
            tokens, logprobs = _sample(logits.squeeze(0))
            mx.eval(tokens, draft_tokens)

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

            if prev_tokens is not None:
                prev_tokens = prev_tokens[: -max(num_draft - n, 1)]
            _rewind_cache(num_draft, n)

            # === PIPELINE: start next round's draft NOW ===
            # These forward passes are queued on GPU. They'll execute
            # while Python processes the yields above and loops back.
            next_num_draft = min(max_tokens - ntoks, num_draft_tokens)
            if next_num_draft > 0:
                pipeline_draft = _draft_generate(draft_y, next_num_draft)
                # Don't eval — let it run async while Python yields

    finally:
        _rewind_cache(num_draft, n)
        stats.wall_time = time.perf_counter() - start_time

    ssd_generate_step._last_stats = stats
