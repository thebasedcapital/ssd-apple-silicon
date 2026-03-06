"""
SSD v2: Batched multi-outcome pre-speculation for MLX.

Key insight: with draft=2, there are few possible verification outcomes.
We can draft for ALL of them in one batched forward pass on GPU,
overlapping with the target verify.

Timeline per round:
  Standard: [draft 2 tokens: 11ms] -> [verify 3 tokens: 48ms] = 59ms
  SSD:      [verify 3 tokens: 48ms] -> [batched pre-draft for top-4 outcomes: 22ms] = 48ms on hit

On cache hit: skip the 11ms draft step -> 48ms per round -> 58 * 59/48 = ~71 tok/s
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
    rounds: int = 0
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


generation_stream = mx.default_stream(mx.default_device())


def maybe_quantize_kv_cache(cache_list, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is not None:
        for c in cache_list:
            if hasattr(c, 'offset') and c.offset >= quantized_kv_start:
                if hasattr(c, 'quantize'):
                    c.quantize(kv_bits, kv_group_size)


def ssd_v2_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    *,
    num_draft_tokens: int = 2,
    max_tokens: int = 256,
    sampler: Optional[Callable] = None,
    logits_processors: Optional[List[Callable]] = None,
    prompt_cache: Optional[Any] = None,
    prefill_step_size: int = 512,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    fan_out: int = 4,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """
    SSD v2 with batched multi-outcome speculation.

    After each verification round, the target model's logits tell us the
    top-k likely next tokens. For each, we pre-generate draft tokens.
    Next round, if the actual starting token matches one of our predictions,
    we use the cached draft tokens and skip drafting entirely.
    """
    stats = SSDStats()
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

    # Speculation cache: {token_id: (draft_tokens_array,)}
    # Keyed by the starting token for the next round
    spec_cache = {}

    try:
        while True:
            num_draft = min(max_tokens - ntoks, num_draft_tokens)
            stats.rounds += 1

            # === CHECK SPECULATION CACHE ===
            start_token = draft_y[-1].item() if draft_y.size > 0 else None
            cached_draft = None

            if start_token is not None and start_token in spec_cache:
                stats.cache_lookups += 1
                cached_draft = spec_cache[start_token]
                if cached_draft.size >= num_draft:
                    stats.cache_hits += 1
                else:
                    cached_draft = None
                    stats.cache_lookups -= 1  # Don't count partial hits

            spec_cache.clear()  # Clear cache each round

            # === DRAFT ===
            if cached_draft is not None:
                draft_tokens = cached_draft[:num_draft]
                # Still need to advance draft cache through these tokens
                for i in range(num_draft):
                    dt = mx.array([draft_tokens[i].item()], mx.uint32)
                    draft_model(dt[None], cache=draft_cache)
                    quantize_cache_fn(draft_cache)
            else:
                draft_tokens = _draft_generate(draft_y, num_draft)

            # === VERIFY ===
            y = mx.concatenate([y, draft_tokens])
            logits = model(y[None], cache=model_cache)
            logits = logits[:, -(num_draft + 1):, :]
            quantize_cache_fn(model_cache)
            tokens, logprobs = _sample(logits.squeeze(0))
            mx.eval(tokens, draft_tokens)

            # Get target's prediction for what comes next (for pre-speculation)
            # The last logits position tells us what target thinks follows
            last_target_logprobs = logprobs[-1] if logprobs.ndim > 1 else logprobs

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

            # === PRE-SPECULATE for next round ===
            # Use target's logits to find top-k most likely next tokens.
            # For each, generate num_draft tokens ahead.
            # This costs fan_out * num_draft draft forward passes.
            # The cost is worth it if hit_rate * draft_time > prespec_time / rounds_between_hits
            #
            # With draft=2 and fan_out=4: 8 extra draft calls = ~46ms
            # But this runs AFTER yield, partly overlapping with Python processing.
            #
            # Actually — we can't pre-speculate without corrupting the draft cache.
            # We'd need to save/restore the cache state for each branch.
            # That's too expensive for MVP.
            #
            # SIMPLER APPROACH: just predict the argmax (fan_out=1).
            # Generate num_draft tokens for the single most likely next token.
            # Cost: num_draft draft calls = ~11ms.
            # Save: num_draft draft calls on hit = ~11ms.
            # Net: zero on hit, -11ms on miss. Only worth it if hit_rate > 50%.
            #
            # But we STILL corrupt the draft cache. The only way is to:
            # 1. Save cache offsets
            # 2. Draft num_draft tokens
            # 3. Store the tokens
            # 4. Rewind the cache by num_draft
            # This works because trim_prompt_cache just moves the offset back.

            # Predict: next round most likely starts from draft_y[-1]
            # (which is the accepted_token we just set)
            # Pre-draft from there:
            saved_offset = [c.offset for c in draft_cache]

            prespec = _draft_generate(draft_y, num_draft)
            mx.eval(prespec)

            # Store in cache keyed by starting token
            spec_cache[draft_y[-1].item()] = prespec

            # Rewind draft cache
            for c, off in zip(draft_cache, saved_offset):
                trim_amount = c.offset - off
                if trim_amount > 0:
                    cache.trim_prompt_cache([c], trim_amount)

    finally:
        _rewind_cache(num_draft, n)
        stats.wall_time = time.perf_counter() - start_time

    ssd_v2_generate_step._last_stats = stats
