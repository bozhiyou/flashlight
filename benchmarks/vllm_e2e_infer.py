"""End-to-end vLLM inference benchmark with custom attention variants.

Integrates FlexAttention and Flashlight attention variants from
``run_flex_variants.py`` into a real vLLM generation pipeline.  This provides
a realistic, full-stack counterpart to the kernel-level microbenchmarks in
``run_flex_variants.py``.

Patching strategy
-----------------
vLLM on GPU uses the FLASH_ATTN backend by default (TORCH_SDPA is CPU-only
in vLLM 0.6.x).  vLLM 0.6.x bundles its own flash-attn fork at
``vllm.vllm_flash_attn`` (it does **not** use the standalone ``flash_attn``
package).  We monkey-patch ``vllm.vllm_flash_attn.flash_attn_varlen_func``
— the function vLLM calls during prefill — **before** importing
``vllm.attention.backends.flash_attn`` so that its
``from vllm.vllm_flash_attn import flash_attn_varlen_func`` picks up the
hooked version.

Decode steps use ``flash_attn_with_kvcache`` which reads directly from the
paged KV cache; reconstructing contiguous K/V from paged blocks would add
overhead that does not reflect real usage, so decode is left unpatched and
runs with the native flash-attn kernel.  TTFT therefore captures the
variant's prefill performance while ITL reflects standard paged-attention
decode.

Usage examples
--------------

Each invocation handles one mode + variant and sweeps over hardcoded
batch_size × input_len × max_tokens configs.

Baseline (unpatched vLLM, sweeps all configs)::

    python benchmarks/vllm_e2e_infer.py --mode baseline

FlexAttention with causal mask (mask caching on by default)::

    python benchmarks/vllm_e2e_infer.py --mode flex --variant causal

FlexAttention with sliding-window mask, caching disabled::

    python benchmarks/vllm_e2e_infer.py --mode flex --variant sliding_window --no-mask-cache

Flashlight-compiled causal attention::

    python benchmarks/vllm_e2e_infer.py --mode flashlight --variant causal

Single config (override defaults)::

    python benchmarks/vllm_e2e_infer.py --mode baseline --batch-size 1 --input-len 4096 --max-tokens 128

Verify hooks reach the attention layer for a given model::

    python benchmarks/vllm_e2e_infer.py --mode debug
    python benchmarks/vllm_e2e_infer.py --mode debug --model meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Path setup – make benchmarks/ and repo-root importable
# ---------------------------------------------------------------------------
_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_BENCH_DIR)
for _p in (_BENCH_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Friendly variant name  →  registry key
# ---------------------------------------------------------------------------
FLEX_VARIANT_MAP: Dict[str, str] = {
    "causal": "flex_full_with_causal",
    "sliding_window": "flex_full_with_sliding_window",
    "prefix_lm": "flex_full_with_prefix_lm",
    "document_mask": "flex_full_with_document_mask",
    "full": "flex_full",
    "alibi": "flex_full_with_alibi",
    "softcap": "flex_full_with_softcap",
}

FLASHLIGHT_VARIANT_MAP: Dict[str, str] = {
    "causal": "full_with_causal",
    "sliding_window": "full_with_sliding_window",
    "prefix_lm": "full_with_prefix_lm",
    "document_mask": "full_with_document_mask",
    "full": "full",
    "alibi": "full_with_alibi",
    "softcap": "full_with_softcap",
}

ALL_VARIANTS = sorted(FLEX_VARIANT_MAP.keys())


def _check_gqa_compat(model_id: str) -> None:
    """Fail early if the model's GQA ratio is not a power of 2.

    FlexAttention's inductor lowering requires
    ``num_q_heads / num_kv_heads`` to be a power of 2.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    n_q = getattr(cfg, "num_attention_heads", None)
    n_kv = getattr(cfg, "num_key_value_heads", None)
    if n_q is None or n_kv is None or n_kv == 0:
        return  # can't determine — let it fail later with a clearer trace
    if n_q == n_kv:
        return  # MHA, no GQA
    ratio = n_q // n_kv
    if ratio * n_kv != n_q or (ratio & (ratio - 1)) != 0:
        raise ValueError(
            f"Model {model_id!r} has {n_q} query heads and {n_kv} KV heads "
            f"(ratio {n_q / n_kv:.1f}).  FlexAttention requires this ratio "
            f"to be a power of 2.  Use a compatible model such as "
            f"meta-llama/Llama-3.2-1B (32/8 = 4) or "
            f"TinyLlama/TinyLlama-1.1B-Chat-v1.0 (32/4 = 8)."
        )


# ---------------------------------------------------------------------------
# Attention monkey-patching  (hooks flash_attn, NOT SDPA)
# ---------------------------------------------------------------------------

def _install_flash_attn_hook(
    attn_fn: Callable[..., torch.Tensor],
    mask_mod: Optional[Callable[..., Any]] = None,
) -> None:
    """Replace ``flash_attn_varlen_func`` with *attn_fn* for prefill.

    Must be called **before** ``from vllm import LLM`` so that vLLM's own
    ``from vllm.vllm_flash_attn import flash_attn_varlen_func`` resolves to
    the hook.

    vLLM 0.6.x bundles its own flash-attn fork at ``vllm.vllm_flash_attn``
    and never imports the standalone ``flash_attn`` package.  We patch both
    the interface sub-module and the package ``__init__`` so that any
    ``from vllm.vllm_flash_attn import flash_attn_varlen_func`` executed
    later (e.g. inside ``vllm.attention.backends.flash_attn``) picks up
    the hooked version.

    Only uniform-length batches (the benchmark scenario) are intercepted;
    variable-length or unexpected shapes fall back to the original kernel.
    """
    try:
        import vllm.vllm_flash_attn as vfa
        import vllm.vllm_flash_attn.flash_attn_interface as fai
    except ImportError:
        raise RuntimeError(
            "vllm with bundled vllm_flash_attn is required.  "
            "Install with:  pip install 'vllm>=0.6.4,<0.7.0'"
        )

    _orig = fai.flash_attn_varlen_func

    # Minimum seqlen to intercept.  Anything below this is a vLLM-internal
    # call (e.g. memory-profiling with seqlen=9) that the compiled kernel
    # cannot handle (Triton requires power-of-2 arange).  Fall back to
    # the original flash-attn kernel for these calls — they don't affect
    # benchmark results.
    _MIN_BENCH_SEQLEN = 256

    def _hooked(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        **kwargs: Any,
    ) -> Any:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]

        # Fall back to the original kernel for non-benchmark calls:
        #  - vLLM profiling pass (tiny seqlens, e.g. 9)
        #  - variable-length / packed batches
        if (batch_size == 0
                or total_q != batch_size * max_seqlen_q
                or max_seqlen_q < _MIN_BENCH_SEQLEN):
            return _orig(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p, softmax_scale=softmax_scale,
                causal=causal, **kwargs,
            )

        nheads_q, headdim = q.shape[1], q.shape[2]

        # (total, H, D) → (B, S, H, D) → (B, H, S, D)
        q_4d = q.view(batch_size, max_seqlen_q, nheads_q, headdim).transpose(1, 2).contiguous()
        k_4d = k.view(batch_size, max_seqlen_k, k.shape[1], headdim).transpose(1, 2).contiguous()
        v_4d = v.view(batch_size, max_seqlen_k, v.shape[1], headdim).transpose(1, 2).contiguous()

        # vLLM passes flash-attn-specific kwargs (window_size, causal,
        # alibi_slopes, softcap) — we intentionally drop them because
        # the variant supplies its own attention pattern via mask_mod /
        # score_mod, just like benchmarks/run_flex_variants.py.
        fn_kwargs: Dict[str, Any] = {}
        if mask_mod is not None:
            fn_kwargs["mask_mod"] = mask_mod

        # Detect GQA: when query has more heads than key/value,
        # flex_attention requires enable_gqa=True.
        if q_4d.shape[1] != k_4d.shape[1]:
            fn_kwargs["enable_gqa"] = True

        result = attn_fn(q_4d, k_4d, v_4d, **fn_kwargs)

        # (B, H, S, D) → (B, S, H, D) → (total, H, D)
        result_flat = result.transpose(1, 2).reshape(-1, nheads_q, headdim)

        # vLLM passes a pre-allocated ``out`` buffer; write into it.
        out_buf = kwargs.get("out")
        if out_buf is not None:
            out_buf.copy_(result_flat)
            return out_buf
        return result_flat

    # Patch both the interface sub-module and the package __init__ so that
    # ``from vllm.vllm_flash_attn import flash_attn_varlen_func`` (executed
    # later by vllm.attention.backends.flash_attn) binds to the hook.
    fai.flash_attn_varlen_func = _hooked  # type: ignore[assignment]
    vfa.flash_attn_varlen_func = _hooked  # type: ignore[assignment]


def apply_flex_patch(variant: str, mask_cache: bool, max_seq_len: int) -> None:
    """Hook ``flash_attn_varlen_func`` to route prefill through a FlexAttention
    variant.

    Parameters
    ----------
    variant : str
        Friendly variant name (e.g. ``"causal"``).
    mask_cache : bool
        When *True* the ``create_block_mask_cached`` LRU cache is active;
        when *False* we swap it with the uncached ``create_block_mask`` so
        every call rebuilds the block mask (useful for measuring mask-creation
        overhead).
    max_seq_len : int
        Upper-bound sequence length used to pre-build the ``mask_mod`` (for
        variants that derive document offsets from sequence length).
    """
    import run_flex_variants as rv

    flex_key = FLEX_VARIANT_MAP[variant]
    flex_fn = rv.FLEX_ATTENTION_REGISTRY[flex_key]
    mask_factory = rv.FLEX_MASK_REGISTRY.get(flex_key)

    # Toggle mask caching at the *module* level so that the registry lambdas
    # (which close over the module-global ``create_block_mask_cached``) see
    # the change.
    if not mask_cache:
        rv.create_block_mask_cached = rv.create_block_mask  # type: ignore[assignment]
    else:
        try:
            rv.create_block_mask_cached.cache_clear()  # type: ignore[attr-defined]
        except AttributeError:
            pass

    # Pre-build the mask_mod once so that the LRU cache can recognise the same
    # function object across calls (closures like ``generate_sliding_window``
    # would otherwise produce a new object each time).
    _mask_mod: Optional[Callable[..., Any]] = None
    if mask_factory is not None:
        from _utils import Config
        cfg = Config(1, max_seq_len, 1, 64, 1, 0.0)
        _mask_mod = mask_factory(cfg)

    _install_flash_attn_hook(flex_fn, _mask_mod)
    print(
        f"[patch] flash_attn_varlen_func → FlexAttention variant={variant} "
        f"(key={flex_key}), mask_cache={mask_cache}"
    )


def apply_flashlight_patch(variant: str) -> None:
    """Apply Flashlight compiler patches and hook ``flash_attn_varlen_func``
    to route prefill through a Flashlight-compiled variant.
    """
    from run_flex_variants import _torch_compile_attn

    fl_registry = _torch_compile_attn(enable_flashlight=True)

    # Work around inductor pad_mm bug: _should_pad benchmarks BMM ops with
    # mismatched dtypes (float32 vs bfloat16) when max_autotune is on.
    # Instead of disabling shape_padding globally (which hurts Triton tile-size
    # optimization), catch the RuntimeError only for broken ops.
    import torch._inductor.fx_passes.pad_mm as _pad_mm
    _orig_should_pad = _pad_mm.should_pad

    def _safe_should_pad(*args, **kwargs):
        try:
            return _orig_should_pad(*args, **kwargs)
        except RuntimeError:
            return False  # skip padding if benchmark fails (dtype bug)

    _pad_mm.should_pad = _safe_should_pad

    fl_key = FLASHLIGHT_VARIANT_MAP[variant]
    fl_fn = fl_registry[fl_key]

    _install_flash_attn_hook(fl_fn, mask_mod=None)
    print(f"[patch] flash_attn_varlen_func → Flashlight variant={variant} (key={fl_key})")


# ---------------------------------------------------------------------------
# Debug mode – verify hooks reach attention for a given model
# ---------------------------------------------------------------------------

def _install_debug_hook() -> List[Dict[str, Any]]:
    """Install a pass-through tracing hook and return the shared call log.

    Every call to ``flash_attn_varlen_func`` is logged with tensor shapes and
    kwargs, then forwarded to the original kernel so the model still works.
    """
    import vllm.vllm_flash_attn as vfa
    import vllm.vllm_flash_attn.flash_attn_interface as fai

    _orig = fai.flash_attn_varlen_func
    call_log: List[Dict[str, Any]] = []

    def _tracing(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        **kwargs: Any,
    ) -> Any:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        call_log.append({
            "q_shape": tuple(q.shape),
            "k_shape": tuple(k.shape),
            "batch_size": int(batch_size),
            "max_seqlen_q": int(max_seqlen_q),
            "max_seqlen_k": int(max_seqlen_k),
            "uniform": total_q == batch_size * max_seqlen_q,
            "has_out": "out" in kwargs,
            "has_block_table": (
                "block_table" in kwargs
                and kwargs["block_table"] is not None
            ),
            "extra_kwargs": sorted(
                k for k in kwargs if k not in ("out", "block_table")
            ),
        })
        return _orig(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k, **kwargs,
        )

    fai.flash_attn_varlen_func = _tracing  # type: ignore[assignment]
    vfa.flash_attn_varlen_func = _tracing  # type: ignore[assignment]
    return call_log


def run_debug(args: argparse.Namespace) -> None:
    """Load *model*, generate a few tokens, and print a hook trace."""
    call_log = _install_debug_hook()

    from vllm import LLM, SamplingParams

    batch_size = args.batch_size[0] if isinstance(args.batch_size, list) else args.batch_size
    input_len = args.input_len[0] if isinstance(args.input_len, list) else args.input_len
    max_tokens = args.max_tokens[0] if isinstance(args.max_tokens, list) else args.max_tokens
    max_seq_len = input_len + max_tokens
    print(f"[debug] Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        max_model_len=max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    profiling_calls = len(call_log)

    prompts = build_prompts(batch_size, input_len)
    outputs = llm.generate(
        prompts, SamplingParams(max_tokens=8, temperature=0.0),
    )
    gen_calls = len(call_log) - profiling_calls
    out_toks = sum(len(o.outputs[0].token_ids) for o in outputs)

    # ---- summary ----
    print(f"\n{'=' * 60}")
    print(f"  Model:           {args.model}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Input tokens:    {input_len}")
    print(f"  Output tokens:   {out_toks}")
    print(f"{'=' * 60}")
    print(f"  flash_attn_varlen_func calls")
    print(f"    profiling (model init):  {profiling_calls}")
    print(f"    generation (prefill):    {gen_calls}")
    print(f"    total:                   {len(call_log)}")

    if call_log:
        first = call_log[profiling_calls] if gen_calls else call_log[0]
        all_uniform = all(c["uniform"] for c in call_log)
        any_block_table = any(c["has_block_table"] for c in call_log)
        print(f"  All uniform-length?        {all_uniform}")
        print(f"  Any block_table (paged)?   {any_block_table}")
        print(f"  Head config (first gen call):")
        print(f"    q heads (Hq):  {first['q_shape'][1]}")
        print(f"    kv heads (Hkv): {first['k_shape'][1]}")
        print(f"    head dim:      {first['q_shape'][2]}")
        print(f"    extra kwargs:  {first['extra_kwargs']}")
    else:
        print("  WARNING: hook was never called — "
              "this model may not use flash_attn_varlen_func")

    hookable = len(call_log) > 0 and all(c["uniform"] for c in call_log)
    print(f"{'=' * 60}")
    if hookable:
        print("  RESULT: Hook is reachable. "
              "flex / flashlight modes will work for this model.")
    else:
        print("  RESULT: Hook may NOT work for this model. "
              "Check the trace above.")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompts(
    batch_size: int,
    input_len: int,
    vocab_size: int = 10000,
) -> List[Dict[str, Any]]:
    """Return *batch_size* prompts each having exactly *input_len* tokens.

    Uses random token IDs (same approach as vLLM's ``benchmark_latency.py``)
    so that prompt construction is tokenizer-independent and gives exact
    control over input length.  The actual token content does not affect
    attention-kernel performance.
    """
    import numpy as np
    rng = np.random.RandomState(0)
    token_ids = rng.randint(0, vocab_size, size=(batch_size, input_len))
    return [{"prompt_token_ids": ids.tolist()} for ids in token_ids]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    llm: Any,
    prompts: List[Dict[str, Any]],
    max_tokens: int,
    *,
    warmup: int = 2,
    repeats: int = 3,
) -> Dict[str, float]:
    """Run generation and return timing metrics.

    Returns a dict with:
      total_s          – mean wall-clock time for full generation
      ttft_s           – mean time-to-first-token (single output token)
      decode_s         – estimated decode time (total - ttft)
      mean_itl_ms      – mean inter-token latency in milliseconds
      tokens_per_s     – output throughput (tokens / total time)
      num_output_tokens – mean total output tokens across the batch
    """
    from vllm import SamplingParams

    sp_full = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    sp_one = SamplingParams(max_tokens=1, temperature=0.0)

    # ---- warmup ----
    for _ in range(warmup):
        llm.generate(prompts, sp_full)

    # ---- TTFT: generate exactly 1 token (~ prefill + 1 decode step) ----
    ttft_samples: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp_one)
        torch.cuda.synchronize()
        ttft_samples.append(time.perf_counter() - t0)

    # ---- full generation ----
    total_samples: List[float] = []
    output_tok_counts: List[int] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sp_full)
        torch.cuda.synchronize()
        total_samples.append(time.perf_counter() - t0)
        output_tok_counts.append(
            sum(len(o.outputs[0].token_ids) for o in outputs)
        )

    mean_ttft = sum(ttft_samples) / len(ttft_samples)
    mean_total = sum(total_samples) / len(total_samples)
    mean_out_toks = sum(output_tok_counts) / len(output_tok_counts)
    batch_size = len(prompts)

    # Decode-only time ~ total - ttft.  The TTFT run produces 1 token per
    # request, so the difference covers the remaining (max_tokens - 1) decode
    # steps per request.
    decode_s = max(mean_total - mean_ttft, 0.0)
    decode_tokens = mean_out_toks - batch_size  # subtract the first tokens
    mean_itl_ms = (decode_s / decode_tokens * 1000) if decode_tokens > 0 else 0.0

    return {
        "total_s": mean_total,
        "ttft_s": mean_ttft,
        "decode_s": decode_s,
        "mean_itl_ms": mean_itl_ms,
        "tokens_per_s": mean_out_toks / mean_total if mean_total > 0 else 0.0,
        "num_output_tokens": mean_out_toks,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "mode", "variant", "model", "batch_size", "input_len", "max_tokens",
    "mask_cache", "total_s", "ttft_s", "decode_s", "mean_itl_ms",
    "tokens_per_s", "num_output_tokens",
]


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    # ── Debug mode (short-circuit) ──────────────────────────────────────
    if args.mode == "debug":
        run_debug(args)
        return

    batch_sizes = args.batch_size
    input_lens = args.input_len
    max_tokens_list = args.max_tokens
    max_seq_len = max(input_lens) + max(max_tokens_list)

    # ── Validate model GQA ratio for FlexAttention ──────────────────────
    if args.mode in ("flex", "flashlight"):
        _check_gqa_compat(args.model)

    # ── Apply attention patches (before vLLM import) ─────────────────────
    if args.mode == "flex":
        apply_flex_patch(
            args.variant,
            mask_cache=args.mask_cache,
            max_seq_len=max_seq_len,
        )
    elif args.mode == "flashlight":
        apply_flashlight_patch(args.variant)

    # ── Load model via vLLM (once for all configs) ───────────────────────
    from vllm import LLM

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        max_model_len=max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # ── CSV setup ────────────────────────────────────────────────────────
    out_dir = os.path.join(_BENCH_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.output}.csv")

    write_header = not os.path.exists(out_path)
    if write_header:
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    # ── Sweep over configs ───────────────────────────────────────────────
    configs = [
        (bs, il, mt)
        for bs, il in zip(batch_sizes, input_lens)
        for mt in max_tokens_list
    ]
    variant = args.variant if args.mode != "baseline" else "n/a"

    for i, (bs, il, mt) in enumerate(configs, 1):
        print(
            f"\n[{i}/{len(configs)}] mode={args.mode}, variant={variant}, "
            f"batch={bs}, input_len={il}, max_tokens={mt}, "
            f"mask_cache={args.mask_cache}"
        )

        prompts = build_prompts(bs, il)
        metrics = run_benchmark(
            llm, prompts, mt,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        print(f"  TTFT={metrics['ttft_s']:.4f}s  "
              f"Decode={metrics['decode_s']:.4f}s  "
              f"Total={metrics['total_s']:.4f}s  "
              f"ITL={metrics['mean_itl_ms']:.2f}ms  "
              f"Tput={metrics['tokens_per_s']:.1f}tok/s")

        row = {
            "mode": args.mode,
            "variant": args.variant,
            "model": args.model,
            "batch_size": bs,
            "input_len": il,
            "max_tokens": mt,
            "mask_cache": args.mask_cache,
            **metrics,
        }
        with open(out_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

    print(f"\n{len(configs)} configs complete. Results in: {out_path}")


if __name__ == "__main__":
    ##################
    # configurations
    ##################
    # Prefill-heavy scenarios to highlight attention kernel & mask overhead.
    E2E_CONFIGS = {
        "batch_sizes": [32, 16, 8, 4, 2, 1],
        "input_lens": [512, 1024, 2048, 4096, 8192, 16384],
        "max_tokens": [256],
    }

    parser = argparse.ArgumentParser(
        description="End-to-end vLLM inference benchmark with custom attention variants.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "flex", "flashlight", "debug"],
        required=True,
        help="Execution mode.",
    )
    parser.add_argument(
        "--variant",
        choices=ALL_VARIANTS,
        default="causal",
        help="Attention variant (ignored in baseline/debug mode).",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--batch-size", type=int, nargs="+",
        default=E2E_CONFIGS["batch_sizes"],
        help="Batch size(s).",
    )
    parser.add_argument(
        "--input-len", type=int, nargs="+",
        default=E2E_CONFIGS["input_lens"],
        help="Prompt token count(s).",
    )
    parser.add_argument(
        "--max-tokens", type=int, nargs="+",
        default=E2E_CONFIGS["max_tokens"],
        help="Max new tokens per request.",
    )
    parser.add_argument(
        "--mask-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="(flex mode) Enable/disable block-mask caching.",
    )
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup iterations.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timed iterations to average.")
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable CUDA graph capture.",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory for vLLM.",
    )
    parser.add_argument(
        "--output", default="vllm_e2e",
        help="Base name for CSV in benchmarks/results/.",
    )
    args = parser.parse_args()

    main(args)
