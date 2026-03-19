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
batch_size × input_len × output_len configs.

Baseline (unpatched vLLM, sweeps all configs)::

    python benchmarks/vllm_e2e_infer.py --mode baseline

FlexAttention with causal mask (mask caching on by default)::

    python benchmarks/vllm_e2e_infer.py --mode flex --variant causal

FlexAttention with sliding-window mask, caching disabled::

    python benchmarks/vllm_e2e_infer.py --mode flex --variant sliding_window --no-mask-cache

Flashlight-compiled causal attention::

    python benchmarks/vllm_e2e_infer.py --mode flashlight --variant causal

Single config (override defaults)::

    python benchmarks/vllm_e2e_infer.py --mode baseline --batch-size 1 --input-len 4096 --output-len 128

Verify hooks reach the attention layer for a given model::

    python benchmarks/vllm_e2e_infer.py --mode debug
    python benchmarks/vllm_e2e_infer.py --mode debug --model meta-llama/Llama-3.2-1B

Trace-driven mode (Mooncake JSONL trace with variable-length requests)::

    python benchmarks/vllm_e2e_infer.py --mode flashlight --variant causal \
        --trace benchmarks/traces/conversation_trace.jsonl \
        --max-requests 50 --max-input-len 8192
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

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except (ValueError, KeyError):
        return  # unrecognised model type — skip check, let vLLM handle it
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
            f"Qwen/Qwen2.5-3B (16/2 = 8)."
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

    Input formats handled
    ---------------------
    The hook converts whatever vLLM passes into dense ``(B, H, S, D)``
    tensors before calling *attn_fn*.  Three input shapes are supported:

    1. **Uniform-length packed** (synthetic benchmarks) — Q/K/V are 3-D
       ``(total, heads, dim)`` with ``total == B * max_seqlen``.
       Converted via zero-copy view reshape.
    2. **Variable-length packed** (trace-driven workloads where the vLLM
       scheduler batches multiple prefills) — Q/K/V are 3-D but
       ``total != B * max_seqlen``.  Converted via pad-and-unpad.
    3. **Paged KV cache** (chunked prefill, enabled by default for
       ``max_model_len > 32 K``) — Q is 3-D packed, but K/V arrive as
       the **entire paged cache** ``(num_blocks, block_size, heads, dim)``
       together with a ``block_table`` in *kwargs*.  Contiguous K/V are
       reconstructed by gathering pages via ``block_table``, following the
       same principle as PyTorch's official ``PagedAttention`` wrapper in
       ``attention-gym`` (``attn_gym.paged_attention``): neither
       FlexAttention nor any compiled attention kernel receives raw paged
       blocks — a translation layer always reconstructs dense tensors or
       remaps indices first.

    Decode steps use ``flash_attn_with_kvcache`` (a separate function),
    so they never reach this hook.
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

    def _reconstruct_paged_kv(
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        batch_size: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Gather contiguous KV from a paged cache.

        Args:
            kv_cache: (num_blocks, block_size, num_kv_heads, head_dim)
            block_table: (B, max_pages_per_seq)  — physical page indices
            cu_seqlens_k: (B+1,) — cumulative KV lengths
            batch_size: number of sequences
            max_seqlen_k: max KV length in the batch

        Returns:
            (B, max_seqlen_k, num_kv_heads, head_dim) — zero-padded contiguous KV
        """
        block_size = kv_cache.shape[1]
        nheads_k = kv_cache.shape[2]
        headdim = kv_cache.shape[3]
        out = kv_cache.new_zeros(batch_size, max_seqlen_k, nheads_k, headdim)
        for i in range(batch_size):
            sk = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
            num_pages = (sk + block_size - 1) // block_size
            pages = block_table[i, :num_pages]           # physical page indices
            kv_pages = kv_cache[pages]                    # (num_pages, block_size, H, D)
            out[i, :sk] = kv_pages.reshape(-1, nheads_k, headdim)[:sk]
        return out

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

        # Fall back to the original kernel for profiling / tiny-seqlen calls
        if batch_size == 0 or max_seqlen_q < _MIN_BENCH_SEQLEN:
            return _orig(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p, softmax_scale=softmax_scale,
                causal=causal, **kwargs,
            )

        nheads_q, headdim = q.shape[1], q.shape[2]
        block_table = kwargs.get("block_table")
        paged = k.ndim == 4 and block_table is not None

        # ── Q: always 3D (total_q, nheads_q, headdim) ──────────────────
        uniform_q = (total_q == batch_size * max_seqlen_q)
        if uniform_q:
            q_4d = q.view(batch_size, max_seqlen_q, nheads_q, headdim) \
                    .transpose(1, 2).contiguous()
        else:
            q_4d = q.new_zeros(batch_size, max_seqlen_q, nheads_q, headdim)
            for i in range(batch_size):
                sq = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                q_4d[i, :sq] = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
            q_4d = q_4d.transpose(1, 2).contiguous()  # (B, H, S, D)

        # ── K/V: 3D packed  OR  4D paged cache ─────────────────────────
        if paged:
            # k/v: (num_blocks, block_size, num_kv_heads, head_dim)
            # Reconstruct contiguous K/V via block_table
            k_4d = _reconstruct_paged_kv(
                k, block_table, cu_seqlens_k, batch_size, max_seqlen_k,
            ).transpose(1, 2).contiguous()  # (B, H, S, D)
            v_4d = _reconstruct_paged_kv(
                v, block_table, cu_seqlens_k, batch_size, max_seqlen_k,
            ).transpose(1, 2).contiguous()
            nheads_k = k.shape[2]
        else:
            nheads_k = k.shape[1]
            total_k = k.shape[0]
            uniform_k = (total_k == batch_size * max_seqlen_k)
            if uniform_q and uniform_k:
                k_4d = k.view(batch_size, max_seqlen_k, nheads_k, headdim) \
                        .transpose(1, 2).contiguous()
                v_4d = v.view(batch_size, max_seqlen_k, nheads_k, headdim) \
                        .transpose(1, 2).contiguous()
            else:
                k_4d = k.new_zeros(batch_size, max_seqlen_k, nheads_k, headdim)
                v_4d = v.new_zeros(batch_size, max_seqlen_k, nheads_k, headdim)
                for i in range(batch_size):
                    sk = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
                    k_4d[i, :sk] = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                    v_4d[i, :sk] = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                k_4d = k_4d.transpose(1, 2).contiguous()  # (B, H, S, D)
                v_4d = v_4d.transpose(1, 2).contiguous()

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

        # flex_attention with enable_gqa=True returns 5D:
        # (B, nheads_k, GQA_ratio, S, D) → merge to (B, nheads_q, S, D)
        if result.ndim == 5:
            result = result.reshape(batch_size, nheads_q, -1, headdim)

        # ── Flatten result back to (total_q, nheads_q, headdim) ────────
        if uniform_q:
            result_flat = result.transpose(1, 2).reshape(-1, nheads_q, headdim)
        else:
            result_bhsd = result.transpose(1, 2)  # (B, S, H, D)
            result_flat = q.new_empty(total_q, nheads_q, headdim)
            for i in range(batch_size):
                sq = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                result_flat[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = result_bhsd[i, :sq]

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

    # Disable inductor shape_padding (pad_mm): the cat ops it inserts to
    # pad non-aligned BMM dimensions produce graph patterns that
    # Flashlight's block reduction codegen cannot handle.  Flashlight does
    # its own tiling, so the padding is unnecessary.
    torch._inductor.config.shape_padding = False

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
    output_len = args.output_len[0] if isinstance(args.output_len, list) else args.output_len
    max_seq_len = input_len + output_len
    print(f"[debug] Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        max_model_len=max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    profiling_calls = len(call_log)

    vocab_size = len(llm.get_tokenizer())
    prompts = synthesize_prompts(batch_size, input_len, vocab_size=vocab_size)
    outputs = llm.generate(
        prompts, SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
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

def synthesize_prompts(
    batch_size: int,
    input_len: int,
    vocab_size: int,
) -> List[Dict[str, Any]]:
    """Return prompts as random token-ID lists for synthetic workloads."""
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
    output_len: "int | List[int]",
    *,
    warmup: int = 2,
    repeats: int = 3,
) -> Dict[str, float]:
    """Run generation and return timing metrics.

    *output_len* may be a single int (uniform) or a per-request list (trace
    mode).

    Returns a dict with:
      total_s          – mean wall-clock time for full generation
      ttft_s           – mean time-to-first-token (aggregate batch prefill)
      decode_s         – estimated decode time (total - ttft)
      mean_itl_ms      – mean inter-token latency in milliseconds
      tokens_per_s     – output throughput (tokens / total time)
      num_output_tokens – mean total output tokens across the batch
    """
    from vllm import SamplingParams

    per_request = isinstance(output_len, list)
    if per_request:
        sp_full = [SamplingParams(max_tokens=ol, temperature=0.0, ignore_eos=True)
                   for ol in output_len]
    else:
        sp_full = SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True)

    # ---- warmup: trigger compilations before timed runs ----
    for w in range(warmup):
        print(f"  Warmup {w + 1}/{warmup} ...")
        llm.generate(prompts, sp_full)

    # ---- TTFT (prefill time: generate 1 token per request) ----
    sp_one = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
    ttft_samples: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts, sp_one)
        torch.cuda.synchronize()
        ttft_samples.append(time.perf_counter() - t0)
    mean_ttft = sum(ttft_samples) / len(ttft_samples)

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

    mean_total = sum(total_samples) / len(total_samples)
    mean_out_toks = sum(output_tok_counts) / len(output_tok_counts)
    batch_size = len(prompts)

    # Decode-only time ~ total - ttft.  The TTFT run produces 1 token per
    # request, so the difference covers the remaining (output_len - 1) decode
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

COMMON_INPUT_FIELDS = [
    "mode", "variant", "model", "batch_size", "input_len", "output_len",
    "mask_cache", "prefix_caching",
]

COMMON_OUTPUT_FIELDS = [
    "total_s", "ttft_s", "decode_s", "mean_itl_ms",
    "tokens_per_s", "num_output_tokens",
]

def _csv_setup(args: argparse.Namespace, csv_fields: List[str]) -> str:
    """Create results dir and return the CSV path, writing header if needed."""
    out_dir = os.path.join(_BENCH_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    output_name = getattr(args, "output", None)
    if output_name is None:
        if args.trace is not None:
            output_name = "vllm_e2e_online" if args.online else "vllm_e2e_offline"
        else:
            output_name = "vllm_e2e_synth"
            
    out_path = os.path.join(out_dir, f"{output_name}.csv")
    if not os.path.exists(out_path):
        with open(out_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()
    return out_path


def _append_row(out_path: str, row: Dict[str, Any], csv_fields: List[str]) -> None:
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writerow(row)


class BaseTraceWorkload:
    csv_fields = COMMON_INPUT_FIELDS + [
        "trace", "num_requests", "total_input_tokens",
    ] + COMMON_OUTPUT_FIELDS

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.records = self._load_trace(args.trace,
                             max_requests=args.max_requests,
                             max_input_len=args.max_input_len)
        if not self.records:
            raise RuntimeError("No trace records after filtering.")
        
        self.input_lens = [r["input_length"] for r in self.records]
        self.output_lens = [r["output_length"] for r in self.records]
        self.hash_ids_list = [r.get("hash_ids", []) for r in self.records]
        self._max_seq_len = max(il + ol for il, ol in zip(self.input_lens, self.output_lens))

    @staticmethod
    def _load_trace(
        trace_path: str,
        max_requests: Optional[int] = None,
        max_input_len: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load a Mooncake JSONL trace file.

        Returns a list of ``{"timestamp": float, "input_length": int, "output_length": int}`` dicts,
        optionally filtered by *max_input_len* and truncated to *max_requests*.
        """
        import json
        records: List[Dict[str, Any]] = []
        with open(trace_path) as f:
            for line in f:
                rec = json.loads(line)
                il, ol = rec["input_length"], rec["output_length"]
                if max_input_len is not None and il > max_input_len:
                    continue
                records.append({
                    "timestamp": rec.get("timestamp", 0.0),
                    "input_length": il,
                    "output_length": ol,
                    "hash_ids": rec.get("hash_ids", []),
                })
        if max_requests is not None:
            records = records[:max_requests]
        return records

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def _build_row(self, variant: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        return {
            "mode": self.args.mode,
            "variant": variant,
            "model": self.args.model,
            "batch_size": len(self.records),
            "input_len": int(np.median(self.input_lens)),
            "output_len": int(np.median(self.output_lens)),
            "mask_cache": self.args.mask_cache,
            "prefix_caching": self.args.enable_prefix_caching,
            **metrics,
            "trace": os.path.basename(self.args.trace),
            "num_requests": len(self.records),
            "total_input_tokens": sum(self.input_lens),
        }
    
    @staticmethod
    def build_prompts_from_trace(
        input_lens: List[int],
        vocab_size: int,
        hash_ids_list: Optional[List[List[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Return prompts as random token-ID lists for trace-driven workloads.
        
        If *hash_ids_list* is provided, the prompts will be built from shared blocks 
        to enable prefix caching. Each hash ID corresponds to a specific token block.
        """
        import numpy as np
        rng = np.random.RandomState(0)
        
        if hash_ids_list is not None:
            # Build prompts using hash IDs to enable prefix caching
            block_store: Dict[int, List[int]] = {}
            prompts = []
            
            for length, hash_ids in zip(input_lens, hash_ids_list):
                prompt_tokens = []
                
                # If there are hash IDs, process them
                if hash_ids:
                    # Calculate block size (assume uniform size except possibly the last block)
                    # We estimate block size from max length and number of hashes
                    block_size = length // len(hash_ids) if len(hash_ids) > 0 else length
                    # Fallback to a default if math is weird
                    if block_size == 0: block_size = 16 
                    
                    for i, h in enumerate(hash_ids):
                        # Generate or retrieve the block
                        if h not in block_store:
                            block_store[h] = rng.randint(0, vocab_size, size=block_size).tolist()
                        
                        prompt_tokens.extend(block_store[h])
                
                # Truncate or pad to exactly match the requested length
                if len(prompt_tokens) > length:
                    prompt_tokens = prompt_tokens[:length]
                elif len(prompt_tokens) < length:
                    padding = rng.randint(0, vocab_size, size=length - len(prompt_tokens)).tolist()
                    prompt_tokens.extend(padding)
                    
                prompts.append({"prompt_token_ids": prompt_tokens})
            return prompts
        else:
            return [
                {"prompt_token_ids": rng.randint(0, vocab_size, size=l).tolist()}
                for l in input_lens
            ]

class OfflineTraceWorkload(BaseTraceWorkload):
    """Offline/bulk inference"""
    def __init__(self, args: argparse.Namespace):
        import numpy as np
        super().__init__(args)
        print(f"Trace: {os.path.basename(args.trace)}, "
              f"{len(self.records)} requests, "
              f"median input={int(np.median(self.input_lens))}, "
              f"median output={int(np.median(self.output_lens))}, "
              f"max_seq_len={self._max_seq_len}")

    def run(self, llm: Any, variant: str, out_path: str) -> None:
        vocab_size = len(llm.get_tokenizer())
        prompts = self.build_prompts_from_trace(
            input_lens=self.input_lens,
            vocab_size=vocab_size,
            hash_ids_list=self.hash_ids_list
        )

        print(f"\n[offline trace] mode={self.args.mode}, variant={variant}, "
              f"requests={len(self.records)}, mask_cache={self.args.mask_cache}")

        metrics = run_benchmark(
            llm, prompts, self.output_lens,
            warmup=self.args.warmup,
            repeats=self.args.repeats,
        )

        total_in = sum(self.input_lens)
        print(f"  ↳ TTFT: {metrics['ttft_s']:<7.4f}s  |  "
              f"Total: {metrics['total_s']:<7.4f}s  |  "
              f"Tput: {metrics['tokens_per_s']:<6.1f} tok/s  |  "
              f"InTok: {total_in:<6}  |  "
              f"OutTok: {metrics['num_output_tokens']:.0f}")

        row = self._build_row(variant, metrics)
        _append_row(out_path, row, self.csv_fields)
        print(f"\nOffline trace complete. Results in: {out_path}")

class OnlineTraceWorkload(BaseTraceWorkload):
    """Online/streaming inference"""
    def __init__(self, args: argparse.Namespace):
        import numpy as np
        super().__init__(args)
        
        # timestamps are usually in ms in Mooncake trace -> convert to seconds
        self.timestamps = [r["timestamp"] / 1000.0 for r in self.records]
        
        # Sort by timestamp
        sorted_indices = np.argsort(self.timestamps)
        self.records = [self.records[i] for i in sorted_indices]
        self.input_lens = [self.input_lens[i] for i in sorted_indices]
        self.output_lens = [self.output_lens[i] for i in sorted_indices]
        self.hash_ids_list = [self.hash_ids_list[i] for i in sorted_indices]
        self.timestamps = [self.timestamps[i] for i in sorted_indices]
        
        # Shift timestamps to start at 0
        t0 = self.timestamps[0]
        self.timestamps = [t - t0 for t in self.timestamps]
        
        print(f"Trace: {os.path.basename(args.trace)}, "
              f"{len(self.records)} requests, "
              f"duration={self.timestamps[-1]:.1f}s, "
              f"median input={int(np.median(self.input_lens))}, "
              f"median output={int(np.median(self.output_lens))}, "
              f"max_seq_len={self._max_seq_len}")

    def run(self, llm: Any, variant: str, out_path: str) -> None:
        import time
        import json
        from vllm import SamplingParams
        
        vocab_size = len(llm.get_tokenizer())
        prompts = self.build_prompts_from_trace(
            input_lens=self.input_lens,
            vocab_size=vocab_size,
            hash_ids_list=self.hash_ids_list
        )
        engine = llm.llm_engine
        
        print(f"\n[online trace] mode={self.args.mode}, variant={variant}, "
              f"requests={len(self.records)}, mask_cache={self.args.mask_cache}")

        warmup = self.args.warmup
        repeats = self.args.repeats
        all_metrics = []

        for r_idx in range(-warmup, repeats):
            is_warmup = r_idx < 0
            label = f"  Warmup {warmup + r_idx + 1}/{warmup}" if is_warmup \
                    else f"  Run {r_idx + 1}/{repeats}"
            print(f"{label} ...")
            req_idx = 0
            n_reqs = len(self.records)

            first_token_times = {}
            end_times = {}
            arrival_times = {}
            output_tokens_dict = {}

            start_time = time.perf_counter()

            while req_idx < n_reqs or engine.has_unfinished_requests():
                current_time = time.perf_counter() - start_time

                # Submit requests that have arrived
                while req_idx < n_reqs and current_time >= self.timestamps[req_idx]:
                    req_id = f"r{r_idx}_req_{req_idx}"
                    prompt_token_ids = prompts[req_idx]["prompt_token_ids"]
                    sp = SamplingParams(max_tokens=self.output_lens[req_idx], temperature=0.0, ignore_eos=True)

                    engine.add_request(
                        request_id=req_id,
                        prompt={"prompt_token_ids": prompt_token_ids},
                        params=sp,
                    )
                    arrival_times[req_id] = current_time
                    req_idx += 1

                if engine.has_unfinished_requests():
                    step_outputs = engine.step()
                    now = time.perf_counter() - start_time
                    for out in step_outputs:
                        req_id = out.request_id
                        if req_id not in first_token_times and out.outputs[0].token_ids:
                            first_token_times[req_id] = now
                        if out.finished:
                            end_times[req_id] = now
                            output_tokens_dict[req_id] = len(out.outputs[0].token_ids)
                else:
                    time_to_next = self.timestamps[req_idx] - (time.perf_counter() - start_time)
                    if time_to_next > 0:
                        time.sleep(min(time_to_next, 0.01))

            total_duration = time.perf_counter() - start_time
            ttfts = []
            itls = []
            decode_times = []

            for i in range(n_reqs):
                req_id = f"r{r_idx}_req_{i}"
                if req_id in end_times and req_id in first_token_times:
                    ttft = first_token_times[req_id] - arrival_times[req_id]
                    decode_time = end_times[req_id] - first_token_times[req_id]
                    num_out = output_tokens_dict[req_id]

                    ttfts.append(ttft)
                    decode_times.append(decode_time)
                    if num_out > 1:
                        itls.append((decode_time / (num_out - 1)) * 1000.0)

            mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
            mean_itl = sum(itls) / len(itls) if itls else 0.0
            mean_decode = sum(decode_times) / len(decode_times) if decode_times else 0.0
            total_out_tokens = sum(output_tokens_dict.values())
            tput = total_out_tokens / total_duration if total_duration > 0 else 0.0

            print(f"    ↳ TTFT: {mean_ttft:<7.4f}s  |  "
                  f"ITL: {mean_itl:<6.2f}ms  |  "
                  f"Total: {total_duration:<7.4f}s  |  "
                  f"Tput: {tput:.1f} tok/s")

            if is_warmup:
                continue

            all_metrics.append({
                "total_s": total_duration,
                "ttft_s": mean_ttft,
                "decode_s": mean_decode,
                "mean_itl_ms": mean_itl,
                "tokens_per_s": tput,
                "num_output_tokens": total_out_tokens / n_reqs,
            })
            
        # Average metrics across repeats
        avg_metrics = {k: sum(m[k] for m in all_metrics) / repeats for k in all_metrics[0]}
        
        row = self._build_row(variant, avg_metrics)
        _append_row(out_path, row, self.csv_fields)
        print(f"\nOnline trace complete. Results in: {out_path}")


class SyntheticWorkload:
    csv_fields = COMMON_INPUT_FIELDS + COMMON_OUTPUT_FIELDS

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.batch_sizes = args.batch_size
        self.input_lens_cfg = args.input_len
        self.output_lens_list = args.output_len
        self._max_seq_len = max(self.input_lens_cfg) + max(self.output_lens_list)

    @property
    def max_seq_len(self) -> int:
        return self._max_seq_len

    def run(self, llm: Any, variant: str, out_path: str) -> None:
        configs = [
            (bs, il, ol)
            for bs, il in zip(self.batch_sizes, self.input_lens_cfg)
            for ol in self.output_lens_list
        ]

        vocab_size = len(llm.get_tokenizer())
        for i, (bs, il, ol) in enumerate(configs, 1):
            print(
                f"\n[{i}/{len(configs)}] mode={self.args.mode}, variant={variant}, "
                f"batch={bs}, input_len={il}, output_len={ol}, "
                f"mask_cache={self.args.mask_cache}"
            )

            prompts = synthesize_prompts(bs, il, vocab_size=vocab_size)
            metrics = run_benchmark(
                llm, prompts, ol,
                warmup=self.args.warmup,
                repeats=self.args.repeats,
            )

            print(f"  ↳ TTFT: {metrics['ttft_s']:<7.4f}s  |  "
                  f"Decode: {metrics['decode_s']:<7.4f}s  |  "
                  f"Total: {metrics['total_s']:<7.4f}s  |  "
                  f"ITL: {metrics['mean_itl_ms']:<6.2f}ms  |  "
                  f"Tput: {metrics['tokens_per_s']:.1f} tok/s")

            row = {
                "mode": self.args.mode,
                "variant": variant,
                "model": self.args.model,
                "batch_size": bs,
                "input_len": il,
                "output_len": ol,
                "mask_cache": self.args.mask_cache,
                "prefix_caching": self.args.enable_prefix_caching,
                **metrics,
            }
            _append_row(out_path, row, self.csv_fields)

        print(f"\n{len(configs)} configs complete. Results in: {out_path}")


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    # ── Debug mode (short-circuit) ──────────────────────────────────────
    if args.mode == "debug":
        run_debug(args)
        return

    # ── Determine max_seq_len (needed for patches and LLM init) ─────────
    use_trace = args.trace is not None
    if use_trace:
        if args.online:
            workload = OnlineTraceWorkload(args)
        else:
            workload = OfflineTraceWorkload(args)
    else:
        workload = SyntheticWorkload(args)
        
    max_seq_len = workload.max_seq_len

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
        enable_prefix_caching=args.enable_prefix_caching,
    )

    out_path = _csv_setup(args, workload.csv_fields)
    variant = args.variant if args.mode != "baseline" else "n/a"

    workload.run(llm, variant, out_path)


if __name__ == "__main__":
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
        default="Qwen/Qwen2.5-3B",  # meta-llama/Llama-3.2-1B is gated
        help="HuggingFace model ID.",
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
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable prefix caching in vLLM.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Base name for CSV in benchmarks/results/ (defaults to vllm_e2e_{synth|offline|online}).",
    )

    # Trace-driven workload
    trace_group = parser.add_argument_group("Trace-driven workload configuration")
    trace_group.add_argument(
        "--trace", type=str, default=None,
        help="Path to Mooncake JSONL trace file (replaces synthetic configs).",
    )
    trace_group.add_argument(
        "--max-requests", type=int, default=None,
        help="Max requests to replay from the trace (default: all).",
    )
    trace_group.add_argument(
        "--max-input-len", type=int, default=None,
        help="Filter out trace requests exceeding this input length.",
    )
    trace_group.add_argument(
        "--online", action="store_true",
        help="Run trace workload in online mode (respecting trace timestamps).",
    )
    trace_group.add_argument(
        "--offline", action="store_false", dest="online",
        help="Run trace workload in offline mode (bulk inference).",
    )
    parser.set_defaults(online=True)
    
    # Synthetic workload
    synth_group = parser.add_argument_group("Synthetic workload configuration")
    # Prefill-heavy scenarios to highlight attention kernel & mask overhead.
    synth_group.add_argument(
        "--batch-size", type=int, nargs="+",
        default=[32, 16, 8, 4, 2, 1],
        help="Number of concurrent input sequences per batch.",
    )
    synth_group.add_argument(
        "--input-len", type=int, nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384],
        help="Number of tokens per input sequence.",
    )
    synth_group.add_argument(
        "--output-len", type=int, nargs="+",
        default=[256],
        help="Guaranteed number of output tokens to generate per sequence (ignores EOS).",
    )
    args = parser.parse_args()

    main(args)
