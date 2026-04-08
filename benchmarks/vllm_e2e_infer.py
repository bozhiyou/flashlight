"""End-to-end vLLM inference benchmark for the e2e variant set.

Routes vLLM prefill through a FlexAttention or Flashlight backend via a
single monkey-patch on ``vllm.vllm_flash_attn.flash_attn_varlen_func``.
Decode continues to use the native flash-attn kvcache kernel, so TTFT
captures the variant's prefill performance while ITL reflects standard
paged-attention decode.

The four e2e variants (2×2 matrix of mask shape × score_mod, all packed
onto vLLM's varlen document-masked substrate):

    - causal_packed          (control: triangular, identity)
    - sliding_window_packed  (banded, identity)
    - causal_alibi_packed    (triangular, linear score_mod)
    - causal_softcap_packed  (triangular, non-linear score_mod)

Layout contract
---------------
Only Path A (full prefill, packed 3D varlen, Q/K/V shape ``(total, H, D)``,
``cu_seqlens_q == cu_seqlens_k``) is supported. Chunked prefill and prefix
caching are disabled on the vLLM side; the hook raises on Path B (4D paged
K/V or non-empty ``block_table``) rather than silently falling back.

The hook pads ``total`` up to the next bucket in ``--bucket-sizes`` (default
``1024,2048,4096,8192,16384,32768``) and extends ``cu_seqlens_q`` with a
sentinel doc over the padding tail. This bounds the distinct compile
shapes to ``|buckets|`` so ``torch.compile(dynamic=False)`` doesn't blow
the dynamo cache.

Usage
-----

Baseline (unpatched vLLM, trace mode)::

    python benchmarks/vllm_e2e_infer.py --mode baseline \\
        --trace benchmarks/traces/conversation_trace.jsonl --online \\
        --max-requests 200 --max-input-len 16384

FlexAttention, packed-causal variant::

    python benchmarks/vllm_e2e_infer.py --mode flex --variant causal_packed \\
        --trace benchmarks/traces/conversation_trace.jsonl --online \\
        --max-requests 200 --max-input-len 16384

Flashlight (packed variants via monkeypatch/fusion)::

    python benchmarks/vllm_e2e_infer.py --mode flashlight --variant causal_packed ...

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
# E2e variant set
# ---------------------------------------------------------------------------
# The four rows of the 2×2 (mask shape × score_mod) matrix. Every variant sits
# on top of the packed-doc substrate (attn_gym's `generate_doc_mask_mod`), hence
# the `_packed` suffix — a reader grepping this CSV cannot mistake an e2e
# number for a kernel-bench number.
E2E_VARIANTS: List[str] = [
    "causal_packed",           # control: triangular mask, identity score_mod
    "sliding_window_packed",   # perturbs mask shape   (banded)
    "causal_alibi_packed",     # perturbs score_mod    (linear)
    "causal_softcap_packed",   # perturbs score_mod    (non-linear)
]

# Default bucket sizes. `dynamic=False` would otherwise recompile on every
# distinct `padded_total`; bucketing caps the number of distinct shapes to at
# most |buckets| regardless of trace length distribution.
DEFAULT_BUCKET_SIZES: List[int] = [1024, 2048, 4096, 8192, 16384, 32768]

# Sliding-window configuration (Mistral-7B-v0.1 window size).
SLIDING_WINDOW_SIZE: int = 256

# Softcap value (Gemma-2).
SOFTCAP_VALUE: int = 30

# Minimum seqlen to intercept. Anything below this is a vLLM-internal call
# (e.g. memory-profiling with seqlen=9) that the compiled kernel cannot handle
# (Triton requires power-of-2 arange). Fall back to the original flash-attn
# kernel for these calls — they don't affect benchmark results.
_MIN_BENCH_SEQLEN: int = 256


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
# E2e attention path: layout normalizer + pluggable backend
# ---------------------------------------------------------------------------
#
# The hook is factored in two pieces so FlexAttention and Flashlight receive
# byte-identical inputs:
#
#   1. `_pad_and_pack` — layout normalizer. Takes vLLM's packed varlen 3D
#      `(total, H, D)` tensors, pads the token dimension to the next bucket,
#      reshapes to `(1, H, padded_total, D)`, and extends `cu_seqlens_q` with
#      a sentinel doc covering the padding tail.
#
#   2. Backend runner — a callable from the backend registry that takes
#      `(q, k, v, offsets, padded_total)` and returns the attention output.
#      Two backends:
#
#        - flex:       flex_attention(q, k, v, block_mask=doc_mask(...),
#                                     score_mod=...) compiled with
#                                     `dynamic=False`.
#        - flashlight: @torch.compile(dynamic=False) plain-PyTorch packed
#                      attention in attention_variants/packed/*.py with
#                      monkeypatch/fusion loaded.
#
# Only Path A (full prefill, packed 3D varlen) is supported; Path B (chunked
# prefill / prefix-cache hit, 4D paged KV) raises.


def _next_bucket(n: int, buckets: List[int]) -> int:
    """Round ``n`` up to the smallest bucket size >= ``n``."""
    for b in buckets:
        if b >= n:
            return b
    raise RuntimeError(
        f"vllm_e2e_infer: total_tokens={n} exceeds largest bucket "
        f"{buckets[-1]}. Increase --bucket-sizes or lower --max-input-len."
    )


def _pad_and_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    bucket_sizes: List[int],
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]":
    """Normalize vLLM's packed varlen layout for the e2e backends.

    Input:  Q/K/V as packed ``(total, H, D)`` with cumulative
            per-doc offsets ``cu_seqlens_q = [0, s1, s1+s2, ..., total]``.

    Output: Q/K/V as ``(1, H, padded_total, D)`` where
            ``padded_total = next_bucket(total)``, plus an ``offsets`` tensor
            extended with a sentinel doc covering ``[total, padded_total)``.
            The sentinel's doc_id never matches a real doc, so padding tokens
            are zeroed out by the ``same_doc`` check in
            ``generate_doc_mask_mod``.
    """
    total = q.shape[0]
    padded_total = _next_bucket(total, bucket_sizes)
    pad = padded_total - total

    if pad > 0:
        # F.pad for a 3D tensor takes (last_dim_left, last_dim_right,
        # mid_dim_left, mid_dim_right, first_dim_left, first_dim_right).
        # We pad the token dimension (dim 0) on the right.
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))

    # (padded_total, H, D) -> (1, H, padded_total, D)  (zero-copy view)
    q_bhsd = q.transpose(0, 1).unsqueeze(0).contiguous()
    k_bhsd = k.transpose(0, 1).unsqueeze(0).contiguous()
    v_bhsd = v.transpose(0, 1).unsqueeze(0).contiguous()

    # Append one sentinel doc whose range is [total, padded_total).
    # `cu_seqlens_q` already ends in `total`; we append `padded_total`.
    # When pad==0 the sentinel is an empty doc (count 0) and is harmless.
    sentinel = torch.tensor(
        [padded_total], device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype,
    )
    offsets = torch.cat([cu_seqlens_q, sentinel])

    return q_bhsd, k_bhsd, v_bhsd, offsets, total, padded_total


def _build_e2e_flex_backend() -> Dict[str, Callable[..., Any]]:
    """Build FlexAttention runners for the four packed e2e variants.

    Each runner has the signature::

        runner(q, k, v, offsets, padded_total) -> (output, block_mask_build_s)

    where ``q/k/v`` are ``(1, H, padded_total, D)``, ``offsets`` includes the
    sentinel doc, and ``output`` is ``(1, H_q, padded_total, D)``.
    """
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask,
    )
    from attn_gym.masks import generate_sliding_window, generate_doc_mask_mod
    from attn_gym.masks.document_mask import _offsets_to_doc_ids_tensor
    from attn_gym.mods import generate_tanh_softcap

    # Fresh compiled handle — do NOT reuse run_flex_variants' module-global
    # compiled flex_attention, which is tuned for kernel-bench shapes.
    _compiled_flex = torch.compile(flex_attention, dynamic=False)

    # Inner mask mods operate on *per-doc local* indices (generate_doc_mask_mod
    # subtracts the doc offset before calling the inner rule).
    def _causal_inner(_b: Any, _h: Any, q_idx: Any, kv_idx: Any) -> Any:
        return q_idx >= kv_idx

    _sliding_inner = generate_sliding_window(SLIDING_WINDOW_SIZE)

    # ALiBi's bias is relative distance within a document, so the score_mod
    # must convert global q/kv indices to per-doc local indices via the
    # captured (doc_id, offsets) tensors. Softcap is position-independent,
    # so it doesn't need per-call rebinding.
    _softcap = generate_tanh_softcap(SOFTCAP_VALUE, approx=False)

    def _make_packed_alibi_score_mod(doc_id: torch.Tensor, offsets: torch.Tensor,
                                     num_heads: int) -> Callable[..., Any]:
        def alibi(score: Any, _b: Any, h: Any, q_idx: Any, kv_idx: Any) -> Any:
            q_local = q_idx - offsets[doc_id[q_idx]]
            kv_local = kv_idx - offsets[doc_id[kv_idx]]
            scale = torch.exp2(-((h + 1) * 8.0 / num_heads))
            return score + (kv_local - q_local) * scale
        return alibi

    def _make_runner(
        inner_mask_mod: Callable[..., Any],
        score_mod_builder: Optional[Callable[..., Callable[..., Any]]],
    ) -> Callable[..., Any]:
        def runner(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   offsets: torch.Tensor, padded_total: int) -> Any:
            doc_mask_mod = generate_doc_mask_mod(inner_mask_mod, offsets)

            t0 = time.perf_counter()
            block_mask = create_block_mask(
                doc_mask_mod, B=1, H=1,
                Q_LEN=padded_total, KV_LEN=padded_total,
                device=str(q.device),
            )
            build_s = time.perf_counter() - t0

            score_mod: Optional[Callable[..., Any]] = None
            if score_mod_builder is not None:
                doc_id = _offsets_to_doc_ids_tensor(offsets)
                score_mod = score_mod_builder(doc_id, offsets, q.shape[1])

            enable_gqa = q.shape[1] != k.shape[1]
            out = _compiled_flex(
                q, k, v,
                block_mask=block_mask,
                score_mod=score_mod,
                enable_gqa=enable_gqa,
            )
            return out, build_s
        return runner

    return {
        "causal_packed":         _make_runner(_causal_inner, None),
        "sliding_window_packed": _make_runner(_sliding_inner, None),
        "causal_alibi_packed":   _make_runner(
            _causal_inner, _make_packed_alibi_score_mod,
        ),
        "causal_softcap_packed": _make_runner(
            _causal_inner, lambda _doc_id, _offsets, _H: _softcap,
        ),
    }


def _build_e2e_flashlight_backend() -> Dict[str, Callable[..., Any]]:
    """Build Flashlight runners for the four packed e2e variants.

    Each runner has the same signature as the flex backend::

        runner(q, k, v, offsets, padded_total) -> (output, block_mask_build_s)

    ``block_mask_build_s`` is always 0.0 — there is no BlockMask in the
    Flashlight path (which is itself a finding for the paper).

    Activation sequence:
    1. Import ``monkeypatch/fusion`` patches (modifies TorchInductor).
    2. Disable SFDP pattern matching so Dynamo doesn't replace the attention
       subgraph with FlashAttention.
    3. Enable ``max_autotune`` for best Triton kernel selection.
    4. Import the four ``attention_variants/packed/*.py`` modules and wrap
       each with ``torch.compile(dynamic=False)``.
    """
    # --- Flashlight activation ---
    from monkeypatch.fusion import dependent_reduction_fusion  # noqa: F401
    from monkeypatch.fusion import block_reduction  # noqa: F401
    from monkeypatch.fusion import reduction_kernel_fusion  # noqa: F401
    from monkeypatch import disable_flashattention_replacement
    disable_flashattention_replacement()

    import torch._inductor.config
    torch._inductor.config.max_autotune = True

    # --- Import packed variants ---
    # Each packed variant is split into a mask builder (eager, data-dependent
    # indexing) and a compiled attention function (fusible matmul→mask→softmax→matmul).
    from attention_variants.packed import (
        attention_packed_causal, build_packed_causal_mask,
        attention_packed_sliding_window, build_packed_sliding_window_mask,
        attention_packed_causal_alibi, build_packed_causal_alibi_mask,
        attention_packed_causal_softcap, build_packed_causal_softcap_mask,
    )

    from attn_gym.masks.document_mask import _offsets_to_doc_ids_tensor

    # Compile the attention functions (NOT the mask builders — those stay eager
    # because indirect indexing into doc_id/offsets trips block_reduction).
    _compiled_causal = torch.compile(attention_packed_causal, dynamic=False)
    _compiled_sliding = torch.compile(attention_packed_sliding_window, dynamic=False)
    _compiled_alibi = torch.compile(attention_packed_causal_alibi, dynamic=False)
    _compiled_softcap = torch.compile(attention_packed_causal_softcap, dynamic=False)

    def _runner_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       offsets: torch.Tensor, padded_total: int) -> Any:
        doc_id = _offsets_to_doc_ids_tensor(offsets)
        mask = build_packed_causal_mask(doc_id, offsets, padded_total, q.device)
        out = _compiled_causal(q, k, v, mask)
        return out, 0.0

    def _runner_sliding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        offsets: torch.Tensor, padded_total: int) -> Any:
        doc_id = _offsets_to_doc_ids_tensor(offsets)
        mask = build_packed_sliding_window_mask(
            doc_id, offsets, padded_total, q.device,
            window_size=SLIDING_WINDOW_SIZE,
        )
        out = _compiled_sliding(q, k, v, mask)
        return out, 0.0

    def _runner_alibi(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      offsets: torch.Tensor, padded_total: int) -> Any:
        doc_id = _offsets_to_doc_ids_tensor(offsets)
        mask, local_pos = build_packed_causal_alibi_mask(
            doc_id, offsets, padded_total, q.device,
        )
        out = _compiled_alibi(q, k, v, mask, local_pos)
        return out, 0.0

    def _runner_softcap(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        offsets: torch.Tensor, padded_total: int) -> Any:
        doc_id = _offsets_to_doc_ids_tensor(offsets)
        mask = build_packed_causal_softcap_mask(doc_id, offsets, padded_total, q.device)
        out = _compiled_softcap(q, k, v, mask, softcap=float(SOFTCAP_VALUE))
        return out, 0.0

    return {
        "causal_packed":         _runner_causal,
        "sliding_window_packed": _runner_sliding,
        "causal_alibi_packed":   _runner_alibi,
        "causal_softcap_packed": _runner_softcap,
    }


# ---------------------------------------------------------------------------
# Attention monkey-patching (hooks flash_attn, NOT SDPA)
# ---------------------------------------------------------------------------

def _install_e2e_hook(
    backend: Dict[str, Callable[..., Any]],
    variant: str,
    bucket_sizes: List[int],
    stats: Dict[str, Any],
) -> None:
    """Replace ``flash_attn_varlen_func`` with an e2e-backend hook.

    Must be called **before** ``from vllm import LLM`` so vLLM's own
    ``from vllm.vllm_flash_attn import flash_attn_varlen_func`` resolves
    to the hook. vLLM 0.6.x bundles its own flash-attn fork at
    ``vllm.vllm_flash_attn``; we patch both the interface sub-module and
    the package ``__init__`` so later re-imports bind to the hook.

    Only Path A (full-prefill packed varlen, Q/K/V 3D, cu_seqlens_q == cu_seqlens_k)
    is supported. Path B (chunked prefill / prefix-cache hit, 4D paged K/V,
    non-empty ``block_table``) raises — this is a config error, not a
    silent fallback. Decode uses ``flash_attn_with_kvcache`` (different
    function) and never reaches this hook.

    Profiling probes with ``max_seqlen_q < _MIN_BENCH_SEQLEN`` delegate to
    the original kernel and are counted as ``fallback_calls``.
    """
    try:
        import vllm.vllm_flash_attn as vfa
        import vllm.vllm_flash_attn.flash_attn_interface as fai
    except ImportError:
        raise RuntimeError(
            "vllm with bundled vllm_flash_attn is required. "
            "Install with:  pip install 'vllm>=0.6.4,<0.7.0'"
        )

    _orig = fai.flash_attn_varlen_func
    if variant not in backend:
        raise KeyError(
            f"Variant {variant!r} not in backend registry {sorted(backend)}"
        )
    runner = backend[variant]

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
        block_table = kwargs.get("block_table")

        # Profiling / tiny-seqlen calls — delegate to native kernel.
        if batch_size == 0 or max_seqlen_q < _MIN_BENCH_SEQLEN:
            stats["fallback_calls"] += 1
            return _orig(
                q, k, v, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p, softmax_scale=softmax_scale,
                causal=causal, **kwargs,
            )

        # Path B (chunked prefill / prefix-cache) — config error, fail loud.
        if block_table is not None and block_table.numel() > 0:
            raise RuntimeError(
                "e2e hook received a Path-B call (non-empty block_table). "
                "Set enable_chunked_prefill=False and enable_prefix_caching=False "
                "on LLM(...), and cap --max-input-len below vLLM's chunked-prefill "
                "auto-enable threshold."
            )
        if k.ndim == 4:
            raise RuntimeError(
                "e2e hook received 4D K (paged cache) — Path B not supported."
            )
        if cu_seqlens_q.shape != cu_seqlens_k.shape or not torch.equal(
            cu_seqlens_q, cu_seqlens_k,
        ):
            raise RuntimeError(
                "e2e hook expects full-prefill Path A "
                "(cu_seqlens_q == cu_seqlens_k)."
            )

        nheads_q, headdim = q.shape[1], q.shape[2]

        t_call = time.perf_counter()
        q_bhsd, k_bhsd, v_bhsd, offsets, _total, padded_total = _pad_and_pack(
            q, k, v, cu_seqlens_q, bucket_sizes,
        )
        out, build_s = runner(q_bhsd, k_bhsd, v_bhsd, offsets, padded_total)
        # Force compile / kernel completion before stopping the clock so the
        # first-call compile cost is attributed to compile_time_s.
        torch.cuda.synchronize()
        call_elapsed = time.perf_counter() - t_call

        # flex_attention with enable_gqa=True returns 5D
        # (B, H_kv, GQA_ratio, S, D) → merge to (B, H_q, S, D).
        if out.ndim == 5:
            out = out.reshape(1, nheads_q, padded_total, headdim)

        # (1, H_q, padded_total, D) → (padded_total, H_q, D) → slice padding
        out_packed = out.squeeze(0).transpose(0, 1).contiguous()
        out_packed = out_packed[:total_q].contiguous()

        # First in-scope call absorbs compile cost.
        if stats["prefill_calls"] == 0:
            stats["compile_time_s"] = call_elapsed
        stats["prefill_calls"] += 1
        stats["block_mask_build_s"] += build_s

        out_buf = kwargs.get("out")
        if out_buf is not None:
            out_buf.copy_(out_packed)
            return out_buf
        return out_packed

    fai.flash_attn_varlen_func = _hooked  # type: ignore[assignment]
    vfa.flash_attn_varlen_func = _hooked  # type: ignore[assignment]


def _make_stats() -> Dict[str, Any]:
    return {
        "prefill_calls": 0,
        "fallback_calls": 0,
        "block_mask_build_s": 0.0,
        "compile_time_s": 0.0,
    }


def apply_flex_patch(variant: str, bucket_sizes: List[int]) -> Dict[str, Any]:
    """Install the e2e FlexAttention hook for ``variant``.

    Returns the mutable ``stats`` dict the hook writes into; the caller
    reads it after the run to populate the summary CSV.
    """
    if variant not in E2E_VARIANTS:
        raise ValueError(
            f"variant={variant!r} is not in the e2e variant set {E2E_VARIANTS}"
        )
    backend = _build_e2e_flex_backend()
    stats = _make_stats()
    _install_e2e_hook(backend, variant, bucket_sizes, stats)
    print(
        f"[patch] flash_attn_varlen_func → e2e FlexAttention variant={variant}, "
        f"buckets={bucket_sizes}"
    )
    return stats


def apply_flashlight_patch(variant: str, bucket_sizes: List[int]) -> Dict[str, Any]:
    """Install the e2e Flashlight hook for ``variant``.

    Loads ``monkeypatch/fusion`` patches, disables SFDP replacement, compiles
    the packed variant from ``attention_variants/packed/``, and installs the
    hook through the same ``_install_e2e_hook`` path as flex.
    """
    if variant not in E2E_VARIANTS:
        raise ValueError(
            f"variant={variant!r} is not in the e2e variant set {E2E_VARIANTS}"
        )
    backend = _build_e2e_flashlight_backend()
    stats = _make_stats()
    _install_e2e_hook(backend, variant, bucket_sizes, stats)
    print(
        f"[patch] flash_attn_varlen_func → e2e Flashlight variant={variant}, "
        f"buckets={bucket_sizes}"
    )
    return stats


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
]

COMMON_OUTPUT_FIELDS = [
    "total_s", "ttft_s", "decode_s", "mean_itl_ms",
    "tokens_per_s", "num_output_tokens",
]

# ---------------------------------------------------------------------------
# E2e two-CSV schema (summary + per_request, joined on run_id)
# ---------------------------------------------------------------------------

E2E_SUMMARY_FIELDS: List[str] = [
    "run_id", "mode", "variant", "model",
    "trace", "num_requests", "max_input_len",
    "mean_ttft_s", "p50_ttft_s", "p95_ttft_s", "p99_ttft_s",
    "mean_itl_ms", "p50_itl_ms", "p95_itl_ms", "p99_itl_ms",
    "total_s", "tput_tok_s",
    "prefill_frac",
    "block_mask_build_s", "compile_time_s",
    "prefill_calls", "fallback_calls",
]

E2E_PER_REQUEST_FIELDS: List[str] = [
    "run_id", "mode", "variant",
    "request_idx", "input_length", "output_length",
    "arrival_t_s", "ttft_s", "decode_s", "num_output_tokens",
    "batch_size_at_prefill",
]


def _percentile(xs: List[float], p: float) -> float:
    """Plain nearest-rank percentile; avoids a numpy dep in the hot path."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _e2e_csv_paths(args: argparse.Namespace) -> "tuple[str, str]":
    """Return (summary_csv_path, per_request_csv_path)."""
    out_dir = os.path.join(_BENCH_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    base = getattr(args, "output", None)
    if base is None:
        base = "vllm_e2e_online" if args.online else "vllm_e2e_offline"
    return (
        os.path.join(out_dir, f"{base}_summary.csv"),
        os.path.join(out_dir, f"{base}_per_request.csv"),
    )


def _append_csv_row(path: str, row: Dict[str, Any], fields: List[str]) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_csv_rows(path: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


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

    def print_length_histogram(self, variant: str) -> None:
        """Print p50/p95/p99 of trace input lengths and warn on null-result risk.

        Sliding-window has trivially-equivalent output to causal when the
        trace's contexts are shorter than ~2·window, because the window
        covers the full causal triangle per doc. Surface this before any
        timed iteration.
        """
        p50 = _percentile(list(map(float, self.input_lens)), 50)
        p95 = _percentile(list(map(float, self.input_lens)), 95)
        p99 = _percentile(list(map(float, self.input_lens)), 99)
        print(
            f"[trace] input_length distribution: "
            f"p50={int(p50)}, p95={int(p95)}, p99={int(p99)}, "
            f"min={min(self.input_lens)}, max={max(self.input_lens)}"
        )
        if variant == "sliding_window_packed" and p50 < 2 * SLIDING_WINDOW_SIZE:
            print(
                f"[trace] WARNING: sliding_window_packed (W={SLIDING_WINDOW_SIZE}) "
                f"is numerically equivalent to causal_packed when p50 < {2 * SLIDING_WINDOW_SIZE}. "
                f"Expect a null result on this trace."
            )

    def _build_row(self, variant: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        import numpy as np
        return {
            "mode": self.args.mode,
            "variant": variant,
            "model": self.args.model,
            "batch_size": len(self.records),
            "input_len": int(np.median(self.input_lens)),
            "output_len": int(np.median(self.output_lens)),
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
              f"requests={len(self.records)}")

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
        from vllm import SamplingParams

        vocab_size = len(llm.get_tokenizer())
        prompts = self.build_prompts_from_trace(
            input_lens=self.input_lens,
            vocab_size=vocab_size,
            hash_ids_list=self.hash_ids_list
        )
        engine = llm.llm_engine

        print(f"\n[online trace] mode={self.args.mode}, variant={variant}, "
              f"num_requests={len(self.records)}")

        warmup = self.args.warmup
        repeats = self.args.repeats
        all_metrics = []
        # Per-request records captured on the final timed iteration, used by
        # the e2e two-CSV writer.
        final_per_request: List[Dict[str, Any]] = []
        final_ttfts: List[float] = []
        final_itls: List[float] = []
        final_total_s: float = 0.0
        final_tput: float = 0.0
        final_prefill_s: float = 0.0

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

            # --- Simulated-time trace replay ---
            # We use "simulated time" (sim_time) to pace request submissions
            # instead of wall-clock time. This decouples submission pacing from
            # engine speed:
            #  - Wall-clock is only used for latency/throughput measurement.
            #  - sim_time advances by the *trace gap* between consecutive
            #    requests each time we submit one, so the relative arrival
            #    pattern is preserved regardless of how fast/slow engine.step()
            #    runs.
            #  - Between submissions we drain all pending engine work, which
            #    mirrors how a real async server would schedule: accept a
            #    request, then process until the next arrival.
            sim_time = 0.0  # tracks position in the trace timeline

            while req_idx < n_reqs or engine.has_unfinished_requests():

                # Submit the next request whose trace timestamp has been
                # reached in simulated time.  We submit at most one request
                # per outer-loop iteration so the engine can make progress
                # between arrivals and we never burst-load the scheduler.
                if req_idx < n_reqs and sim_time >= self.timestamps[req_idx]:
                    req_id = f"r{r_idx}_req_{req_idx}"
                    prompt_token_ids = prompts[req_idx]["prompt_token_ids"]
                    sp = SamplingParams(max_tokens=self.output_lens[req_idx], temperature=0.0, ignore_eos=True)

                    engine.add_request(
                        request_id=req_id,
                        prompt={"prompt_token_ids": prompt_token_ids},
                        params=sp,
                        arrival_time=self.timestamps[req_idx],
                    )
                    arrival_times[req_id] = time.perf_counter() - start_time
                    req_idx += 1

                    # Advance sim_time to the next request's timestamp so
                    # we'll submit it on the next iteration (preserving trace
                    # inter-arrival gaps).  If multiple requests share the
                    # same timestamp they'll be submitted in consecutive
                    # iterations with no engine steps in between — fast enough
                    # to be effectively simultaneous.
                    if req_idx < n_reqs:
                        sim_time = self.timestamps[req_idx]

                # Run one engine step to make progress on in-flight requests.
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
                            done_idx = int(req_id.rsplit("_", 1)[1])
                            print(f"\r  [{self.args.mode} {variant}] t={now:.1f}s  done={len(end_times)}/{n_reqs}  "
                                  f"last_req={{id={done_idx},in={self.input_lens[done_idx]},out={output_tokens_dict[req_id]}}}  ",
                                  end="", flush=True)
                elif req_idx < n_reqs:
                    # No in-flight work and next request hasn't "arrived" yet.
                    # Jump sim_time forward — no point waiting in a sim.
                    sim_time = self.timestamps[req_idx]

            print("done")

            total_duration = time.perf_counter() - start_time
            ttfts = []
            itls = []
            decode_times = []
            # Capture per-request rows for this iteration. If this is the
            # final timed iteration, they end up persisted by the e2e
            # two-CSV writer at the end of `run`.
            iter_per_request: List[Dict[str, Any]] = []

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

                    iter_per_request.append({
                        "run_id": getattr(self.args, "run_id", ""),
                        "mode": self.args.mode,
                        "variant": variant,
                        "request_idx": i,
                        "input_length": self.input_lens[i],
                        "output_length": self.output_lens[i],
                        "arrival_t_s": arrival_times[req_id],
                        "ttft_s": ttft,
                        "decode_s": decode_time,
                        "num_output_tokens": num_out,
                        # Phase 2 will populate this from the scheduler's
                        # batch at prefill time; Phase 1 leaves it blank.
                        "batch_size_at_prefill": "",
                    })

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
            # Overwrite final_* each timed iteration so what survives is the
            # last run's data.
            final_per_request = iter_per_request
            final_ttfts = ttfts
            final_itls = itls
            final_total_s = total_duration
            final_tput = tput
            final_prefill_s = sum(ttfts)

        # Average aggregate metrics across timed repeats (used by the
        # legacy single-CSV writer below).
        avg_metrics = {k: sum(m[k] for m in all_metrics) / repeats for k in all_metrics[0]}

        row = self._build_row(variant, avg_metrics)
        _append_row(out_path, row, self.csv_fields)
        print(f"\nOnline trace complete. Results in: {out_path}")

        # E2e two-CSV writer — uses the final timed iteration's per-request
        # data plus the hook's `e2e_stats` (block_mask_build_s, compile_time_s,
        # prefill_calls, fallback_calls). In baseline mode there is no hook
        # and `args.e2e_stats` is None, so we still write the summary row
        # with zero overhead columns so baseline / flex / flashlight rows
        # can be compared on the same axes.
        self._write_e2e_csvs(
            variant=variant,
            per_request=final_per_request,
            ttfts=final_ttfts,
            itls=final_itls,
            total_s=final_total_s,
            tput_tok_s=final_tput,
            prefill_s=final_prefill_s,
        )

    def _write_e2e_csvs(
        self,
        variant: str,
        per_request: List[Dict[str, Any]],
        ttfts: List[float],
        itls: List[float],
        total_s: float,
        tput_tok_s: float,
        prefill_s: float,
    ) -> None:
        """Write `<output>_summary.csv` and `<output>_per_request.csv`.

        Rows from the final timed iteration are appended; the header is
        written only on first creation.
        """
        stats = getattr(self.args, "e2e_stats", None) or {}
        summary_path, per_req_path = _e2e_csv_paths(self.args)

        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        mean_itl = sum(itls) / len(itls) if itls else 0.0

        summary_row = {
            "run_id": getattr(self.args, "run_id", ""),
            "mode": self.args.mode,
            "variant": variant,
            "model": self.args.model,
            "trace": os.path.basename(self.args.trace),
            "num_requests": len(self.records),
            "max_input_len": self.args.max_input_len,
            "mean_ttft_s": mean_ttft,
            "p50_ttft_s": _percentile(ttfts, 50),
            "p95_ttft_s": _percentile(ttfts, 95),
            "p99_ttft_s": _percentile(ttfts, 99),
            "mean_itl_ms": mean_itl,
            "p50_itl_ms": _percentile(itls, 50),
            "p95_itl_ms": _percentile(itls, 95),
            "p99_itl_ms": _percentile(itls, 99),
            "total_s": total_s,
            "tput_tok_s": tput_tok_s,
            "prefill_frac": (prefill_s / total_s) if total_s > 0 else 0.0,
            "block_mask_build_s": stats.get("block_mask_build_s", 0.0),
            "compile_time_s": stats.get("compile_time_s", 0.0),
            "prefill_calls": stats.get("prefill_calls", 0),
            "fallback_calls": stats.get("fallback_calls", 0),
        }
        _append_csv_row(summary_path, summary_row, E2E_SUMMARY_FIELDS)
        _append_csv_rows(per_req_path, per_request, E2E_PER_REQUEST_FIELDS)
        print(f"[e2e csv] summary     → {summary_path}")
        print(f"[e2e csv] per_request → {per_req_path}  ({len(per_request)} rows)")


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
                f"batch={bs}, input_len={il}, output_len={ol}"
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
                **metrics,
            }
            _append_row(out_path, row, self.csv_fields)

        print(f"\n{len(configs)} configs complete. Results in: {out_path}")


def main(args: argparse.Namespace) -> None:
    import torch.cuda
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    # ── Debug mode (short-circuit) ──────────────────────────────────────
    if args.mode == "debug":
        run_debug(args)
        return

    # ── Resolve bucket sizes and validate --max-input-len ───────────────
    bucket_sizes = _parse_bucket_sizes(args.bucket_sizes)
    args.bucket_sizes_list = bucket_sizes
    if args.max_input_len is not None and args.max_input_len > bucket_sizes[-1]:
        raise ValueError(
            f"--max-input-len={args.max_input_len} exceeds the largest "
            f"bucket {bucket_sizes[-1]}. Raise --bucket-sizes or lower "
            f"--max-input-len."
        )

    # ── Resolve run_id ──────────────────────────────────────────────────
    if getattr(args, "run_id", None) is None:
        import uuid
        args.run_id = uuid.uuid4().hex[:12]
    print(f"[run] run_id={args.run_id}")

    # ── Determine max_seq_len (needed for patches and LLM init) ─────────
    use_trace = args.trace is not None
    if use_trace:
        if args.online:
            workload = OnlineTraceWorkload(args)
        else:
            workload = OfflineTraceWorkload(args)
        # Print trace length histogram + sliding-window null-result warning
        # before we spend time on LLM init / compile.
        variant_for_warn = args.variant if args.mode != "baseline" else "baseline"
        workload.print_length_histogram(variant_for_warn)
    else:
        workload = SyntheticWorkload(args)

    max_seq_len = workload.max_seq_len

    # ── Validate model GQA ratio for FlexAttention ──────────────────────
    if args.mode in ("flex", "flashlight"):
        _check_gqa_compat(args.model)

    # ── Apply attention patches (before vLLM import) ─────────────────────
    stats: Optional[Dict[str, Any]] = None
    if args.mode == "flex":
        stats = apply_flex_patch(args.variant, bucket_sizes)
    elif args.mode == "flashlight":
        stats = apply_flashlight_patch(args.variant, bucket_sizes)
    args.e2e_stats = stats

    # ── Load model via vLLM (once for all configs) ───────────────────────
    from vllm import LLM

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=args.enforce_eager,
        max_model_len=max_seq_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        # Path A only: no chunked prefill, no prefix-cache hits. The e2e
        # hook raises on Path B so config mistakes surface immediately.
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )

    # Cap dynamo compilation cache. Must be set AFTER vLLM import because
    # vllm/worker/model_runner.py hardcodes the limits at module import
    # time. With dynamic=False + bucket padding, the shape count is
    # bounded by |bucket_sizes|, so a modest limit (1024) is plenty; we
    # don't need the old 10000 value.
    import torch._dynamo.config
    torch._dynamo.config.cache_size_limit = 1024
    torch._dynamo.config.accumulated_cache_size_limit = 1024

    out_path = _csv_setup(args, workload.csv_fields)
    variant = args.variant if args.mode != "baseline" else "n/a"

    workload.run(llm, variant, out_path)


def _parse_bucket_sizes(s: "str | List[int]") -> List[int]:
    """Parse ``--bucket-sizes`` into a sorted list of ints."""
    if isinstance(s, list):
        parsed = [int(x) for x in s]
    else:
        parsed = [int(x) for x in str(s).split(",") if x.strip()]
    if not parsed:
        raise ValueError("--bucket-sizes must not be empty")
    return sorted(parsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end vLLM inference benchmark for the 4 e2e packed "
                    "variants (causal / sliding_window / causal_alibi / "
                    "causal_softcap) through a FlexAttention or Flashlight "
                    "backend.",
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
        choices=E2E_VARIANTS,
        default="causal_packed",
        help="E2e attention variant (ignored in baseline/debug mode).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",  # meta-llama/Llama-3.2-1B is gated
        help="HuggingFace model ID.",
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
        "--run-id", default=None,
        help="Join key for summary/per-request CSVs (default: new UUID).",
    )
    parser.add_argument(
        "--bucket-sizes",
        default=",".join(str(b) for b in DEFAULT_BUCKET_SIZES),
        help="Comma-separated padded_total bucket sizes for the e2e hook. "
             "`dynamic=False` compiles once per unique bucket; keeping this "
             "set small bounds the compile count.",
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
