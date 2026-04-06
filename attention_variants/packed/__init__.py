"""Packed-doc attention variants for the Flashlight e2e backend.

Each module splits into two parts:

1. A mask builder (``build_packed_*``) that computes the ``(S, S)`` boolean
   mask from ``doc_id`` and ``offsets`` in eager Python. This must stay outside
   ``@torch.compile`` because the data-dependent indirect indexing trips the
   fusion patches' block-reduction pass.

2. An attention function (``attention_packed_*``) that takes precomputed mask
   (and optional bias) and does the fusible attention computation:
   matmul → [softcap] → mask_fill → softmax → matmul V.
   This is the part that ``@torch.compile(dynamic=False)`` fuses into a single
   Triton kernel via the ``monkeypatch/fusion`` patches.
"""
from .causal import attention_packed_causal, build_packed_causal_mask
from .sliding_window import (
    attention_packed_sliding_window,
    build_packed_sliding_window_mask,
)
from .causal_alibi import (
    attention_packed_causal_alibi,
    build_packed_causal_alibi_mask_and_bias,
)
from .causal_softcap import attention_packed_causal_softcap
from .causal_softcap import build_packed_causal_mask as build_packed_causal_softcap_mask

__all__ = [
    "attention_packed_causal",
    "build_packed_causal_mask",
    "attention_packed_sliding_window",
    "build_packed_sliding_window_mask",
    "attention_packed_causal_alibi",
    "build_packed_causal_alibi_mask_and_bias",
    "attention_packed_causal_softcap",
    "build_packed_causal_softcap_mask",
]
