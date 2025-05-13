"""
examples: https://github.com/pytorch-labs/attention-gym/blob/main/examples/benchmark.py#L29-L41
configs: https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L74-L78
"""
import argparse
import collections
import math
import os
from einops import rearrange
from functools import lru_cache, partial
import flash_attn.utils
import flash_attn.utils.benchmark
from typing import Optional, List, Literal
import torch._dynamo
torch._dynamo.config.cache_size_limit = 65536

###########
# formatting
###########
from tabulate import tabulate, simple_separated_format
csv = simple_separated_format(',')

Config = collections.namedtuple('Config',
    ['batch_size', 'seqlen', 'nheads', 'headdim', 'causal', 'dropout_p'])
Result = collections.namedtuple('Result',
    ["Implementation", "FW_Time_ms", "FW_TFLOPS", 
     #"BW_Time_ms", "BW_TFLOPS", "Total_Time_ms", "Total_TFLOPS"
    ])




################
# metrics
################

def nflop(batch: int, seqlen: int, headdim: int, nheads: int, causal: bool,
          mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd"
    ) -> float:
    """Calculate FLOPS for attention computation."""
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return {"fwd": f, "bwd": 2.5 * f}.get(mode, 3.5 * f)

def efficiency(flop: float, time: float, unit: Literal['s', 'ms'] = 's') -> float:
    assert time
    scalar = {'s': 1e12, 'ms': 1e9}[unit]
    return (flop / time / scalar) if not math.isnan(time) else 0.0


##################
# timing
##################
import torch

def warmup_welldone(fn, n=3):
    """Cook the GPU until well-done."""
    assert callable(fn)
    import pynvml
    handle = torch.cuda._get_pynvml_handler()
    freq_MHz = pynvml.nvmlDeviceGetMaxClockInfo(handle, 1)
    while torch.cuda.clock_rate() < freq_MHz or n == 0:
        fn()
        n -= 1

# timer
def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean",
             device_type="cuda"):
    f"""
    Adapted from {triton.testing.do_bench} which defines warmup and repeatition time in ms.
    Here we define number of times.
    """
    assert return_mode in ["min", "max", "mean", "median"]

    di = torch._dynamo.device_interface.get_interface_for_device(device_type)

    fn()
    di.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    if fast_flush:
        cache = torch.empty(int(256e6 // 4), dtype=torch.int, device=device_type)
    else:
        cache = torch.empty(int(256e6), dtype=torch.int8, device=device_type)



    # compute number of warmup and repeat
    n_warmup = max(2, int(warmup // 100))  # max(1, int(warmup / estimate_ms))
    n_repeat = max(3, int(rep // 100))  # max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # warmup_welldone(fn, n=n_warmup)
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    di.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


def benchmark_forward(
    fn, *inputs,
    repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    f"""
    Shim for {flash_attn.utils.benchmark.benchmark_forward}, which includes CUDA synctime.
    Here we don't mesure CUDA sync time.
    """
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    return do_bench(
        (lambda: amp_wrapper(*inputs, **kwinputs)) if amp else (lambda: fn(*inputs, **kwinputs)),
        rep=repeats,
    )

###########
# targets
###########

import torch.nn.functional as F

try:
    import flash_attn
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None

try:
    import triton
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

# Flex attention related mask and modification functions
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
    _identity,
    _score_mod_signature,
    _mask_mod_signature,
)

@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
    return block_mask


flex_attention = torch.compile(flex_attention, dynamic=False)
try:
    from attn_gym.masks import causal_mask
    from attn_gym.masks.document_mask import length_to_offsets
    from attn_gym.masks import (
        generate_sliding_window,
        generate_prefix_lm_mask,
        generate_doc_mask_mod,

    )
    from attn_gym.mods import generate_alibi_bias, generate_tanh_softcap

    def alibi_bias(h, q_len, kv_len):
        return torch.exp2(-((torch.arange(h, dtype=dtype) + 1) * 8.0 / h))[:, None, None] * (torch.arange(kv_len, dtype=dtype)[None, :] - torch.arange(q_len, dtype=dtype)[:, None])
    def generate_alibi_bias_pytorch(nheads): return lambda q, k: alibi_bias(nheads, q, k)
except ImportError:
    print("IMPORT ERROR")
    def causal_mask(b, h, q, k): return torch.ones((b, h, q, k), device="cuda", dtype=torch.bool)
    def generate_sliding_window(window_size): return lambda b, h, q, k: torch.tril(torch.ones(q, k), diagonal=window_size)
    def generate_prefix_lm_mask(prefix_length): return lambda b, h, q, k: torch.tril(torch.ones(q, k), diagonal=prefix_length)
    def generate_doc_mask_mod(b, h, q, k): return torch.ones((b, h, q, k), device="cuda", dtype=torch.bool)
    def generate_alibi_bias(nheads): return lambda s, a, q, k: s - torch.arange(nheads, device="cuda").view(1, -1, 1, 1)
    def generate_tanh_softcap(cap, approx): return lambda s, a, q, k: cap * torch.tanh(s / cap)


torch.set_default_device("cuda")
torch.manual_seed(0)

dtype = torch.float16

def apply_patch():
    import monkeypatch.dependent_reduction_fusion
apply_patch()


from baselines import multihead_diffattn


@torch.compile
def attention_pytorch(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        attn_mask=None, dropout_p=0.0, is_causal: bool=False, scale=None, enable_gqa=False) -> torch.Tensor:
    r"""
    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
        attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
            which is :math:`(N,..., L, S)`. Two types of masks are supported.
            A boolean mask where a value of True indicates that the element *should* take part in attention.
            A float mask of the same type as query, key, value that is added to the attention score.
        dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
        is_causal (bool): If set to true, the attention masking is a lower triangular matrix when the mask is a
            square matrix. The attention masking has the form of the upper left causal bias due to the alignment
            (see :class:`torch.nn.attention.bias.CausalBias`) when the mask is a non-square matrix.
            An error is thrown if both attn_mask and is_causal are set.
        scale (optional float, keyword-only): Scaling factor applied prior to softmax. If None, the default value is set
            to :math:`\frac{1}{\sqrt{E}}`.
        enable_gqa (bool): If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.
    """
    # L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


@torch.compile
def attention_pytorch_alibi(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, score_mod=None,
        attn_mask=None, dropout_p=0.0, is_causal: bool=False, scale=None, enable_gqa=False) -> torch.Tensor:
        
    
    # Scale factor calculation
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    # # ALiBi slope generation (compiler-friendly)
    # Hq = query.size(-3)  # number of heads in query
    # device, dtype = query.device, query.dtype
    
    # # Generating the slope factor based on head index and number of heads (same as in flex_attention)
    # slopes = torch.pow(2, torch.arange(-8, -8*(Hq+1), -8, device=device) / Hq).to(dtype)

    # # Attention computation (query @ key) scaled by scale_factor
    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    # # ALiBi bias injection
    # L, S = query.size(-2), key.size(-2)
    # q_idx = torch.arange(L, device=device).view(-1, 1).to(dtype)
    # k_idx = torch.arange(S, device=device).view(1, -1).to(dtype)
    # rel_dist = (q_idx - k_idx)  # [L, S]

    # Head-specific bias addition as in flex_attention
    # attn_weight = attn_weight + (slopes.view(-1, 1, 1) * rel_dist).unsqueeze(0)
    # N, Hq, L, E = query.shape

    attn_weight = attn_weight + score_mod(attn_weight.size(-2), attn_weight.size(-1))

    attn_weight = torch.softmax(attn_weight, dim=-1)

    # Cast to value's dtype before matmul (ensuring dtype match)
    # attn_weight = attn_weight.to(value.dtype)

    # Matrix multiply with value tensor
    return attn_weight @ value


@torch.compile
def attention_softcapped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: None,
    attn_mask=None,
    dropout_p=0.0,
    is_causal: bool=False,
    scale=None,
    enable_gqa: bool=False,
    softcap_threshold: float = 30.0,
) -> torch.Tensor:
    r"""
    Scaled dot-product attention with soft-capping.

    Args:
        query (Tensor): Query tensor; shape :math:`(N, ..., Hq, L, E)`.
        key (Tensor): Key tensor; shape :math:`(N, ..., H, S, E)`.
        value (Tensor): Value tensor; shape :math:`(N, ..., H, S, Ev)`.
        attn_mask (optional Tensor): Broadcastable to :math:`(N,..., L, S)`.
            If bool, True means attend; if float, added to attention scores.
        dropout_p (float): Dropout probability.
        is_causal (bool): If True, apply lower-triangular causal mask.
        scale (optional float): Scale factor for logits (default: 1 / sqrt(E)).
        enable_gqa (bool): If True, enable Grouped Query Attention.
        softcap_threshold (float): Soft cap threshold; larger logits saturate to this value.

    Returns:
        Output tensor of shape :math:`(N, ..., Hq, L, Ev)`.
    """
    N, Hq, L, E = query.shape
    H = key.size(-3)
    S = key.size(-2)
    Ev = value.size(-1)

    scale_factor = 1 / math.sqrt(E) if scale is None else scale

    # Scaled dot product
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    # Apply soft-capping: cap * tanh(score / cap)
    attn_weight = score_mod(attn_weight,  N, Hq, query, key)

    # Softmax and dropout
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight @ value  # (N, ..., Hq, L, Ev)

@torch.compile
def attention_softcap_approx(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float = None,
    enable_gqa: bool = False,
    softcap_threshold: float = 30.0,
) -> torch.Tensor:
    r"""
    Scaled dot-product attention with softcap approximation.

    Args:
        query (Tensor): shape (N, ..., Hq, L, E)
        key (Tensor): shape (N, ..., H, S, E)
        value (Tensor): shape (N, ..., H, S, Ev)
        attn_mask (optional Tensor): Broadcastable to (..., L, S)
        dropout_p (float): Dropout probability.
        is_causal (bool): If True, apply causal masking.
        scale (float): Optional scaling factor (default = 1/sqrt(E))
        enable_gqa (bool): Enables Grouped Query Attention (GQA).
        softcap_threshold (float): Threshold for softcap approximation.

    Returns:
        Tensor of shape (N, ..., Hq, L, Ev)
    """
    N, Hq, L, E = query.shape
    H = key.size(-3)
    S = key.size(-2)

    scale_factor = 1 / math.sqrt(E) if scale is None else scale

    # Scaled dot-product attention scores
    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    # Softcap approximation: x / (1 + |x| / t)
    abs_attn = attn_weight.abs()
    attn_weight = attn_weight / (1 + attn_weight.abs() / softcap_threshold) # Corrected

    # Causal masking
    # if is_causal:
    #     causal_mask = torch.triu(torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1)
    #     attn_weight = attn_weight.masked_fill(causal_mask, float('-inf'))

    # Softmax normalization and dropout
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value


# @torch.compile
# def attention_pytorch_causal(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     attn_mask=None,
#     dropout_p: float = 0.0,
#     is_causal: bool = True,
#     scale: float = None,
#     enable_gqa: bool = False
# ) -> torch.Tensor:
#     r"""
#     Args:
#         query (Tensor): shape (N, ..., Hq, L, E)
#         key (Tensor): shape (N, ..., H, S, E)
#         value (Tensor): shape (N, ..., H, S, Ev)
#         attn_mask (Tensor, optional): Broadcastable to (..., L, S). Can be bool or float.
#         dropout_p (float): Dropout probability applied to attention weights.
#         is_causal (bool): If True, apply causal masking (upper triangle is masked).
#         scale (float, optional): Scaling factor for dot product.
#         enable_gqa (bool): If True, enables Grouped Query Attention.
        
#     """

#     N, Hq, L, E = query.shape
#     H, S = key.size(-3), key.size(-2)

#     scale_factor = 1 / math.sqrt(E) if scale is None else scale

#     # Compute raw attention scores
#     attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (N, ..., Hq, L, S)

#     # Causal masking (upper triangle is masked with -inf)
#     causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
#     attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

#     # Normalize and apply dropout
#     attn_weights = torch.softmax(attn_scores, dim=-1)

#     return attn_weights @ value

@torch.compile
def attention_pytorch_causal(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    scale: float = None,
) -> torch.Tensor:
    """
    Args:
        query: (B, H, L, D)
        key:   (B, H, S, D)
        value: (B, H, S, Dv)
        dropout_p: Dropout prob for attention weights
        scale: Optional scaling factor. Defaults to 1/sqrt(D).
    Returns:
        output: (B, H, L, Dv)
    """
    B, H, L, D = query.shape
    _, _, S, _ = key.shape

    scale = scale or 1.0 / math.sqrt(D)

    # Compute raw attention scores
    attn_scores = query @ key.transpose(-2, -1) * scale

    # # Build causal mask using broadcasting and indexing
    # q_idx = torch.arange(L, device=query.device).view(L, 1)  # (L, 1)
    # kv_idx = torch.arange(S, device=key.device).view(1, S)   # (1, S)
    # causal_mask = q_idx >= kv_idx                            # (L, S) boolean


    # Apply causal mask: convert True/False to 0/-inf
    # attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
    attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

    # Apply softmax and dropout
    attn_weights = torch.softmax(attn_scores, dim=-1)
    # if dropout_p > 0.0:
    #     attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)

    # Final attention output
    output = attn_weights @ value  # (B, H, L, Dv)
    return output


@torch.compile
def attention_pytorch_sliding_window(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window_size: int = 1024,
    attn_mask: torch.Tensor = None,
    dropout_p: float = 0.0,
    scale: float = None,
    enable_gqa: bool = False
) -> torch.Tensor:
    """
    PyTorch implementation matching FlexAttention's generate_sliding_window:
    - Left-only attention (causal)
    - Sliding window of size `window_size` behind each token
    """
    N, H, L, D = query.shape
    assert L == key.size(-2) == value.size(-2)

    scale_factor = 1.0 / math.sqrt(D) if scale is None else scale
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (N, H, L, L)

    attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

    # Apply softmax and dropout
    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value


from typing import Optional, Union
def attention_pytorch_prefix_lm(
    query: torch.Tensor,  # [B, H, S, D]
    key: torch.Tensor,
    value: torch.Tensor,
    prefix_lengths: Union[int, torch.Tensor],  # scalar or [B]
    attn_mask = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    training: bool = False,
) -> torch.Tensor:
    B, H, S, D = query.shape

    # if isinstance(prefix_lengths, int):
    #     prefix_lengths = torch.full((B,), prefix_lengths, dtype=torch.long, device=query.device)
    # assert prefix_lengths.shape == (B,), f"Expected prefix_lengths shape [B], got {prefix_lengths.shape}"

    # Scale factor
    scale = scale or (1.0 / math.sqrt(D))

    # Compute attention scores: [B, H, S, S]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # # Build combined prefix-lm-causal mask: allow k <= max(prefix_len[b]-1, q)
    # q_idx = torch.arange(S, device=query.device).view(1, 1, S, 1)  # [1, 1, S, 1]
    # k_idx = torch.arange(S, device=query.device).view(1, 1, 1, S)  # [1, 1, 1, S]
    # prefix_idx = prefix_lengths.view(B, 1, 1, 1) - 1  # [B, 1, 1, 1]

    # max_idx = torch.maximum(prefix_idx, q_idx)  # [B, 1, S, 1]
    # causal_prefix_mask = k_idx > max_idx  # [B, 1, S, S]

    attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))

    # Compute softmax over attention scores
    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value  # [B, H, S, D]


@torch.compile
def attention_pytorch_with_document(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    segment_ids: torch.Tensor,
    doc_token_mask: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    scale: float = None,
    enable_gqa: bool = False
) -> torch.Tensor:
    r"""
    Args:
        query (Tensor): shape (N, Hq, L, E)
        key (Tensor): shape (N, H, S, E)
        value (Tensor): shape (N, H, S, Ev)
        segment_ids (LongTensor): shape (N, L); segment id for each token.
        doc_token_mask (BoolTensor): shape (N, L); True for document tokens.
        attn_mask (optional Tensor): broadcastable to (N, Hq, L, S)
        dropout_p (float): Dropout probability.
        scale (float, optional): Scaling factor.
        enable_gqa (bool): Enables Grouped Query Attention.
    """
    N, Hq, L, E = query.shape
    H, S = key.size(-3), key.size(-2)

    assert L == S, "L == S expected for self-attention"

    scale_factor = 1 / math.sqrt(E) if scale is None else scale

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor  # (N, Hq, L, S)

    # Construct the segment-based attention mask
    seg_i = segment_ids.unsqueeze(2)  # (N, L, 1)
    seg_j = segment_ids.unsqueeze(1)  # (N, 1, L)
    same_segment = seg_i == seg_j     # (N, L, L)

    doc_mask = doc_token_mask.unsqueeze(1).expand(-1, L, -1)  # (N, L, L)
    can_attend = same_segment | doc_mask  # regular tokens attend to same segment + doc tokens

    # Document tokens attend to everything
    doc_as_query = doc_token_mask.unsqueeze(2).expand(-1, -1, L)  # (N, L, L)
    doc_can_attend = torch.ones_like(can_attend)
    final_mask = torch.where(doc_as_query, doc_can_attend, can_attend)  # (N, L, L)

    # Expand mask to (N, Hq, L, S)
    final_mask = final_mask.unsqueeze(1).expand(-1, Hq, -1, -1)
    attn_scores = attn_scores.masked_fill(~final_mask, float('-inf'))

    attn_weights = torch.softmax(attn_scores, dim=-1)

    return attn_weights @ value


##################
# configurations
##################

CONFIGS = {
    "batch_sizes": [32, 16, 8, 4, 2, 1],
    "seq_lengths": [512, 1024, 2048, 4096, 8192, 16384],
    "head_dims": [64, 128],
    "causal": [True],
    "dropout_p": 0.0,
}


def get_causal_mask(L: int, S: int, device: torch.device):
    q_idx = torch.arange(L, device=device).view(L, 1)
    kv_idx = torch.arange(S, device=device).view(1, S)
    return (q_idx < kv_idx).to(torch.bool)  # shape: (L, S)

def get_sliding_mask(query, window_size):
    # Create (L, L) mask: True where j ∉ [i - window_size, i]
    N, H, L, D = query.shape
    q_idx = torch.arange(L, device=query.device).view(L, 1)
    k_idx = torch.arange(L, device=query.device).view(1, L)
    causal_sliding_mask = (q_idx < k_idx) | ((q_idx - k_idx) > window_size)  # shape (L, L)

    # Expand to (N, H, L, L)
    full_mask = causal_sliding_mask.unsqueeze(0).unsqueeze(0).expand(N, H, L, L)
    return full_mask

def get_prefix_lm_mask(query, prefix_lengths):
    B, H, S, D = query.shape
    prefix_lengths = torch.full((B,), prefix_lengths, dtype=torch.long, device=query.device)
     # Build combined prefix-lm-causal mask: allow k <= max(prefix_len[b]-1, q)
    q_idx = torch.arange(S, device=query.device).view(1, 1, S, 1)  # [1, 1, S, 1]
    k_idx = torch.arange(S, device=query.device).view(1, 1, 1, S)  # [1, 1, 1, S]
    prefix_idx = prefix_lengths.view(B, 1, 1, 1) - 1  # [B, 1, 1, 1]

    max_idx = torch.maximum(prefix_idx, q_idx)  # [B, 1, S, 1]
    causal_prefix_mask = k_idx > max_idx  # [B, 1, S, S]
    return causal_prefix_mask

# 'qkv' in name uses packed qkv input
ATTENTION_REGISTRY = {
    "full": lambda q, k, v: attention_pytorch(
        q, k, v,# dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full": lambda q, k, v: flex_attention(q, k, v, score_mod=_identity),
    "full_with_alibi": lambda q, k, v: attention_pytorch_alibi(
        q, k, v, score_mod=generate_alibi_bias_pytorch(q.size(-3))# dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full_with_alibi": lambda q, k, v: flex_attention(q, k, v, score_mod=generate_alibi_bias(q.size(-3))),

    "full_with_softcap": lambda q, k, v: attention_softcapped(q, k, v, score_mod=generate_tanh_softcap(30, approx=False)
    ),
    "flex_full_with_softcap": lambda q, k, v: flex_attention(q, k, v, score_mod=generate_tanh_softcap(30, approx=False)),

    # "full_with_softcap_approx": lambda q, k, v: attention_softcap_approx(
    #     q, k, v, tahn_fnc=generate_tanh_softcap(30, approx=False) # dropout_p=dropout_p#, is_causal=causal
    # ),
    # "flex_full_with_softcap_approx": lambda q, k, v: flex_attention(q, k, v, score_mod=generate_tanh_softcap(30, approx=True)),

    "full_with_causal": lambda q, k, v: attention_pytorch_causal(
        q, k, v, attn_mask=get_causal_mask(q.shape[-2], k.shape[-2], "cuda")# dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full_with_causal": lambda q, k, v: flex_attention(q, k, v, block_mask=create_block_mask_cached(causal_mask, B=q.size(0),H=q.size(1),   M=q.size(2),  N=k.size(2))),

    "full_with_sliding_window": lambda q, k, v: attention_pytorch_sliding_window(
        q, k, v, window_size=256, attn_mask=get_sliding_mask(q, 256), # dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full_with_sliding_window": lambda q, k, v: flex_attention(q, k, v,block_mask=create_block_mask_cached(generate_sliding_window(window_size=256), B=q.size(0),H=q.size(1),   M=q.size(2),  N=k.size(2))),
    
    "full_with_prefix_lm": lambda q, k, v: attention_pytorch_prefix_lm(
        q, k, v, prefix_lengths=256, attn_mask=get_prefix_lm_mask(q, 256) # dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full_with_prefix_lm": lambda q, k, v: flex_attention(q, k, v, block_mask=create_block_mask_cached(generate_prefix_lm_mask(prefix_length=256),B=q.size(0),H=q.size(1),   M=q.size(2),  N=k.size(2))),
    
    # "full_with_document": lambda q, k, v: attention_pytorch_with_document(
    #     q, k, v,# dropout_p=dropout_p#, is_causal=causal
    # ),

}




#############
# benchmark
#############


# flashattn input shape is (batch_size, seqlen, 3, nheads, headdim)
# PyTorch input shape is (batch_size, nheads, seqlen, headdim)
def torch_order(qkv: torch.Tensor):
    q, k, v = qkv.unbind(dim=2)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()


def run_benchmark(config, attention_name: str, attention_func,
    flops: int,
    device: str = "cuda",
    mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd",
):
    batch_size, seqlen, nheads, headdim, causal, dropout_p = config
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype, requires_grad='bwd' in mode)
    if "qkv" in attention_name:
        # packed inputs
        time_f = benchmark_forward(lambda: attention_func(qkv)) # if mode == 'fwd' else do_bench(lambda: out.backward(torch.randn_like(out)))
    else:
        q, k, v = torch_order(qkv)
        time_f = benchmark_forward(lambda: attention_func(q, k, v)) # if mode == 'fwd' else do_bench(lambda: out.backward(torch.randn_like(out)))

    # Calculate TFLOPS
    tflops_fwd = efficiency(flops, time_f)
    # tfps_bwd = efficiency(flop_bwd, time_b)

    # === Correctness check ===
    if not attention_name.startswith("full"):
        # ref_name = "full"
        ref_name = attention_name.replace("flex_", "")
        print(ref_name)
        ref_func = ATTENTION_REGISTRY.get(ref_name, None)
        if ref_func is not None:
            with torch.no_grad():
                out_ref = ref_func(q, k, v)
                out_test = attention_func(q, k, v)
            try:
                torch.testing.assert_close(out_test, out_ref, rtol=1e-2, atol=1e-2)
            except AssertionError as e:
                print(f"❌ {attention_name} failed correctness check vs {ref_name}:\n{e}")
            else:
                print(f"✅ {attention_name} passed correctness check vs {ref_name}")

    return time_f, tflops_fwd


class SubList(list):
    """Hierarchical list for result collection."""
    def sublist(self):
        sublist = SubList()
        setattr(sublist, '_parent', self)
        return sublist
    
    def append(self, item):
        if hasattr(self, '_parent'):
            getattr(self, '_parent').append(item)
        return super().append(item)

def main(args):
    all_results = SubList()
    for causal in args.causal:
        for headdim in args.headdim:
            for batch_size, seqlen in zip(args.batch_size, args.seqlen):
                nheads = args.dim // headdim
                config = Config(batch_size, seqlen, nheads, headdim, causal, args.dropout_p)
                print(f"### Config: {config} ###")
                # Calculate FLOPS
                flop_fwd = nflop(batch_size, seqlen, headdim, nheads, causal, mode="fwd")
                # flop_bwd = flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd")

                results = all_results.sublist()
                for attention_name, attention_func in ATTENTION_REGISTRY.items():
                    assert callable(attention_func), attention_name
                    # res = run_benchmark(config, attention_name, attention_func, flops=flop_fwd)
                    try:
                        res = run_benchmark(config, attention_name, attention_func, flops=flop_fwd)
                    except:
                        res = (-1,-1)
                    result = Result(attention_name, *res)
                    results.append([*result, *config])
                # Print results for this config
                print(
                    tabulate(
                        results,
                        headers=Result._fields + Config._fields,
                        tablefmt="grid",
                    )
                )
    headers = Result._fields + Config._fields
    with open(f"{os.path.dirname(__file__)}/benchmark.csv", 'w') as f:
        f.write(tabulate(
            all_results,
            headers=headers,
            colalign=[None for _ in headers],
            tablefmt=csv,
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Benchmark")
    parser.add_argument("--implementations", type=str, nargs="+", default=["all"], help="List of implementations to benchmark")
    parser.add_argument("--batch_size", type=int, nargs="+", default=CONFIGS["batch_sizes"], help="Batch size")
    parser.add_argument("--seqlen", type=int, nargs="+", default=CONFIGS["seq_lengths"], help="Sequence length")
    parser.add_argument("--dim", type=int, default=2048, help="Input dimension")
    parser.add_argument("--headdim", type=int, nargs="+", default=CONFIGS["head_dims"], help="Head dimension")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, help="Use causal attention")
    parser.add_argument("--dropout_p", type=float, default=CONFIGS["dropout_p"], help="Dropout probability")
    parser.add_argument("--skip_correctness", action="store_true", help="Skip correctness checks")
    args = parser.parse_args()
    args.causal = [args.causal] if args.causal is not None else CONFIGS["causal"]
    main(args)