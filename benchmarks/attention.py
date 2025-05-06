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

    # Estimate the runtime of the function
    # start_event = di.Event(enable_timing=True)
    # end_event = di.Event(enable_timing=True)
    # start_event.record()
    # for _ in range(5):
    #     cache.zero_()
    #     fn()
    # end_event.record()
    # di.synchronize()
    # estimate_ms = start_event.elapsed_time(end_event) / 5

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


try:
    from attn_gym.masks import causal_mask
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
except ImportError:
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
# apply_patch()

# # Mask modifications
# MASK_MODS = {
#     "causal": causal_mask,
#     "sliding_window": generate_sliding_window(window_size=1024),
#     "prefix_lm": generate_prefix_lm_mask(prefix_length=1024),
#     "document": generate_doc_mask_mod,
# }
# block_mask = create_block_mask(mask_mod, 1, 1, seqlen, seqlen, device=device) if mask_mod else None

# # Score modifications
# SCORE_MODS = {
#     "alibi": generate_alibi_bias(16),
#     "softcap": generate_tanh_softcap(30, approx=False),
#     "softcap_approx": generate_tanh_softcap(30, approx=True),
# }

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
    # attn_bias = torch.zeros(L, S, dtype=query.dtype)
    # if is_causal:
    #     assert attn_mask is None
    #     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
    #     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    #     attn_bias.to(query.dtype)

    # if attn_mask is not None:
    #     if attn_mask.dtype == torch.bool:
    #         attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    #     else:
    #         attn_bias += attn_mask

    # if enable_gqa:
    #     key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
    #     value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value



# def _make_bench():
#     ref_func = lambda: attention_pytorch(qkv, dropout_p, causal)
#     if impl == "flex_attention":
#         if flex_attention is None or causal_mask is None or create_block_mask is None:
#             return

#         flex_attn_func = lambda: flex_attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], block_mask=block_mask)
#     elif impl == "flash_attention":
#         if flash_attn_qkvpacked_func is None:
#             return
#         flash_attn_func = lambda: flash_attn_qkvpacked_func(qkv, dropout_p, causal=causal)
#     elif impl == "pytorch":
#         pytorch_attn_func = lambda: attention_pytorch(qkv, dropout_p, causal)
#         ref_func = pytorch_attn_func
#     elif impl == "triton":
#         if attention_triton is None:
#             return
#         q, k, v = qkv.unbind(dim=2)
#         triton_attn_func = lambda: attention_triton(q, k, v, causal, headdim ** (-0.5))
#     elif impl == "xformers_cutlass":
#         if xops is None:
#             return
#         q, k, v = [rearrange(t, 'b s h d -> b s h d') for t in qkv.unbind(dim=2)]
#         attn_bias = xops.LowerTriangularMask() if causal else None
#         xformers_attn_func = lambda: xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp))
#     elif impl == "xformers_flash":
#         if xops is None:
#             return
#         q, k, v = [rearrange(t, 'b s h d -> b s h d') for t in qkv.unbind(dim=2)]
#         attn_bias = xops.LowerTriangularMask() if causal else None
#         xformers_attn_func = lambda: xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp))


##################
# configurations
##################

CONFIGS = {
    "batch_sizes": [32, 16, 8, 4, 2, 1],
    "seq_lengths": [512, 1024, 2048, 4096, 8192, 16384],
    "head_dims": [64, 128],
    "causal": [True, False],
    "dropout_p": 0.0,
}

# 'qkv' in name uses packed qkv input
ATTENTION_REGISTRY = {
    "full": lambda q, k, v: attention_pytorch(
        q, k, v,# dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full": lambda q, k, v: flex_attention(q, k, v, score_mod=_identity),

    # "pytorch_sdpa": lambda q, k, v: F.scaled_dot_product_attention(
    #     q, k, v,# dropout_p=dropout_p#, is_causal=causal
    # ),
    # "flex": lambda qkv, score_mod, block_mask: flex_attention(
    #     qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], score_mod=score_mod, block_mask=block_mask
    # ),
    # "flash2": flash_attn_qkvpacked_func,
    # "triton": attention_triton if attention_triton else None,
    # "xformers.c": lambda q, k, v, causal: xops.memory_efficient_attention(
    #     q, k, v, attn_bias=xops.LowerTriangularMask() if causal else None,
    #     op=(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp)
    # ) if xops else None,
    # "xformers.f": lambda q, k, v, causal: xops.memory_efficient_attention(
    #     q, k, v, attn_bias=xops.LowerTriangularMask() if causal else None,
    #     op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
    # ) if xops else None,
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

    # # Correctness check
    # if not skip_correctness
    #     if impl != "pytorch":
    #         ref_out = ref_func()
    #         attn_out = attn_func()
    #         torch.testing.assert_close(attn_out, ref_out, atol=1e-1, rtol=1e-2)
    #         if verbose:
    #             print(f"Correctness check passed for {impl} ✅")


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
                    res = run_benchmark(config, attention_name, attention_func, flops=flop_fwd)
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