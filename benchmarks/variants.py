"""
examples: https://github.com/pytorch-labs/attention-gym/tree/6a65742f/examples/benchmark.py#L29-L41
configs: https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L74-L78
"""
import argparse
import collections
import os
from functools import lru_cache
import flash_attn.utils.benchmark

import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 65536

###########
# formatting
###########
from tabulate import tabulate, simple_separated_format
csv = simple_separated_format(',')

Result = collections.namedtuple('Result',
    ["Implementation", "FW_Time_ms", "FW_TFLOPS", 
     #"BW_Time_ms", "BW_TFLOPS", "Total_Time_ms", "Total_TFLOPS"
    ])


###########
# targets
###########


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
    from attn_gym.masks import (
        causal_mask,
        generate_sliding_window,
        generate_prefix_lm_mask,
        generate_doc_mask_mod,

    )
    from attn_gym.mods import (
        generate_alibi_bias,
        generate_tanh_softcap
    )
except ImportError:
    print("attn_gym IMPORT ERROR")
    def causal_mask(b, h, q, k): return torch.ones((b, h, q, k), device="cuda", dtype=torch.bool)
    def generate_sliding_window(window_size): return lambda b, h, q, k: torch.tril(torch.ones(q, k), diagonal=window_size)
    def generate_prefix_lm_mask(prefix_length): return lambda b, h, q, k: torch.tril(torch.ones(q, k), diagonal=prefix_length)
    def generate_doc_mask_mod(b, h, q, k): return torch.ones((b, h, q, k), device="cuda", dtype=torch.bool)
    def generate_alibi_bias(nheads): return lambda s, a, q, k: s - torch.arange(nheads, device="cuda").view(1, -1, 1, 1)
    def generate_tanh_softcap(cap, approx): return lambda s, a, q, k: cap * torch.tanh(s / cap)


ENABLE_FLASHLIGHT = False
def apply_patch():
    global ENABLE_FLASHLIGHT
    from monkeypatch import disable_flashattention_replacement
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion
    ENABLE_FLASHLIGHT = True
    disable_flashattention_replacement()
    print("FLASHLIGHT enabled")
apply_patch()


from tests.test_vanilla import attention_pytorch
attention_pytorch = torch.compile(dynamic=False)(attention_pytorch)

from tests.test_alibi import attention_pytorch_alibi, generate_alibi_bias_pytorch
attention_pytorch_alibi = torch.compile(dynamic=False)(attention_pytorch_alibi)

from tests.test_softcap import attention_softcapped
attention_softcapped = torch.compile(dynamic=False)(attention_softcapped)


from tests.test_causal import attention_pytorch_causal, get_causal_mask
attention_pytorch_causal = torch.compile(dynamic=False)(attention_pytorch_causal)

from tests.test_sliding_window import attention_pytorch_sliding_window, get_sliding_mask
attention_pytorch_sliding_window = torch.compile(dynamic=False)(attention_pytorch_sliding_window)

from tests.test_prefix_lm import attention_pytorch_prefix_lm, get_prefix_lm_mask
attention_pytorch_prefix_lm = torch.compile(dynamic=False)(attention_pytorch_prefix_lm)






# flex-able benchmarks
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

    # "full_with_causal": lambda q, k, v: attention_pytorch_causal(
    #     q, k, v, attn_mask=get_causal_mask(q.shape[-2], k.shape[-2], "cuda")# dropout_p=dropout_p#, is_causal=causal
    # ),
    # "flex_full_with_causal": lambda q, k, v: flex_attention(q, k, v, block_mask=create_block_mask_cached(causal_mask, B=q.size(0),H=q.size(1),   M=q.size(2),  N=k.size(2))),

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

from _utils import AttentionConfig as Config, run_benchmark, run_test

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

def main(args, benchmark_registry):
    all_results = SubList()
    for causal in args.causal:
        for headdim in args.headdim:
            for batch_size, seqlen in zip(args.batch_size, args.seqlen):
                nheads = args.dim // headdim
                config = Config(batch_size, seqlen, nheads, headdim, causal, args.dropout_p)
                # config = Config(4, 4096, 32, 64, False, 0.0)
                print(f"### Config: {config} ###")
                # Calculate FLOPS
                flop_fwd = config.nflop(mode="fwd")
                # flop_bwd = flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd")
                results = all_results.sublist()
                for attention_name, attention_func in benchmark_registry.items():
                    assert callable(attention_func), attention_name
                    print(attention_name)

                    # run_torch_profiler(config, attention_name, attention_func, flops=flop_fwd)
                    # return

                    # res = run_test(config, attention_name, attention_func, flops=flop_fwd)
                    # return
                    # continue

                    # res = run_benchmark(config, attention_name, attention_func, flops=flop_fwd)
                    # print(res)
                    # return  # run only one config
                    # continue

                    try:
                        res = run_benchmark(config, attention_name, attention_func, flops=flop_fwd)
                    except:
                        res = (float('nan'), float('nan'))
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
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    ##################
    # configurations
    ##################
    DEFAULT_CONFIGS = {
        "batch_sizes": [32, 16, 8, 4, 2, 1],
        "seq_lengths": [512, 1024, 2048, 4096, 8192, 16384],
        "head_dims": [64, 128],
        # "head_dims": [128],
        "causal": [False],
        "dropout_p": 0.0,
    }
    DEFAULT = DEFAULT_CONFIGS
    DEFAULT_MODEL_DIM = 2048
    parser = argparse.ArgumentParser(description="Attention Benchmark")
    parser.add_argument("--implementations", type=str, nargs="+", default=["all"], help="List of implementations to benchmark")
    parser.add_argument("--batch_size", type=int, nargs="+", default=DEFAULT["batch_sizes"], help="Batch size")
    parser.add_argument("--seqlen", type=int, nargs="+", default=DEFAULT["seq_lengths"], help="Sequence length")
    parser.add_argument("--dim", type=int, default=DEFAULT_MODEL_DIM, help="Input dimension")
    parser.add_argument("--headdim", type=int, nargs="+", default=DEFAULT["head_dims"], help="Head dimension")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, help="Use causal attention")
    parser.add_argument("--dropout_p", type=float, default=DEFAULT["dropout_p"], help="Dropout probability")
    parser.add_argument("--skip_correctness", action="store_true", help="Skip correctness checks")
    args = parser.parse_args()
    args.causal = [args.causal] if args.causal is not None else DEFAULT["causal"]

    # main(args, ATTENTION_REGISTRY)

    registry = {
        attention_name: attn for attention_name, attn in ATTENTION_REGISTRY.items()
        if (ENABLE_FLASHLIGHT ^ attention_name.startswith("flex"))
        # if 'sliding' in attention_name
        # if 'alibi' in attention_name
        # if 'full_' not in attention_name  # only 'full'
    }
    main(args, registry)