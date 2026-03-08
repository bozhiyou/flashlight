"""
examples: https://github.com/pytorch-labs/attention-gym/tree/6a65742f/examples/benchmark.py#L29-L41
"""
import argparse
import os
from functools import lru_cache

import torch
import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 65536


###########
# FlexAttention and its benchmark helpers (attention-gym)
###########

from torch.nn.attention.flex_attention import flex_attention
flex_attention = torch.compile(flex_attention, dynamic=False)

# Flex attention related mask and modification functions
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    _identity,
    create_mask,
    create_block_mask,
    _score_mod_signature,
    _mask_mod_signature,
)

@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
    return block_mask

from attn_gym.mods import (
    generate_alibi_bias,
    generate_tanh_softcap
)

from attn_gym.masks import (
    causal_mask,
    generate_sliding_window,
    generate_prefix_lm_mask,
    document_mask
)

# Benchmarks from https://github.com/pytorch-labs/attention-gym/tree/6a65742f/examples/benchmark.py#L29-L41
FLEX_ATTENTION_REGISTRY = {
    "flex_full": lambda q, k, v, **kwargs: flex_attention(q, k, v, score_mod=_identity, **kwargs),
    "flex_full_with_alibi": lambda q, k, v, **kwargs: flex_attention(q, k, v, score_mod=generate_alibi_bias(q.size(-3)), **kwargs),
    "flex_full_with_softcap": lambda q, k, v, **kwargs: flex_attention(q, k, v, score_mod=generate_tanh_softcap(30, approx=False), **kwargs),
    "flex_full_with_causal": lambda q, k, v, mask_mod, **kwargs: flex_attention(q, k, v, block_mask=create_block_mask_cached(mask_mod, 1, 1, q.size(-2), k.size(-2)), **kwargs),
    "flex_full_with_sliding_window": lambda q, k, v, mask_mod, **kwargs: flex_attention(q, k, v,block_mask=create_block_mask_cached(mask_mod, 1, 1, q.size(-2), k.size(-2)), **kwargs),
    "flex_full_with_prefix_lm": lambda q, k, v, mask_mod, **kwargs: flex_attention(q, k, v, block_mask=create_block_mask_cached(mask_mod, 1, 1, q.size(-2), k.size(-2)), **kwargs),
    "flex_full_with_document_mask": lambda q, k, v, mask_mod, **kwargs: flex_attention(q, k, v, block_mask=create_block_mask_cached(mask_mod, 1, 1, q.size(-2), k.size(-2)), **kwargs),
}

FLEX_MASK_REGISTRY = {
    "flex_full_with_causal": lambda config: causal_mask,
    "flex_full_with_sliding_window": lambda config: generate_sliding_window(window_size=256),
    "flex_full_with_prefix_lm": lambda config: generate_prefix_lm_mask(prefix_length=256),
    "flex_full_with_document_mask": lambda config: generate_doc_mask_mod(max_seq_len=config.seqlen, num_docs=12),
}


def generate_doc_mask_mod(max_seq_len: int, num_docs: int = 12):
    """https://github.com/meta-pytorch/attention-gym/blob/6a65742f/examples/benchmark.py#L201-L220"""
    import random

    random.seed(0)

    def generate_random_lengths(total_length, num_documents):
        # Initialize all lengths to 1 to ensure each document has at least one token
        lengths = [1] * num_documents
        remaining_length = total_length - num_documents

        # Randomly distribute the remaining length
        for _ in range(remaining_length):
            index = random.randint(0, num_documents - 1)
            lengths[index] += 1

        return lengths

    lengths = generate_random_lengths(max_seq_len, num_docs)
    offsets = document_mask.length_to_offsets(lengths, "cuda")
    document_causal_mask = document_mask.generate_doc_mask_mod(causal_mask, offsets)
    return document_causal_mask


# -- Flashlight --

_ENABLE_FLASHLIGHT = False
def _torch_compile_attn(enable_flashlight=False):
    global _ENABLE_FLASHLIGHT, torch
    if enable_flashlight:
        # enable flashlight by importing patches
        from monkeypatch.fusion import dependent_reduction_fusion
        from monkeypatch.fusion import block_reduction
        from monkeypatch.fusion import reduction_kernel_fusion
        _ENABLE_FLASHLIGHT = True
        print("FLASHLIGHT enabled")
        import torch._inductor.config
        torch._inductor.config.max_autotune = True  # enable autotune
        # TorchDynamo, the frontend, detects static attention subgraph and replace it with FlashAttention
        # disable the replacement to enjoy Flashlight generated fused attention
        from monkeypatch import disable_flashattention_replacement
        disable_flashattention_replacement()

    from attention_variants.vanilla import attention_pytorch
    from attention_variants.alibi import attention_pytorch_alibi, generate_alibi_bias_pytorch
    from attention_variants.softcap import attention_softcapped
    from attention_variants.causal import attention_pytorch_causal
    from attention_variants.sliding_window import attention_pytorch_sliding_window
    from attention_variants.prefix_lm import attention_pytorch_prefix_lm
    from attention_variants.document_mask import attention_pytorch_document_mask, create_document_id

    # compile after patches are applied
    attention_pytorch = torch.compile(dynamic=False)(attention_pytorch)
    attention_pytorch_alibi = torch.compile(dynamic=False)(attention_pytorch_alibi)
    attention_softcapped = torch.compile(dynamic=False)(attention_softcapped)
    attention_pytorch_causal = torch.compile(dynamic=False)(attention_pytorch_causal)
    attention_pytorch_sliding_window = torch.compile(dynamic=False)(attention_pytorch_sliding_window)
    attention_pytorch_prefix_lm = torch.compile(dynamic=False)(attention_pytorch_prefix_lm)
    attention_pytorch_document_mask = torch.compile(dynamic=False)(attention_pytorch_document_mask)

    return {
        "full": lambda q, k, v, **kwargs: attention_pytorch(
            q, k, v, **kwargs
        ),
        "full_with_alibi": lambda q, k, v, **kwargs: attention_pytorch_alibi(
            q, k, v, score_mod=generate_alibi_bias_pytorch(q.size(-3)), **kwargs
        ),
        "full_with_softcap": lambda q, k, v, **kwargs: attention_softcapped(q, k, v, score_mod=generate_tanh_softcap(30, approx=False), **kwargs
        ),
        "full_with_causal": lambda q, k, v, **kwargs: attention_pytorch_causal(
            q, k, v, **kwargs
        ),
        "full_with_sliding_window": lambda q, k, v, **kwargs: attention_pytorch_sliding_window(
            q, k, v, window_size=256, **kwargs
        ),
        "full_with_prefix_lm": lambda q, k, v, **kwargs: attention_pytorch_prefix_lm(
            q, k, v, prefix_lengths=256 , **kwargs
        ),
        "full_with_document_mask": lambda q, k, v, **kwargs: attention_pytorch_document_mask(
            q, k, v, document_id=create_document_id(q.size(0), q.size(2), num_docs=12), **kwargs
        ),
    }


from _utils import (
    Config, Result, SubList, run_benchmark, attention_nflop,
    write_results_csv, print_results,
)

def main(args, benchmark_registry):
    all_results = SubList()
    for group_size in args.group_size:
        for headdim in args.headdim:
            for batch_size, seqlen in zip(args.batch_size, args.seqlen):
                nheads = args.dim // headdim
                config = Config(batch_size, seqlen, nheads, headdim, group_size, args.dropout_p)
                # config = Config(4, 4096, 32, 64, False, 0.0)
                print(f"### Config: {config} ###")
                results = all_results.sublist()
                for attention_name, attention_func in benchmark_registry.items():
                    if args.filter and all(keyword not in attention_name for keyword in args.filter):
                        continue
                    assert callable(attention_func), attention_name
                    print(attention_name)
                    if mask_mod := FLEX_MASK_REGISTRY.get(attention_name, None) or FLEX_MASK_REGISTRY.get(f"flex_{attention_name}", None):
                        sparsity = create_block_mask(mask_mod(config), 1, 1, seqlen, seqlen).sparsity()
                        flop_fwd = (100 - sparsity) / 100 * attention_nflop(batch_size, seqlen, nheads, headdim, causal=False, mode="fwd")
                    else:
                        flop_fwd = attention_nflop(batch_size, seqlen, nheads, headdim, causal='causal' in attention_name, mode="fwd")
                    # flop_bwd = flops(batch_size, seqlen, headdim, nheads, mode="bwd")

                    kwargs = {}
                    if attention_name in FLEX_MASK_REGISTRY:
                        kwargs['mask_mod'] = FLEX_MASK_REGISTRY[attention_name](config)

                    # run_torch_profiler(config, attention_name, attention_func, flops=flop_fwd, **kwargs)
                    # return

                    # res = run_test(config, attention_name, attention_func, flops=flop_fwd, **kwargs)
                    # return
                    # continue

                    # res = run_benchmark(config, attention_name, attention_func, flops=flop_fwd, **kwargs)
                    # print(res)
                    # return  # run only one config
                    # continue

                    # try:
                    time_ms, tflops = run_benchmark(config, attention_name, attention_func, flops=flop_fwd, return_mode=None, **kwargs)
                    # except:
                    #     time_ms, tflops = float('nan'), float('nan')
                    for t in time_ms if isinstance(time_ms, list) else [time_ms]:
                        result = Result(attention_name, t, tflops)
                        results.append([*result, *config])
                print_results(results)

    write_results_csv(all_results, os.path.join(os.path.dirname(__file__), f"{args.output}.csv"))

if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    ##################
    # configurations
    ##################
    # https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L74-L78
    FLASH_CONFIGS = {
        "batch_sizes": [32, 16, 8, 4, 2, 1],
        "seq_lengths": [512, 1024, 2048, 4096, 8192, 16384],
        "head_dims": [64, 128],
        "group_size": [1],  # no gqa
        "model_dim": 2048,
    }
    # https://arxiv.org/pdf/2412.05496
    FLEX_CONFIGS = {
        "batch_sizes": [32, 16, 8, 4, 2, 1],
        "seq_lengths": [512, 1024, 2048, 4096, 8192, 16384],
        "head_dims": [64],
        "group_size": [1, 8],
        "model_dim": 1024,  # 16 heads
    }
    DEFAULT = FLEX_CONFIGS
    DEFAULT_MODEL_DIM = DEFAULT['model_dim']
    parser = argparse.ArgumentParser(description="Flex-able Attention Benchmark")
    _targets = parser.add_argument_group("targets")
    _targets.add_argument("--flex", action='store_true', help="benchmark FlexAttention")
    _targets.add_argument("--flashlight", action='store_true', help="benchmark Flashlight")
    _targets.add_argument("--torch.compile", action='store_true', help="benchmark torch.compile")
    _devs = parser.add_argument_group("dev-options")
    _devs.add_argument("--filter", type=str, nargs="+", default=[], help="keywords of variant to benchmark")
    _devs.add_argument("--mask-cache", action=argparse.BooleanOptionalAction, default=True, help="(FlexAttention) enable/disable mask caching (enabled by default)")
    parser.add_argument("--batch_size", type=int, nargs="+", default=DEFAULT["batch_sizes"], help="Batch size")
    parser.add_argument("--seqlen", type=int, nargs="+", default=DEFAULT["seq_lengths"], help="Sequence length")
    parser.add_argument("--dim", type=int, default=DEFAULT_MODEL_DIM, help="Input dimension")
    parser.add_argument("--headdim", type=int, nargs="+", default=DEFAULT["head_dims"], help="Head dimension")
    parser.add_argument("--group_size", type=int, nargs="+", default=DEFAULT["group_size"], help="(GQA) query group size = Hq // Hkv")
    parser.add_argument("--dropout_p", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--skip_correctness", action="store_true", help="Skip correctness checks")
    parser.add_argument("--output", type=str, default='benchmark', help='output file name, `.csv` will be appended')
    args = parser.parse_args()
    if not args.mask_cache:
        create_block_mask_cached = create_block_mask

    if args.flex:
        main(args, FLEX_ATTENTION_REGISTRY)

    if getattr(args, 'torch.compile', False):
        main(args, _torch_compile_attn())

    # always run flashlight at last because it modifies PyTorch compiler
    if args.flashlight:
        main(args, _torch_compile_attn(enable_flashlight=True))

    # main(args, 
    #     benchmark_registry= {
    #         attention_name: attn
    #         for attention_name, attn in (FLASHLIGHT_ATTENTION_REGISTRY if _ENABLE_FLASHLIGHT else FLEX_ATTENTION_REGISTRY).items()
    #             # if 'sliding' in attention_name
    #             # if 'alibi' in attention_name
    #             # if 'full_' not in attention_name  # only 'full'
    #     }
    # )