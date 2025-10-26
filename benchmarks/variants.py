"""
examples: https://github.com/pytorch-labs/attention-gym/tree/6a65742f/examples/benchmark.py#L29-L41
"""
import argparse
import collections
import os
from functools import lru_cache

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

ENABLE_FLASHLIGHT = False
def apply_patch():
    global ENABLE_FLASHLIGHT
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion
    ENABLE_FLASHLIGHT = True
    # TorchDynamo, the frontend, detects static attention subgraph and replace it with FlashAttention
    # disable the replacement to enjoy Flashlight generated fused attention
    from monkeypatch import disable_flashattention_replacement
    disable_flashattention_replacement()
    print("FLASHLIGHT enabled")
# apply_patch()


from tests.test_vanilla import attention_pytorch
attention_pytorch = torch.compile(dynamic=False)(attention_pytorch)

from tests.test_alibi import attention_pytorch_alibi, generate_alibi_bias_pytorch
attention_pytorch_alibi = torch.compile(dynamic=False)(attention_pytorch_alibi)

from tests.test_softcap import attention_softcapped
attention_softcapped = torch.compile(dynamic=False)(attention_softcapped)

from tests.test_causal import attention_pytorch_causal
attention_pytorch_causal = torch.compile(dynamic=False)(attention_pytorch_causal)

from tests.test_sliding_window import attention_pytorch_sliding_window, get_sliding_mask
attention_pytorch_sliding_window = torch.compile(dynamic=False)(attention_pytorch_sliding_window)

from tests.test_prefix_lm import attention_pytorch_prefix_lm, get_prefix_lm_mask
attention_pytorch_prefix_lm = torch.compile(dynamic=False)(attention_pytorch_prefix_lm)

from tests.test_document_mask import attention_pytorch_document_mask, create_document_id
attention_pytorch_document_mask = torch.compile(dynamic=False)(attention_pytorch_document_mask)


# Flex-able benchmarks from https://github.com/pytorch-labs/attention-gym/tree/6a65742f/examples/benchmark.py#L29-L41
ATTENTION_REGISTRY = {
    "full": lambda q, k, v: attention_pytorch(
        q, k, v
    ),
    "flex_full": lambda q, k, v: flex_attention(q, k, v, score_mod=_identity),

    "full_with_alibi": lambda q, k, v: attention_pytorch_alibi(
        q, k, v, score_mod=generate_alibi_bias_pytorch(q.size(-3))# dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full_with_alibi": lambda q, k, v: flex_attention(q, k, v, score_mod=generate_alibi_bias(q.size(-3))),

    "full_with_softcap": lambda q, k, v: attention_softcapped(q, k, v, score_mod=generate_tanh_softcap(30, approx=False)
    ),
    "flex_full_with_softcap": lambda q, k, v: flex_attention(q, k, v, score_mod=generate_tanh_softcap(30, approx=False)),

    "full_with_causal": lambda q, k, v: attention_pytorch_causal(
        q, k, v
    ),
    "flex_full_with_causal": lambda q, k, v: flex_attention(q, k, v, block_mask=create_block_mask_cached(causal_mask, B=q.size(0), H=q.size(1), M=q.size(2), N=k.size(2))),

    "full_with_sliding_window": lambda q, k, v: attention_pytorch_sliding_window(
        q, k, v, window_size=256, attn_mask=get_sliding_mask(q, 256), # dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full_with_sliding_window": lambda q, k, v: flex_attention(q, k, v,block_mask=create_block_mask_cached(generate_sliding_window(window_size=256), B=q.size(0),H=q.size(1),   M=q.size(2),  N=k.size(2))),
    
    "full_with_prefix_lm": lambda q, k, v: attention_pytorch_prefix_lm(
        q, k, v, prefix_lengths=256, attn_mask=get_prefix_lm_mask(q, 256) # dropout_p=dropout_p#, is_causal=causal
    ),
    "flex_full_with_prefix_lm": lambda q, k, v: flex_attention(q, k, v, block_mask=create_block_mask_cached(generate_prefix_lm_mask(prefix_length=256),B=q.size(0),H=q.size(1),   M=q.size(2),  N=k.size(2))),
    
    "full_with_document_mask": lambda q, k, v: attention_pytorch_document_mask(
        q, k, v, document_id=create_document_id(q.size(0), q.size(2), num_docs=12)
    ),
    "flex_full_with_document_mask": lambda q, k, v: flex_attention(q, k, v, block_mask=create_block_mask_cached(generate_doc_mask_mod(max_seq_len=q.size(2), num_docs=12), B=q.size(0), H=q.size(1), M=q.size(2), N=k.size(2))),
    
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
    # https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L74-L78
    FLASH_CONFIGS = {
        "batch_sizes": [32, 16, 8, 4, 2, 1],
        "seq_lengths": [512, 1024, 2048, 4096, 8192, 16384],
        "head_dims": [64, 128],
        "model_dim": 2048,
        "causal": [False],
        "dropout_p": 0.0,
    }
    # https://arxiv.org/pdf/2412.05496
    FLEX_CONFIGS = {
        "batch_sizes": [64, 16, 4, 1],
        # "batch_sizes": [32],
        "seq_lengths": [1024, 4096, 16384, 65536],
        "head_dims": [64],
        "model_dim": 1024,  # 16 heads
        "causal": [False],
        "dropout_p": 0.0,
    }
    DEFAULT = FLEX_CONFIGS
    DEFAULT_MODEL_DIM = DEFAULT['model_dim']
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