"""
reference: https://github.com/microsoft/unilm/tree/master/Diff-Transformer
configs: https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L74-L78
"""
import argparse
import collections
import os

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


from tests import test_diffattn

BENCHMARK_REGISTRY = {
    'diffattn': test_diffattn.diffattn,
    'diffattn_compiled': torch.compile(dynamic=False)(test_diffattn.diffattn)
}


from _utils import Config, run_benchmark, run_test

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
    for group_size in [1]:
        for headdim in args.headdim:
            for batch_size, seqlen in zip(args.batch_size, args.seqlen):
                nheads = args.dim // headdim
                config = Config(batch_size, seqlen, nheads, headdim, group_size, args.dropout_p)
                # config = Config(4, 4096, 32, 64, False, 0.0)
                print(f"### Config: {config} ###")
                # Calculate FLOPS
                # flop_fwd = TODO
                # flop_bwd = flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd")

                # run_test(config, 'diffattn', multihead_diffattn.diffattn, flops=flop_fwd, make_qkv=multihead_diffattn.make_input)
                # run_test(config, 'diffattn_compiled', torch.compile(dynamic=False)(multihead_diffattn.diffattn), flops=flop_fwd, make_qkv=multihead_diffattn.make_input)

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
                        res = run_benchmark(config, attention_name, attention_func, flops=-1, make_qkv=test_diffattn.make_input)
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
    with open(f"{os.path.dirname(__file__)}/diff_attn.csv", 'w') as f:
        f.write(tabulate(
            all_results,
            headers=headers,
            colalign=[None for _ in headers],
            tablefmt=csv,
        ))

if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    import torch._inductor.config
    torch._inductor.config.max_autotune = True  # enable autotune
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
    parser.add_argument("--dropout_p", type=float, default=DEFAULT["dropout_p"], help="Dropout probability")
    parser.add_argument("--skip_correctness", action="store_true", help="Skip correctness checks")
    args = parser.parse_args()

    main(args, BENCHMARK_REGISTRY)