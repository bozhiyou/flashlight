"""
reference: https://github.com/deepspeedai/DeepSpeed/blob/master/tests/benchmarks/DS4Sci_EvoformerAttention_bench.py#L21-L55
configs: https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L74-L78
"""
import argparse
import os

import torch._dynamo.config
torch._dynamo.config.cache_size_limit = 65536


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


from attention_variants import evoformer

BENCHMARK_REGISTRY = {
    'ipa': evoformer.attention_reference,
    'ipa_compiled': torch.compile(dynamic=False)(evoformer.attention_reference)
}

from _utils import (
    Config, Result, SubList, run_benchmark,
    write_results_csv, print_results, get_gpu_suffix,
)

def main(args, benchmark_registry):
    gpu_label = get_gpu_suffix().upper() if get_gpu_suffix() != "unknown" else "unknown"
    all_results = SubList()
    for group_size in [1]:
        for headdim in args.headdim:
            for batch_size, seqlen in zip(args.batch_size, args.seqlen):
                # nheads = args.dim // headdim
                seqlen = 256
                nheads = 4
                config = Config(batch_size, seqlen, nheads, headdim, group_size, args.dropout_p)
                # config = Config(4, 4096, 32, 64, False, 0.0)
                print(f"### Config: {config} ###")
                # Calculate FLOPS
                flop_fwd = 4 * batch_size * 256 * seqlen**2 * nheads * headdim  # estimate as vanilla
                # flop_bwd = flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd")

                # run_test(config, 'ipa', evoformer_attention.attention_reference, flops=flop_fwd, make_qkv=evoformer_attention.make_input)
                # run_test(config, 'ipa', torch.compile(dynamic=False)(evoformer_attention.attention_reference), flops=flop_fwd, make_qkv=evoformer_attention.make_input)

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

                    # try:
                    res = run_benchmark(config, attention_name, attention_func, flops=flop_fwd, make_qkv=evoformer.make_input)
                    # except:
                    #     res = (float('nan'), float('nan'))
                    result = Result(attention_name, *res)
                    results.append([*result, *config, gpu_label])
                print_results(results)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    write_results_csv(all_results, os.path.join(out_dir, "evo_attn.csv"), extra_headers=["GPU"])

if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    import torch._inductor.config
    torch._inductor.config.max_autotune = True  # enable autotune
    ##################
    # configurations
    ##################
    IPA_CONFIGS = {
        "batch_sizes": [32, 16, 8, 4, 2, 1],
        "seq_lengths": [256] * 6,
        "head_dims": [64, 128],
        # "head_dims": [128],
        "causal": [False],
        "dropout_p": 0.0,
    }
    DEFAULT = IPA_CONFIGS
    DEFAULT_MODEL_DIM = 2048
    parser = argparse.ArgumentParser(description="Attention Benchmark")
    parser.add_argument("--implementations", type=str, nargs="+", default=["all"], help="List of implementations to benchmark")
    parser.add_argument("--batch_size", type=int, nargs="+", default=DEFAULT["batch_sizes"], help="Batch size")
    parser.add_argument("--seqlen", type=int, nargs="+", default=DEFAULT["seq_lengths"], help="Sequence length")
    parser.add_argument("--dim", type=int, default=DEFAULT_MODEL_DIM, help="Input dimension")
    parser.add_argument("--headdim", type=int, nargs="+", default=DEFAULT["head_dims"], help="Head dimension")
    parser.add_argument("--dropout_p", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--skip_correctness", action="store_true", help="Skip correctness checks")
    args = parser.parse_args()

    main(args, BENCHMARK_REGISTRY)