import collections
import math
from typing import Literal

import torch


###
# Config
###

input_dtype = torch.bfloat16  # TODO add to config

def attention_nflop(batch: int, seqlen: int, headdim: int, nheads: int, causal: bool = False,
        mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd"
    ) -> float:
    """Calculate FLOPS for attention computation."""
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return {"fwd": f, "bwd": 2.5 * f}.get(mode, 3.5 * f)


Config = collections.namedtuple('Config',
    ['batch_size', 'seqlen', 'nheads', 'headdim', 'group_size', 'dropout_p'])


##################
# timing
##################
import torch
import time

from torch._inductor.runtime.benchmarking import benchmarker

def warmup_max(fn, n=3):
    """Heat the GPU until max frequency."""
    assert callable(fn)
    import pynvml
    handle = torch.cuda._get_pynvml_handler()
    max_freq_MHz = pynvml.nvmlDeviceGetMaxClockInfo(handle, 1)
    # _, max_power_mW = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
    last = torch.cuda.clock_rate()
    retry = 0
    while last < max_freq_MHz:
        for _ in range(n):
            fn()
        if torch.cuda.clock_rate() < last:
            # cannot boost to max freq, maybe for power limit
            if retry >= 3:
                break
            # while torch.cuda.clock_rate() / max_freq_MHz <= pynvml.nvmlDeviceGetPowerUsage(handle) / max_power_mW:
            print(f"{last} MHz | {pynvml.nvmlDeviceGetPowerUsage(handle) / 1000}W | {pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)}C")
            time.sleep(4)
            retry += 1
        last = torch.cuda.clock_rate()
    print(f"Warmup finished: {torch.cuda.clock_rate()} MHz")

# timer
def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean",
             device_type="cuda"):
    f"""
    Adapted from triton.testing.do_bench which defines warmup and repeatition time in ms.
    Here we define number of times.
    """
    assert return_mode is None or return_mode in ["min", "max", "mean", "median"]

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
    n_warmup = max(3, int(warmup // 100))  # max(1, int(warmup / estimate_ms))
    n_repeat = max(20, int(rep // 100))  # max(1, int(rep / estimate_ms))
    start_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [di.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    try:
        warmup_max(fn, n=n_warmup)
    except:
        for _ in range(n_warmup):
            fn()
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
    if return_mode is None:
        return [times.item()] if len(times) == 1 else times.tolist()
    return getattr(torch, return_mode)(times).item()


def benchmark_forward(
    fn, *inputs,
    repeats=10, return_mode='mean', desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    f"""
    Shim for `flash_attn.utils.benchmark.benchmark_forward`, which includes CUDA synctime.
    Here we don't mesure CUDA sync time.
    """
    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    return do_bench(
        (lambda: amp_wrapper(*inputs, **kwinputs)) if amp else (lambda: fn(*inputs, **kwinputs)),
        rep=repeats, return_mode=return_mode
    )



#############
# Input data
#############


# flashattn input shape is (batch_size, seqlen, 3, nheads, headdim)
# flex_attention input shape is (batch_size, nheads, seqlen, headdim)
def torch_order(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = qkv.unbind(dim=2)
    return q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()

def _make_qkv(config, attention_name):
    batch_size, seqlen, nheads, headdim, group_size, dropout_p = config
    if nheads % group_size != 0 or group_size > nheads:
        raise ValueError(
            f"Expect number of query heads to be a multiple of kv heads for GQA "
            f"Cannot divide Hq={nheads} into groups of {group_size}."
        )
    enable_gqa = group_size != 1
    
    device: str = "cuda"
    mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd"
    if not enable_gqa:
        qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=input_dtype, requires_grad='bwd' in mode)
        args = (qkv,) if "qkv" in attention_name else torch_order(qkv)
        return args
    return (
        torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=input_dtype, requires_grad='bwd' in mode),
        torch.randn(batch_size, nheads // group_size, seqlen, headdim, device=device, dtype=input_dtype, requires_grad='bwd' in mode),
        torch.randn(batch_size, nheads // group_size, seqlen, headdim, device=device, dtype=input_dtype, requires_grad='bwd' in mode),
    )


################
# metrics
################

def efficiency(flop: int, time: float, time_unit: Literal['s', 'ms'] = 'ms') -> float:
    """TFLOP/s"""
    assert time
    scalar = {'s': 1e12, 'ms': 1e9}[time_unit]
    return (flop / time / scalar) if not math.isnan(time) else 0.0




###########
# Interface
###########

def run_benchmark(config, attention_name: str, attention_func, *, flops: int = 0, make_qkv=_make_qkv, return_mode='mean'):
    args = make_qkv(config, attention_name)

    time_f = benchmark_forward(attention_func, *args, return_mode=return_mode, enable_gqa=config.group_size != 1) # if mode == 'fwd' else do_bench(lambda: out.backward(torch.randn_like(out)))
    # Calculate TFLOPS
    tflops_fwd = -1
    if flops > 0:
        t = torch.mean(torch.tensor(time_f)).item() if isinstance(time_f, list) else time_f
        tflops_fwd = efficiency(flops, t, time_unit='ms')
    # tfps_bwd = efficiency(flop_bwd, time_b)

    return time_f, tflops_fwd


def run_test(config, attention_name: str, attention_func, *, flops: int = 0, make_qkv=_make_qkv):
    args = make_qkv(config, attention_name)
    target = lambda: attention_func(*args, enable_gqa = config.group_size != 1)

    target()


def run_torch_profiler(config, attention_name: str, attention_func,  *, flops: int = 0, make_qkv=_make_qkv):
    args = make_qkv(config, attention_name)
    target = lambda: attention_func(*args, enable_gqa = config.group_size != 1)

    from torch.profiler import profile, ProfilerActivity
    import json
    import uuid

    target()  # warm up
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        target()

    from torch._dynamo.config import debug_dir_root
    fname = f"{debug_dir_root}/{uuid.uuid4().hex}.json"
    print(f"profile trace exported to {fname}")
    prof.export_chrome_trace(fname)
    with open(fname) as f:
        trace = json.load(f)

    print("===kernels===")
    # cats = set()
    for e in trace["traceEvents"]:
        # cats.add(e.get("cat"))
        if e.get("cat") == "kernel":
            print(e["name"].lower())
    # print(cats)
    print("^^^kernels^^^")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

