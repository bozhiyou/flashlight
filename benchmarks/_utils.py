import collections
import math
from typing import Literal

import torch


###
# Config
###

input_dtype = torch.bfloat16  # TODO add to config

def _nflop(batch: int, seqlen: int, headdim: int, nheads: int, causal: bool,
        mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd"
    ) -> float:
    """Calculate FLOPS for attention computation."""
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return {"fwd": f, "bwd": 2.5 * f}.get(mode, 3.5 * f)


Config = collections.namedtuple('Config',
    ['batch_size', 'seqlen', 'nheads', 'headdim', 'causal', 'dropout_p'])

class AttentionConfig(Config):
    def nflop(self, mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd") -> float:
        """Calculate FLOPS for attention computation. Static to given config."""
        return _nflop(self.batch_size, self.seqlen, self.headdim, self.nheads, self.causal, mode=mode)


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
    return getattr(torch, return_mode)(times).item()


def benchmark_forward(
    fn, *inputs,
    repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
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
        rep=repeats, return_mode='median'
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
    batch_size, seqlen, nheads, headdim, causal, dropout_p = config
    
    device: str = "cuda"
    mode: Literal["fwd", "bwd", "fwd_bwd"] = "fwd"
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=input_dtype, requires_grad='bwd' in mode)
    args = (qkv,) if "qkv" in attention_name else torch_order(qkv)

    return args


################
# metrics
################

def efficiency(flop: int, time: float, unit: Literal['s', 'ms'] = 's') -> float:
    """TFLOP/s"""
    assert time
    scalar = {'s': 1e12, 'ms': 1e9}[unit]
    return (flop / time / scalar) if not math.isnan(time) else 0.0




###########
# Interface
###########

SHOULD_VERIFY = False
def run_benchmark(config, attention_name: str, attention_func, *, flops: int = 0, make_qkv=_make_qkv):
    args = make_qkv(config, attention_name)
    target = lambda: attention_func(*args)
    # warmup
    for _ in range(5):
        target()

    time_f = benchmark_forward(attention_func, *args) # if mode == 'fwd' else do_bench(lambda: out.backward(torch.randn_like(out)))
    # print("bmk: ", benchmarker.benchmark_gpu(target))
    # print("bmk: ", benchmarker.benchmark_gpu(target))
    # print("bmk: ", benchmarker.benchmark_gpu(target))
    # Calculate TFLOPS
    tflops_fwd = efficiency(flops, time_f)
    # tfps_bwd = efficiency(flop_bwd, time_b)

    # === Correctness check ===
    if SHOULD_VERIFY and not attention_name.startswith("full"):
        # ref_name = "full"
        ref_name = attention_name.replace("flex_", "")
        print(f"verify with {ref_name}")
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


def run_test(config, attention_name: str, attention_func, *, flops: int = 0, make_qkv=_make_qkv):
    args = make_qkv(config, attention_name)
    target = lambda: attention_func(*args)

    target()


def run_torch_profiler(config, attention_name: str, attention_func,  *, flops: int = 0, make_qkv=_make_qkv):
    args = make_qkv(config, attention_name)
    target = lambda: attention_func(*args)

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

