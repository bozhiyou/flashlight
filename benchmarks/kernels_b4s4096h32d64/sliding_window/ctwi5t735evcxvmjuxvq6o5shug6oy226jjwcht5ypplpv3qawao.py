# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_byou/xy/cxyjyqevvc5f2evxcnnbkpycnqdzrktnwltvbfnl5ihrweynxgqn.py
# Topologically Sorted Source Nodes: [attn_scores_1, matmul, attn_scores, attn_weights], Original ATen: [aten.masked_fill, aten.bmm, aten.mul, aten._softmax]
# Source node to ATen node mapping:
#   attn_scores => mul
#   attn_scores_1 => full_default, where
#   attn_weights => amax, exp, sub, sum_1
#   matmul => bmm
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %view_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.125), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%arg3_1, %full_default, %mul), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %bmm_default : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_default, %view_4), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%bmm_default, %view_default_1), kwargs = {})
triton_blo_fused__softmax_bmm_masked_fill_mul_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
import monkeypatch.fusion.triton_heuristics

@triton_heuristics.blockreduction(
    size_hints=[524288, 4096],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*i1', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_blo_fused__softmax_bmm_masked_fill_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'block_args': {'xnumel': 4096, 'xnumbl0': 4096, 'RBLOCK': 4096}}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK0: tl.constexpr, xnumbl0: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 524288
    pid = tl.program_id(0)
    x5 = (pid // (32*xnumbl0)) % 4
    x4 = (pid // xnumbl0) % 32
    pid0 = pid % xnumbl0
    xoffset0 = pid0*XBLOCK0
    x0 = xoffset0 + tl.arange(0, XBLOCK0)
    x6 = tl.arange(0, 64)
    xmask = tl.full([XBLOCK0], True, tl.int1)
    r8 = tl.arange(0, 64)
    tmp0 = tl.load(in_ptr0 + ((r8)[None, :] + (64*x0)[:, None] + 262144*x4 + 8388608*x5), (r8 < 64)[None, :] & (x0 < 4096)[:, None] & (x4 < 32) & (x5 < 4), eviction_policy='evict_last', other=0.0)
    # x0 = xindex % 4096
    # x4 = (xindex // 4096) % 32
    # x5 = (xindex // 131072)
    tmp10 = tl.full([XBLOCK0, 1], float("-inf"), tl.float32)
    # x6 = x6
    tmp18 = tl.full([XBLOCK0, 64], 0, tl.float32)
    tmp23 = tl.full([XBLOCK0, 1], 0, tl.float32)
    rbase = tl.arange(0, RBLOCK)
    for roffset in range(0, 4096, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < 4096
        r7 = rindex
        tmp1 = tl.load(in_ptr1 + ((r8)[:, None] + (64*r7)[None, :] + 262144*x4 + 8388608*x5), (r8 < 64)[:, None] & (r7 < 4096)[None, :] & (x4 < 32) & (x5 < 4), eviction_policy='evict_last', other=0.0)
        tmp3 = tl.full([XBLOCK0, RBLOCK], 0, tl.float32)
        tmp5 = tl.load(in_ptr2 + ((r7)[None, :] + (4096*x0)[:, None]), (r7 < 4096)[None, :] & (x0 < 4096)[:, None], eviction_policy='evict_last', other=0.0).to(tl.int1)
        tmp16 = tl.load(in_ptr3 + ((x6)[None, :] + (64*r7)[:, None] + 262144*x4 + 8388608*x5), (x6 < 64)[None, :] & (r7 < 4096)[:, None] & (x4 < 32) & (x5 < 4), eviction_policy='evict_last', other=0.0)
        tmp2 = tl.dot(tmp0, tmp1, input_precision='ieee')
        tmp4 = tmp3 + tmp2
        tmp3 = tmp4
        tmp6 = 0.125
        tmp7 = tmp3 * tmp6
        tmp8 = float("-inf")
        tmp9 = tl.where(tmp5, tmp8, tmp7)
        tmp11 = tl.max(tmp9, 1)
        tmp12 = tl.maximum(tmp10, tmp11[:, None])
        tmp13 = tmp9 - tmp12
        tmp14 = tl_math.exp2((tmp13) * 1.44269504)
        tmp15 = tmp14.to(tl.bfloat16)
        tmp17 = tl.dot(tmp15, tmp16, input_precision='ieee')
        tmp19 = tmp10 - tmp12
        tmp20 = tl_math.exp2((tmp19) * 1.44269504)
        tmp21 = tmp18 * tmp20
        tmp18 = tmp21
        tmp22 = tmp18 + tmp17
        tmp18 = tmp22
        tmp24 = tmp23 * tmp20
        tmp23 = tmp24
        tmp25 = tmp23 + tl.sum(tmp14, 1)[:, None]
        tmp23 = tmp25
        tmp10 = tmp12
    tmp26 = tmp18 / tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + ((x6)[None, :] + (64*x0)[:, None] + 262144*x4 + 8388608*x5), tmp26, (x6 < 64)[None, :] & (x0 < 4096)[:, None] & (x4 < 32) & (x5 < 4))
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 32, 4096, 64), (8388608, 262144, 64, 1))
    assert_size_stride(arg1_1, (4, 32, 4096, 64), (8388608, 262144, 64, 1))
    assert_size_stride(arg2_1, (4, 32, 4096, 64), (8388608, 262144, 64, 1))
    assert_size_stride(arg3_1, (4, 32, 4096, 4096), (0, 0, 4096, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((128, 4096, 64), (262144, 64, 1), torch.float32)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [attn_scores_1, matmul, attn_scores, attn_weights], Original ATen: [aten.masked_fill, aten.bmm, aten.mul, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_blo_fused__softmax_bmm_masked_fill_mul_0.run(buf4, arg0_1, arg1_1, arg3_1, arg2_1, 524288, grid=grid(524288), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (reinterpret_tensor(buf4, (4, 32, 4096, 64), (8388608, 262144, 64, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 32, 4096, 64), (8388608, 262144, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, 32, 4096, 64), (8388608, 262144, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4, 32, 4096, 64), (8388608, 262144, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((4, 32, 4096, 4096), (0, 0, 4096, 1), device='cuda:0', dtype=torch.bool)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
