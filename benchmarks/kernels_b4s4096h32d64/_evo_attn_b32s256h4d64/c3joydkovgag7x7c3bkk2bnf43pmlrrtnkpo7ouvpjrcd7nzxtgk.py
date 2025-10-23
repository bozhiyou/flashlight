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


# kernel path: /tmp/torchinductor_byou/n4/cn4meldb4eskroyocvtzzyb4bnms5hdl47oq7ybh3shlm7vac2yu.py
# Topologically Sorted Source Nodes: [matmul, a, a_1, a_2, a_3], Original ATen: [aten.bmm, aten.mul, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   a => mul
#   a_1 => add
#   a_2 => add_1
#   a_3 => amax, exp, sub, sum_1
#   matmul => bmm
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %view_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.125), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %arg4_1), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_1, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %bmm_default : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_default, %view_4), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%bmm_default, %view_default_1), kwargs = {})
triton_blo_fused__softmax_add_bmm_mul_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
import monkeypatch.fusion.triton_heuristics

@triton_heuristics.blockreduction(
    size_hints=[8388608, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_blo_fused__softmax_add_bmm_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'block_args': {'xnumel': 256, 'xnumbl0': 256, 'RBLOCK': 256}}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK0: tl.constexpr, xnumbl0: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 8388608
    pid = tl.program_id(0)
    x7 = (pid // (1024*xnumbl0)) % 32
    x6 = (pid // (4*xnumbl0)) % 256
    x4 = (pid // xnumbl0) % 4
    pid0 = pid % xnumbl0
    xoffset0 = pid0*XBLOCK0
    x0 = xoffset0 + tl.arange(0, XBLOCK0)
    x8 = tl.arange(0, 64)
    xmask = tl.full([XBLOCK0], True, tl.int1)
    r10 = tl.arange(0, 64)
    tmp0 = tl.load(in_ptr0 + ((r10)[None, :] + 64*x4 + (256*x0)[:, None] + 65536*x6 + 16777216*x7), (r10 < 64)[None, :] & (x4 < 4) & (x0 < 256)[:, None] & (x6 < 256) & (x7 < 32), eviction_policy='evict_last', other=0.0)
    # x0 = xindex % 256
    # x4 = (xindex // 256) % 4
    # x6 = (xindex // 1024) % 256
    # x7 = (xindex // 262144)
    tmp13 = tl.full([XBLOCK0, 1], float("-inf"), tl.float32)
    tmp18 = tl.full([XBLOCK0, 1], 0, tl.float32)
    # x8 = x8
    tmp26 = tl.full([XBLOCK0, 64], 0, tl.float32)
    rbase = tl.arange(0, RBLOCK)
    for roffset in range(0, 256, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < 256
        r9 = rindex
        tmp1 = tl.load(in_ptr1 + ((r10)[:, None] + 64*x4 + (256*r9)[None, :] + 65536*x6 + 16777216*x7), (r10 < 64)[:, None] & (x4 < 4) & (r9 < 256)[None, :] & (x6 < 256) & (x7 < 32), eviction_policy='evict_last', other=0.0)
        tmp3 = tl.full([XBLOCK0, RBLOCK], 0, tl.float32)
        tmp7 = tl.load(in_ptr2 + (r9 + (256*x6) + (65536*x7)), (r9 < 256) & (x6 < 256) & (x7 < 32), eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr3 + ((r9)[None, :] + (256*x0)[:, None] + 65536*x4 + 262144*x7), (r9 < 256)[None, :] & (x0 < 256)[:, None] & (x4 < 4) & (x7 < 32), eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr4 + ((x8)[None, :] + 64*x4 + (256*r9)[:, None] + 65536*x6 + 16777216*x7), (x8 < 64)[None, :] & (x4 < 4) & (r9 < 256)[:, None] & (x6 < 256) & (x7 < 32), eviction_policy='evict_last', other=0.0)
        tmp2 = tl.dot(tmp0, tmp1, input_precision='ieee')
        tmp4 = tmp3 + tmp2
        tmp3 = tmp4
        tmp5 = 0.125
        tmp6 = tmp3 * tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 + tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 + tmp11
        tmp14 = tl.max(tmp12, 1)
        tmp15 = tl.maximum(tmp13, tmp14[:, None])
        tmp16 = tmp12 - tmp15
        tmp17 = tl_math.exp2((tmp16) * 1.44269504)
        tmp19 = tmp13 - tmp15
        tmp20 = tl_math.exp2((tmp19) * 1.44269504)
        tmp21 = tmp18 * tmp20
        tmp18 = tmp21
        tmp22 = tmp18 + tl.sum(tmp17, 1)[:, None]
        tmp18 = tmp22
        tmp23 = tmp17.to(tl.bfloat16)
        tmp25 = tl.dot(tmp23, tmp24, input_precision='ieee')
        tmp27 = tmp26 * tmp20
        tmp26 = tmp27
        tmp28 = tmp26 + tmp25
        tmp26 = tmp28
        tmp13 = tmp15
    tmp29 = tmp26 / tmp18
    tl.debug_barrier()
    tl.store(in_out_ptr0 + ((x8)[None, :] + (64*x0)[:, None] + 16384*x4 + 65536*x6 + 16777216*x7), tmp29, (x8 < 64)[None, :] & (x0 < 256)[:, None] & (x4 < 4) & (x6 < 256) & (x7 < 32))
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 256, 256, 4, 64), (16777216, 65536, 256, 64, 1))
    assert_size_stride(arg1_1, (32, 256, 256, 4, 64), (16777216, 65536, 256, 64, 1))
    assert_size_stride(arg2_1, (32, 256, 256, 4, 64), (16777216, 65536, 256, 64, 1))
    assert_size_stride(arg3_1, (32, 256, 1, 1, 256), (65536, 256, 256, 256, 1))
    assert_size_stride(arg4_1, (32, 1, 4, 256, 256), (262144, 262144, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((32768, 256, 64), (16384, 64, 1), torch.float32)
        buf4 = empty_strided_cuda((32768, 256, 64), (16384, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul, a, a_1, a_2, a_3], Original ATen: [aten.bmm, aten.mul, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_blo_fused__softmax_add_bmm_mul_0.run(buf4, arg0_1, arg1_1, arg3_1, arg4_1, arg2_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del buf2
    return (reinterpret_tensor(buf4, (32, 256, 256, 4, 64), (16777216, 65536, 64, 16384, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 256, 256, 4, 64), (16777216, 65536, 256, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((32, 256, 256, 4, 64), (16777216, 65536, 256, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((32, 256, 256, 4, 64), (16777216, 65536, 256, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((32, 256, 1, 1, 256), (65536, 256, 256, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((32, 1, 4, 256, 256), (262144, 262144, 65536, 256, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
