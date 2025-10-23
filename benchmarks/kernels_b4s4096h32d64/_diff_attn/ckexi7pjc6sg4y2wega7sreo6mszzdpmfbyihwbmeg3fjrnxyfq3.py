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


# kernel path: /tmp/torchinductor_byou/4m/c4mefejtziyt6jk537pbuwj6l4pvg7yjdih6i2rat2injuwirf5m.py
# Topologically Sorted Source Nodes: [matmul, attn_weights0, softmax, matmul_1, attn_weights1, softmax_1, mul_2, attn], Original ATen: [aten.bmm, aten.mul, aten._softmax, aten.sub]
# Source node to ATen node mapping:
#   attn => sub_2
#   attn_weights0 => mul
#   attn_weights1 => mul_1
#   matmul => bmm
#   matmul_1 => bmm_1
#   mul_2 => mul_2
#   softmax => amax, exp, sub, sum_1
#   softmax_1 => amax_1, exp_1, sub_1, sum_2
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %view_1), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.125), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %bmm_default_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_default_2, %view_7), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_3, %view_4), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, 0.125), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_1, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %bmm_default : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_default, %view_10), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [-1], True), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, 0.2), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_8, %mul_2), kwargs = {})
triton_blo_fused__softmax_bmm_mul_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
import monkeypatch.fusion.triton_heuristics

@triton_heuristics.blockreduction(
    size_hints=[262144, 4096],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_blo_fused__softmax_bmm_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 8, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'block_args': {'xnumel': 4096, 'xnumbl0': 4096, 'RBLOCK': 4096}}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK0: tl.constexpr, xnumbl0: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    xnumel = 262144
    pid = tl.program_id(0)
    x5 = (pid // (16*xnumbl0)) % 4
    x4 = (pid // xnumbl0) % 16
    pid0 = pid % xnumbl0
    xoffset0 = pid0*XBLOCK0
    x0 = xoffset0 + tl.arange(0, XBLOCK0)
    x6 = tl.arange(0, 128)
    xmask = tl.full([XBLOCK0], True, tl.int1)
    r9 = tl.arange(0, 64)
    r10 = tl.arange(0, 64)
    tmp0 = tl.load(in_ptr0 + ((r9)[None, :] + (64*x0)[:, None] + 262144*x4 + 8388608*x5), (r9 < 64)[None, :] & (x0 < 4096)[:, None] & (x4 < 16) & (x5 < 4), eviction_policy='evict_last', other=0.0)
    tmp23 = tl.load(in_ptr0 + (4194304 + (r10)[None, :] + (64*x0)[:, None] + 262144*x4 + 8388608*x5), (r10 < 64)[None, :] & (x0 < 4096)[:, None] & (x4 < 16) & (x5 < 4), eviction_policy='evict_last', other=0.0)
    # x0 = xindex % 4096
    # x4 = (xindex // 4096) % 16
    # x5 = (xindex // 65536)
    tmp7 = tl.full([XBLOCK0, 1], float("-inf"), tl.float32)
    # x6 = x6
    tmp15 = tl.full([XBLOCK0, 128], 0, tl.float32)
    tmp20 = tl.full([XBLOCK0, 1], 0, tl.float32)
    tmp29 = tl.full([XBLOCK0, 1], float("-inf"), tl.float32)
    tmp36 = tl.full([XBLOCK0, 128], 0, tl.float32)
    tmp41 = tl.full([XBLOCK0, 1], 0, tl.float32)
    rbase = tl.arange(0, RBLOCK)
    for roffset in range(0, 4096, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < 4096
        r8 = rindex
        tmp1 = tl.load(in_ptr1 + ((r9)[:, None] + (64*r8)[None, :] + 262144*x4 + 8388608*x5), (r9 < 64)[:, None] & (r8 < 4096)[None, :] & (x4 < 16) & (x5 < 4), eviction_policy='evict_last', other=0.0)
        tmp3 = tl.full([XBLOCK0, RBLOCK], 0, tl.float32)
        tmp13 = tl.load(in_ptr2 + ((x6)[None, :] + (128*r8)[:, None] + 524288*x4 + 8388608*x5), (x6 < 128)[None, :] & (r8 < 4096)[:, None] & (x4 < 16) & (x5 < 4), eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr1 + (4194304 + (r10)[:, None] + (64*r8)[None, :] + 262144*x4 + 8388608*x5), (r10 < 64)[:, None] & (r8 < 4096)[None, :] & (x4 < 16) & (x5 < 4), eviction_policy='evict_last', other=0.0)
        tmp26 = tl.full([XBLOCK0, RBLOCK], 0, tl.float32)
        tmp2 = tl.dot(tmp0, tmp1, input_precision='ieee')
        tmp4 = tmp3 + tmp2
        tmp3 = tmp4
        tmp5 = 0.125
        tmp6 = tmp3 * tmp5
        tmp8 = tl.max(tmp6, 1)
        tmp9 = tl.maximum(tmp7, tmp8[:, None])
        tmp10 = tmp6 - tmp9
        tmp11 = tl_math.exp2((tmp10) * 1.44269504)
        tmp12 = tmp11.to(tl.bfloat16)
        tmp14 = tl.dot(tmp12, tmp13, input_precision='ieee')
        tmp16 = tmp7 - tmp9
        tmp17 = tl_math.exp2((tmp16) * 1.44269504)
        tmp18 = tmp15 * tmp17
        tmp15 = tmp18
        tmp19 = tmp15 + tmp14
        tmp15 = tmp19
        tmp21 = tmp20 * tmp17
        tmp20 = tmp21
        tmp22 = tmp20 + tl.sum(tmp11, 1)[:, None]
        tmp20 = tmp22
        tmp25 = tl.dot(tmp23, tmp24, input_precision='ieee')
        tmp27 = tmp26 + tmp25
        tmp26 = tmp27
        tmp28 = tmp26 * tmp5
        tmp30 = tl.max(tmp28, 1)
        tmp31 = tl.maximum(tmp29, tmp30[:, None])
        tmp32 = tmp28 - tmp31
        tmp33 = tl_math.exp2((tmp32) * 1.44269504)
        tmp34 = tmp33.to(tl.bfloat16)
        tmp35 = tl.dot(tmp34, tmp13, input_precision='ieee')
        tmp37 = tmp29 - tmp31
        tmp38 = tl_math.exp2((tmp37) * 1.44269504)
        tmp39 = tmp36 * tmp38
        tmp36 = tmp39
        tmp40 = tmp36 + tmp35
        tmp36 = tmp40
        tmp42 = tmp41 * tmp38
        tmp41 = tmp42
        tmp43 = tmp41 + tl.sum(tmp33, 1)[:, None]
        tmp41 = tmp43
        tmp7 = tmp9
        tmp29 = tmp31
    tmp44 = tmp15 / tmp20
    tmp45 = tmp36 / tmp41
    tmp46 = 0.2
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 - tmp47
    tl.debug_barrier()
    tl.store(in_out_ptr0 + ((x6)[None, :] + (128*x0)[:, None] + 524288*x4 + 8388608*x5), tmp48, (x6 < 128)[None, :] & (x0 < 4096)[:, None] & (x4 < 16) & (x5 < 4))
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 32, 4096, 64), (8388608, 262144, 64, 1))
    assert_size_stride(arg1_1, (4, 32, 4096, 64), (8388608, 262144, 64, 1))
    assert_size_stride(arg2_1, (4, 16, 4096, 128), (8388608, 524288, 128, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((64, 4096, 128), (524288, 128, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 16, 4096, 128), (8388608, 524288, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [matmul, attn_weights0, softmax, matmul_1, attn_weights1, softmax_1, mul_2, attn], Original ATen: [aten.bmm, aten.mul, aten._softmax, aten.sub]
        stream0 = get_raw_stream(0)
        triton_blo_fused__softmax_bmm_mul_sub_0.run(buf8, arg0_1, arg1_1, arg2_1, 262144, grid=grid(262144), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del buf2
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 32, 4096, 64), (8388608, 262144, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((4, 32, 4096, 64), (8388608, 262144, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((4, 16, 4096, 128), (8388608, 524288, 128, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
