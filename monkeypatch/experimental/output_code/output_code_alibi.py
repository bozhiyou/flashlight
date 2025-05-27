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


# kernel path: /work/06112/byou/ls6/torchinductor_cache/rt/crtvepckrphpi4dk4laawkshi6q44zgrl2nf7zatvt44qvphppn5.py
# Topologically Sorted Source Nodes: [matmul, attn_weight, sub, mul_2, attn_weight_1, attn_weight_2], Original ATen: [aten.bmm, aten.mul, aten.sub, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   attn_weight => mul
#   attn_weight_1 => add_1
#   attn_weight_2 => amax, exp, sub_1, sum_1
#   matmul => bmm
#   mul_2 => mul_2
#   sub => sub
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %view_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, 0.125), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_2, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_1, %sub), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_2), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add_1, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %bmm_default : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_default, %view_4), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%bmm_default, %view_default_1), kwargs = {})
triton_blo_fused__softmax_add_bmm_mul_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
from monkeypatch.experimental import block_reduction

@triton_heuristics.blockreduction(
    size_hints=[524288, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_blo_fused__softmax_add_bmm_mul_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'block_hints': {'xnumel': 512, 'xnumbl0': 512, 'RBLOCK': None, 'RBLOCK1': None}}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK0: tl.constexpr, xnumbl0: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr, RBLOCK1: tl.constexpr):
    xnumel = 524288
    pid = tl.program_id(0)
    x5 = (pid // (32*xnumbl0)) % 32
    x1 = (pid // xnumbl0) % 1024
    x4 = (pid // xnumbl0) % 32
    pid0 = pid % xnumbl0
    xoffset0 = pid0*XBLOCK0
    x0 = xoffset0 + tl.arange(0, XBLOCK0)
    x6 = tl.arange(0, 64)
    xmask = tl.full([XBLOCK0], True, tl.int1)
    # x4 = (xindex // 512) % 32
    tmp22 = tl.full([XBLOCK0, 1], float("-inf"), tl.float32)
    # x5 = (xindex // 16384)
    tmp40 = tl.full([XBLOCK0, 64], 0, tl.float32)
    _tmp48 = tl.full([XBLOCK0, RBLOCK], 0, tl.float32)
    rbase = tl.arange(0, RBLOCK)
    for roffset in range(0, 512, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < 512
        r7 = rindex
        # x0 = xindex % 512
        # x1 = (xindex // 512)
        tmp5 = tl.full([XBLOCK0, RBLOCK], 0, tl.float32)
        rbase1 = tl.arange(0, RBLOCK1)
        # for roffset1 in range(0, 64, RBLOCK1):
        #     rindex1 = roffset1 + rbase1
        rindex1 = rbase1
        rmask1 = rindex1 < 64
        r8 = rindex1
        tmp0 = tl.load(in_ptr0 + ((r8)[None, :] + (64*x0)[:, None] + 32768*x1), (r8 < 64)[None, :] & (x0 < 512)[:, None] & (x1 < 1024), eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + ((r8)[:, None] + (64*r7)[None, :] + 32768*x1), (r8 < 64)[:, None] & (r7 < 512)[None, :] & (x1 < 1024), eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0.to(tl.bfloat16)
        tmp3 = tmp1.to(tl.bfloat16)
        tmp4 = tl.dot(tmp2, tmp3)
        tmp6 = tmp5 + tmp4
        tmp5 = tmp6
        tmp35 = tl.load(in_ptr2 + ((x6)[None, :] + (64*r7)[:, None] + 32768*x1), (x6 < 64)[None, :] & (r7 < 512)[:, None] & (x1 < 1024), eviction_policy='evict_last', other=0.0)
        tmp7 = 0.125
        tmp8 = tmp5 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = 1 + x4
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 * tmp12
        tmp14 = 0.03125
        tmp15 = tmp13 * tmp14
        tmp16 = -tmp15
        tmp17 = libdevice.exp2(tmp16)
        tmp18 = (r7)[None, :] + (-x0)[:, None]
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 * tmp19
        tmp21 = tmp9 + tmp20
        tmp23 = triton_helpers.max2(tmp21, 1)
        tmp24 = triton_helpers.maximum(tmp22, tmp23[:, None])
        tmp25 = 1 + (x1 % 32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 * tmp12
        tmp28 = tmp27 * tmp14
        tmp29 = -tmp28
        tmp30 = libdevice.exp2(tmp29)
        tmp31 = tmp30 * tmp19
        tmp32 = tmp9 + tmp31
        tmp33 = tmp32 - tmp24
        tmp34 = tl_math.exp(tmp33)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp34.to(tl.float32)
        tmp38 = tmp36.to(tl.float32)
        tmp39 = tl.dot(tmp37, tmp38)
        tmp41 = tmp22 - tmp24
        tmp42 = tl_math.exp(tmp41)
        tmp43 = tmp40 * tmp42
        tmp40 = tmp43
        tmp44 = tmp40 + tmp39
        tmp40 = tmp44
        tmp45 = tmp21 - tmp24
        tmp46 = tl_math.exp(tmp45)
        tmp47 = tl.broadcast_to(tmp46, [XBLOCK0, RBLOCK])
        tmp49 = _tmp48 * tmp42
        _tmp48 = tmp49
        tmp50 = _tmp48 + tmp47
        _tmp48 = tl.where(rmask[None, :], tmp50, _tmp48)
        tmp22 = tmp24
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tmp51 = tmp40 / tmp48
    tl.store(in_out_ptr0 + ((x6)[None, :] + (64*x0)[:, None] + 32768*x1), tmp51, (x6 < 64)[None, :] & (x0 < 512)[:, None] & (x1 < 1024))
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 32, 512, 64), (1048576, 32768, 64, 1))
    assert_size_stride(arg1_1, (32, 32, 512, 64), (1048576, 32768, 64, 1))
    assert_size_stride(arg2_1, (32, 32, 512, 64), (1048576, 32768, 64, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((1024, 512, 64), (32768, 64, 1), torch.float32)
        buf4 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [matmul, attn_weight, sub, mul_2, attn_weight_1, attn_weight_2], Original ATen: [aten.bmm, aten.mul, aten.sub, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_blo_fused__softmax_add_bmm_mul_sub_0.run(buf4, arg0_1, arg1_1, arg2_1, 524288, grid=grid(524288), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
    return (reinterpret_tensor(buf4, (32, 32, 512, 64), (1048576, 32768, 64, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 32, 512, 64), (1048576, 32768, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((32, 32, 512, 64), (1048576, 32768, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((32, 32, 512, 64), (1048576, 32768, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    fn()
    # return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
    benchmark_compiled_module()
