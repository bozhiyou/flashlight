
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_blo_fused__softmax_bmm_div_mul_tanh_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'block_args': {'xnumel': 4096, 'xnumbl0': 4096, 'RBLOCK': 4096}}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK0: tl.constexpr, xnumbl0: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
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
    tmp12 = tl.full([XBLOCK0, 1], float("-inf"), tl.float32)
    # x6 = x6
    tmp20 = tl.full([XBLOCK0, 64], 0, tl.float32)
    tmp25 = tl.full([XBLOCK0, 1], 0, tl.float32)
    rbase = tl.arange(0, RBLOCK)
    for roffset in range(0, 4096, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < 4096
        r7 = rindex
        tmp1 = tl.load(in_ptr1 + ((r8)[:, None] + (64*r7)[None, :] + 262144*x4 + 8388608*x5), (r8 < 64)[:, None] & (r7 < 4096)[None, :] & (x4 < 32) & (x5 < 4), eviction_policy='evict_last', other=0.0)
        tmp3 = tl.full([XBLOCK0, RBLOCK], 0, tl.float32)
        tmp18 = tl.load(in_ptr2 + ((x6)[None, :] + (64*r7)[:, None] + 262144*x4 + 8388608*x5), (x6 < 64)[None, :] & (r7 < 4096)[:, None] & (x4 < 32) & (x5 < 4), eviction_policy='evict_last', other=0.0)
        tmp2 = tl.dot(tmp0, tmp1, input_precision='ieee')
        tmp4 = tmp3 + tmp2
        tmp3 = tmp4
        tmp5 = 0.125
        tmp6 = tmp3 * tmp5
        tmp7 = 0.03333333333333333
        tmp8 = tmp6 * tmp7
        tmp9 = libdevice.tanh(tmp8)
        tmp10 = 30.0
        tmp11 = tmp9 * tmp10
        tmp13 = tl.max(tmp11, 1)
        tmp14 = tl.maximum(tmp12, tmp13[:, None])
        tmp15 = tmp11 - tmp14
        tmp16 = tl_math.exp2((tmp15) * 1.44269504)
        tmp17 = tmp16.to(tl.bfloat16)
        tmp19 = tl.dot(tmp17, tmp18, input_precision='ieee')
        tmp21 = tmp12 - tmp14
        tmp22 = tl_math.exp2((tmp21) * 1.44269504)
        tmp23 = tmp20 * tmp22
        tmp20 = tmp23
        tmp24 = tmp20 + tmp19
        tmp20 = tmp24
        tmp26 = tmp25 * tmp22
        tmp25 = tmp26
        tmp27 = tmp25 + tl.sum(tmp16, 1)[:, None]
        tmp25 = tmp27
        tmp12 = tmp14
    tmp28 = tmp20 / tmp25
    tl.debug_barrier()
    tl.store(in_out_ptr0 + ((x6)[None, :] + (64*x0)[:, None] + 262144*x4 + 8388608*x5), tmp28, (x6 < 64)[None, :] & (x0 < 4096)[:, None] & (x4 < 32) & (x5 < 4))
