
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
