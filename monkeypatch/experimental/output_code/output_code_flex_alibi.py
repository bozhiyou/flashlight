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
import torch._inductor.kernel.flex_attention

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


# kernel path: /work/06112/byou/ls6/torchinductor_cache/q7/cq7vnxcrlrlhvs7n4zidf6taflwdotkdbsogyczwtxatwmrfy4cs.py
# Topologically Sorted Source Nodes: [child], Original ATen: [aten.ones]
# Source node to ATen node mapping:
#   child => full
# Graph fragment:
#   %full : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1], 1), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_ones_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {1: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.full([1], 1, tl.int32)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /work/06112/byou/ls6/torchinductor_cache/v7/cv77ucso7icz2yombbrfow3tkkqx4uhleysip3fjw2gmyxomamat.py
# Topologically Sorted Source Nodes: [child_1], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   child_1 => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([1, 1, 1, 1], 0), kwargs = {dtype: torch.int32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i32', 1: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {1: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.full([1], 0, tl.int32)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: /work/06112/byou/ls6/torchinductor_cache/33/c33gygpkk2u6g6o7ilgklio2jwzadgec5jbza4aymwbxf6muzbyw.py
# Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
# Source node to ATen node mapping:
#   flex_attention => flex_attention
# Graph fragment:
#   %flex_attention : [num_users=1] = call_function[target=torch.ops.higher_order.flex_attention](args = (%arg0_1, %arg1_1, %arg2_1, %sdpa_score0, (%full, %full_default, None, None, %convert_element_type, %convert_element_type_1, None, None, 1073741824, 1073741824, %sdpa_mask0), 0.125, {ROWS_GUARANTEED_SAFE: False, PRESCALE_QK: False, OUTPUT_LOGSUMEXP: False}, (), ()), kwargs = {})
triton_tem_fused_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*i32', 5: '*i32', 6: '*fp32', 7: '*fp32', 8: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_2', 'backend_hash': 'A9C866B4A14FD3277824029365D703C2427B2E685E54EC9B3EF4ADC8D1EEAC1D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0):
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = False
    QK_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 1073741824
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 1073741824
    Q = arg_Q
    K = arg_K
    V = arg_V
    LSE = arg_LSE
    KV_NUM_BLKS = arg_KV_NUM_BLKS
    KV_IDX = arg_KV_IDX
    FULL_KV_NUM_BLKS = arg_FULL_KV_NUM_BLKS
    FULL_KV_IDX = arg_FULL_KV_IDX

    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # QK_HEAD_DIM: The dimension of the query and key embeddings
    # V_HEAD_DIM: The dimension of the value embeddings
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # GQA_SHARED_HEADS: number of query heads sharing one kv head in GQA setups.
    #
    # The following FULL_* and PARTIAL_* is defined in the block sparse mask grid, rather than the thread block grid.
    # KV_NUM_BLKS: The number of KV blocks (that may or may not require masking) for each query.
    # KV_IDX: The indices of KV blocks (that may or may not require masking) for each query.
    # FULL_KV_NUM_BLKS: The number of fully unmasked KV blocks (so we don't need masking) for each query.
    # FULL_KV_IDX: The indices of fully unmasked KV blocks (so we don't need masking) for each query.
    #
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad
    #
    # (Modifiable) Performance tuning options
    # BLOCK_M: The thread block size across the seqlen dim of Q.
    # BLOCK_N: Iterate over BLOCK_N across the seqlen dim of K/V in each thread block.

    # The below are kernel options that can be applied for certain score_mods,
    # or involve a numerics vs. perf tradeoff
    # PRESCALE_QK: Whether to pre-scale QK by 1/sqrt(d) and change of base. Has
    # about 20% more numerical error, but slightly faster.
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check

    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # Define strides of inputs
    stride_qz, stride_qh, stride_qm, stride_qk = 1048576, 32768, 64, 1
    stride_kz, stride_kh, stride_kn, stride_kk = 1048576, 32768, 64, 1
    stride_vz, stride_vh, stride_vn, stride_vk = 1048576, 32768, 64, 1

    Z = 32
    HQ = 32
    Q_LEN = 512
    KV_LEN = 512

    MATMUL_PRECISION = Q.dtype.element_ty

    q_start = tl.program_id(0)
    off_z = tl.program_id(1) // HQ
    off_hq = tl.program_id(1) % HQ
    off_hkv = off_hq // GQA_SHARED_HEADS
    off_g = off_hq % GQA_SHARED_HEADS

    q_offset = off_z * stride_qz + off_hq * stride_qh
    k_offset = off_z * stride_kz + off_hkv * stride_kh
    v_offset = off_z * stride_vz + off_hkv * stride_vh

    Q = Q + q_offset
    K = K + k_offset
    V = V + v_offset

    SPARSE_Z = 1
    SPARSE_HQ = 1

    sparse_idx_z = off_z % SPARSE_Z
    sparse_idx_hq = off_hq % SPARSE_HQ

    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    SPARSE_Q_BLOCK_CNT: tl.constexpr = tl.cdiv(Q_LEN, SPARSE_Q_BLOCK_SIZE)
    SPARSE_KV_BLOCK_CNT: tl.constexpr = tl.cdiv(KV_LEN, SPARSE_KV_BLOCK_SIZE)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_HEAD_DIM], dtype=tl.float32)

    offs_m = q_start * BLOCK_M + tl.arange(0, BLOCK_M)

    # KV_IDX and KV_NUM_BLKS are always contiguous.
    sparse_hz_offset = sparse_idx_z * SPARSE_HQ + sparse_idx_hq
    sparse_kv_num_blks_offset = sparse_hz_offset * SPARSE_Q_BLOCK_CNT + q_start // SPARSE_Q_MULTIPLE
    sparse_kv_idx_offset = sparse_hz_offset * SPARSE_Q_BLOCK_CNT * SPARSE_KV_BLOCK_CNT + (q_start // SPARSE_Q_MULTIPLE) * SPARSE_KV_BLOCK_CNT  # noqa: B950

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Q_LEN, QK_HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(q_start * BLOCK_M, 0),
        block_shape=(BLOCK_M, QK_HEAD_DIM),
        order=(1, 0)
    )

    # load q: it stays in SRAM throughout the inner loop.
    if IS_DIVISIBLE:
        q = tl.load(Q_block_ptr)
    else:
        # boundary check is not free, so we only do it when necessary.
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option = "zero")

    # ~~~~~~~~~~~~~~ normal blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We don't know anything "special" about these blocks, so we need to apply
    # both score_mod and mask_mod to it
    kv_indices = KV_IDX + sparse_kv_idx_offset
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    kv_num_blocks = tl.load(KV_NUM_BLKS + sparse_kv_num_blks_offset)
    block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(QK_HEAD_DIM, KV_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, kv_start),
        block_shape=(QK_HEAD_DIM, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(KV_LEN, V_HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, V_HEAD_DIM),
        order=(1, 0)
    )
    offs_n = kv_start + tl.arange(0, BLOCK_N)

    acc, l_i, m_i = forward_inner(
        arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
        q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
        acc, l_i, m_i,
        off_z, off_hq, offs_m[:, None], offs_n[None, :],
        kv_indices, kv_num_blocks,
        0, block_n_end,
        MATMUL_PRECISION,
        IS_FULL_BLOCKS=False,
    )

    # ~~~~~~~~~~~~~~ "full" blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # We know these blocks are guaranteed to be "full", so we don't need to
    # apply mask_mod to them - only score_mod
    if HAS_FULL_BLOCKS:
        # FULL_KV_IDX and FULL_KV_NUM_BLKS are always contiguous.
        kv_indices = FULL_KV_IDX + sparse_kv_idx_offset
        kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
        kv_num_blocks = tl.load(FULL_KV_NUM_BLKS + sparse_kv_num_blks_offset)
        block_n_end = tl.minimum(kv_num_blocks * SPARSE_KV_MULTIPLE, tl.maximum(tl.cdiv(KV_LEN, BLOCK_N), 1))

        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(QK_HEAD_DIM, KV_LEN),
            strides=(stride_kk, stride_kn),
            offsets=(0, kv_start),
            block_shape=(QK_HEAD_DIM, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(KV_LEN, V_HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(kv_start, 0),
            block_shape=(BLOCK_N, V_HEAD_DIM),
            order=(1, 0)
        )
        offs_n = kv_start + tl.arange(0, BLOCK_N)

        acc, l_i, m_i = forward_inner(
            arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
            q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
            acc, l_i, m_i,
            off_z, off_hq, offs_m[:, None], offs_n[None, :],
            kv_indices, kv_num_blocks,
            0, block_n_end,
            MATMUL_PRECISION,
            IS_FULL_BLOCKS=True,
        )


    # [Note] Handle fully masked out rows:
    # Li will be the sum(e^(-inf)) == 0.0 for masked out rows, mi will be -inf.
    # We set Li to 1.0 which will result in lse/out = 0.0 | after the log(li) + mi(0.0) step
    l_i = tl.where(l_i == 0.0, 1, l_i)

    acc = acc / l_i[:, None]
    idx_z = tl.program_id(1) // HQ
    idx_hq = tl.program_id(1) % HQ
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, V_HEAD_DIM)[None, :]

    mask = idx_m < Q_LEN
    # TODO generalize and add proper mask support
    xindex = idx_d + (64*idx_m) + (32768*idx_hq) + (1048576*idx_z)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)

    # TODO dont want to write this if we dont require grad
    if OUTPUT_LOGSUMEXP:
        off_hz = tl.program_id(1)
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        if IS_DIVISIBLE:
            tl.store(l_ptrs, lse)
        else:
            tl.store(l_ptrs, lse, mask=offs_m < Q_LEN)

@triton.jit
def forward_inner(
    arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets used as inputs to score_mod & mask_mod
    # of size [BLOCK_M, BLOCK_N] or scalar.
    off_z, off_h, offs_m, offs_n,
    # blocksparse data
    kv_indices, kv_num_blocks,
    # start kv and end kv block
    block_n_start, block_n_end,
    MATMUL_PRECISION,
    IS_FULL_BLOCKS,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = False
    QK_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 1073741824
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 1073741824


    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)
    RCP_LN2: tl.constexpr = 1.44269504

    if PRESCALE_QK:
        q = (q * SM_SCALE * RCP_LN2).to(MATMUL_PRECISION)

    # loop over k, v and update accumulator until block_n_end
    for start_n in range(block_n_start, block_n_end):
        if IS_DIVISIBLE:
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS,
            )
        else:
            # Benchmark shows even we applied mod & mask to each block for non divisible seqlen,
            # it's on par or slightly faster than only applying to the last block in fwd.
            # However, we choose different strategy for bwd, where we only apply mod & mask
            # to the last block because it's faster a lot.
            acc, l_i, m_i = forward_block_mn(
                arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
                q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
                # accumulated values
                acc, l_i, m_i,
                # Offsets
                off_z, off_h, offs_m, offs_n,
                MATMUL_PRECISION, RCP_LN2,
                IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=True,
            )

        # update pointers
        offset = get_offset_for_next_block(
            start_n, kv_indices, kv_num_blocks,
            SPARSE_KV_BLOCK_SIZE, SPARSE_KV_MULTIPLE, BLOCK_N
        )

        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, offset))

        offs_n = offs_n + offset

    return acc, l_i, m_i


@triton.jit
def get_offset_for_next_block(loop_iter, col_indices, total_blocks, SPARSE_BLOCK, SPARSE_BLOCK_MULTIPLE, BLOCK):
    cur_block_idx = loop_iter // SPARSE_BLOCK_MULTIPLE
    cur_block = tl.load(col_indices + cur_block_idx, eviction_policy="evict_last")
    next_block = tl.load(col_indices + cur_block_idx + 1, eviction_policy="evict_last", mask=cur_block_idx + 1 < total_blocks)
    needs_jump = (loop_iter + 1) % SPARSE_BLOCK_MULTIPLE == 0
    jump_to_block = (next_block - cur_block ) * SPARSE_BLOCK - (SPARSE_BLOCK_MULTIPLE - 1) * BLOCK

    offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK
    return offset

@triton.jit
def forward_block_mn(
    arg_Q, arg_K, arg_V, arg_LSE, arg_KV_NUM_BLKS, arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, out_ptr0,
    q, K_block_ptr, V_block_ptr, Q_LEN, KV_LEN,
    # accumulated values
    acc, l_i, m_i,
    # Offsets
    off_z, off_h, offs_m, offs_n,
    MATMUL_PRECISION, RCP_LN2,
    IS_FULL_BLOCKS, CHECK_BLOCK_BOUNDARY=False,
):
    # Redefines all kernel parameters (BLOCK_M, etc.) so we don't need to plumb them all through
    ROWS_GUARANTEED_SAFE : tl.constexpr = False
    PRESCALE_QK : tl.constexpr = False
    OUTPUT_LOGSUMEXP : tl.constexpr = False
    FLOAT32_PRECISION : tl.constexpr = 'ieee'
    IS_DIVISIBLE : tl.constexpr = True
    SM_SCALE : tl.constexpr = 0.125
    GQA_SHARED_HEADS : tl.constexpr = 1
    HAS_FULL_BLOCKS : tl.constexpr = False
    QK_HEAD_DIM : tl.constexpr = 64
    V_HEAD_DIM : tl.constexpr = 64
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 64
    SPARSE_Q_BLOCK_SIZE : tl.constexpr = 1073741824
    SPARSE_KV_BLOCK_SIZE : tl.constexpr = 1073741824


    # -- load k --
    if IS_DIVISIBLE:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option = "zero")
    # -- compute qk ---
    qk = tl.dot(q, k, input_precision=FLOAT32_PRECISION) # TODO: use cuda matmul when q_len <= 2.
    if not PRESCALE_QK:
        qk *= SM_SCALE
    # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
    if CHECK_BLOCK_BOUNDARY:
        # If this is the last block of a non divisible seqlen, we still need to load [BLOCK_M, BLOCK_N] elements,
        # which is larger than the actual number of elements. To avoid access memory out of bound,
        # we need to mask out the elements that are out of Q_LEN & KV_LEN.
        m = offs_m % Q_LEN
        n = offs_n % KV_LEN
    else:
        m = offs_m
        n = offs_n

    tmp0 = (qk).to(tl.float32)
    tmp1 = (n) - (m)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.full([1], 1, tl.int32)
    tmp4 = (off_h) + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 8.0
    tmp7 = tmp5 * tmp6
    tmp8 = 0.03125
    tmp9 = tmp7 * tmp8
    tmp10 = -tmp9
    tmp11 = libdevice.exp2(tmp10)
    tmp12 = tmp2 * tmp11
    tmp13 = tmp0 + tmp12
    post_mod_scores = tmp13


    if CHECK_BLOCK_BOUNDARY:
        # Mask out the elements that are out of the KV_LEN for non divisible seqlen.
        post_mod_scores = tl.where(offs_n < KV_LEN, post_mod_scores, float("-inf"))

    if not IS_FULL_BLOCKS:
        tmp14 = tl.full([1], True, tl.int1)
        mask_mod_output = tmp14


        if CHECK_BLOCK_BOUNDARY:
            mask_mod_output = tl.where(offs_n < KV_LEN, mask_mod_output, float("-inf"))
        # apply mask for partially unmasked blocks
        post_mod_scores = tl.where(mask_mod_output, post_mod_scores, float("-inf"))

    # TODO: In the case that score_mod is linear, this can be LICMed
    if not PRESCALE_QK:
        post_mod_scores *= RCP_LN2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -- compute scaling constant ---
    m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
    if not ROWS_GUARANTEED_SAFE:
        masked_out_rows = (m_ij == float("-inf"))
        m_ij_masked = tl.where(masked_out_rows, 0, m_ij)
    else:
        m_ij_masked = m_ij

    alpha = tl.math.exp2(m_i - m_ij_masked)
    p = tl.math.exp2(post_mod_scores - m_ij_masked[:, None])

    # NB: l_i update is pulled up here since it's a bit faster
    # NB: For headdim=256, it's faster to move it back down to after m_i =
    # m_ij
    l_i = l_i * alpha + tl.sum(p, 1)
    # # -- scale and update acc --
    acc = acc * alpha[:, None]

    if IS_DIVISIBLE:
        v = tl.load(V_block_ptr)
    else:
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option = "zero")
    acc = tl.dot(p.to(MATMUL_PRECISION), v, acc, input_precision=FLOAT32_PRECISION)

    # -- update m_i
    m_i = m_ij

    return acc, l_i, m_i
''', device_str='cuda')
meta0 = {'ROWS_GUARANTEED_SAFE': False, 'PRESCALE_QK': False, 'OUTPUT_LOGSUMEXP': False, 'FLOAT32_PRECISION': "'ieee'", 'IS_DIVISIBLE': True, 'SM_SCALE': 0.125, 'GQA_SHARED_HEADS': 1, 'HAS_FULL_BLOCKS': False, 'QK_HEAD_DIM': 64, 'V_HEAD_DIM': 64, 'BLOCK_M': 128, 'BLOCK_N': 64, 'SPARSE_Q_BLOCK_SIZE': 1073741824, 'SPARSE_KV_BLOCK_SIZE': 1073741824}


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
        buf2 = empty_strided_cuda((1, 1, 1), (1, 1, 1), torch.int32)
        # Topologically Sorted Source Nodes: [child], Original ATen: [aten.ones]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_0.run(buf2, 1, grid=grid(1), stream=stream0)
        buf3 = empty_strided_cuda((1, 1, 1, 1), (1, 1, 1, 1), torch.int32)
        # Topologically Sorted Source Nodes: [child_1], Original ATen: [aten.zeros]
        triton_poi_fused_zeros_1.run(buf3, 1, grid=grid(1), stream=stream0)
        buf6 = empty_strided_cuda((32, 32, 512), (16384, 512, 1), torch.float32)
        buf7 = empty_strided_cuda((0, ), (1, ), torch.float32)
        buf8 = empty_strided_cuda((0, ), (1, ), torch.float32)
        buf9 = empty_strided_cuda((32, 32, 512, 64), (1048576, 32768, 64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [flex_attention], Original ATen: []
        triton_tem_fused_2.run(arg0_1, arg1_1, arg2_1, buf6, buf2, buf3, buf7, buf8, buf9, grid=torch._inductor.kernel.flex_attention.flex_attention_grid(32, 32, 512, 64, meta0), stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del buf2
        del buf3
        del buf6
        del buf7
        del buf8
    return (buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 32, 512, 64), (1048576, 32768, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((32, 32, 512, 64), (1048576, 32768, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((32, 32, 512, 64), (1048576, 32768, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    # return print_performance(fn, times=times, repeat=repeat)
    fn()


if __name__ == "__main__":
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
    benchmark_compiled_module()
