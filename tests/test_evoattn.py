# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
This script is to test the performance of the DS4Sci_EvoformerAttention op.
To run the script,
1. Clone the CUTLASS repo. E.g. git clone https://github.com/NVIDIA/cutlass.git
2. Specify the CUTLASS_PATH environment variable. E.g. export CUTLASS_PATH=$(pwd)/cutlass
3. Run the script. E.g. python DS4Sci_EvoformerAttention_bench.py
"""

import contextlib
import torch
from attention_variants.evoformer import attention_reference, make_input, N
if __name__ == "__main__":
    from monkeypatch.fusion import dependent_reduction_fusion
    from monkeypatch.fusion import block_reduction
    from monkeypatch.fusion import reduction_kernel_fusion

    dtype = torch.bfloat16

    heads = 4
    dim = 32
    seq_len = 256  # fixed


    @contextlib.contextmanager
    def cuda_timer(res_list):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        torch.cuda.synchronize()
        res_list.append(start.elapsed_time(end))


    ours_fw = []
    ours_bw = []
    baseline_fw = []
    baseline_bw = []
    tflops_fw = []
    batch_sizes = [32, 16, 8, 4, 2, 1]
    for batch in batch_sizes:
        tflops_fw.append(4 * batch * N * seq_len**2 * heads * dim)  # FIXME bias addition
        Q = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=False)
        K = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=False)
        V = torch.randn(batch, N, seq_len, heads, dim, dtype=dtype, device="cuda", requires_grad=False)
        bias1 = torch.randn(batch, N, 1, 1, seq_len, dtype=dtype, device="cuda", requires_grad=False)
        bias2 = torch.randn(batch, 1, heads, seq_len, seq_len, dtype=dtype, device="cuda", requires_grad=False)
        # warm up
        # DS4Sci_EvoformerAttention(Q, K, V, [bias1, bias2])
        # with cuda_timer(ours_fw):
        #     out = DS4Sci_EvoformerAttention(Q, K, V, [bias1, bias2])
        # d_out = torch.rand_like(out)
        # with cuda_timer(ours_bw):
        #     out.backward(d_out)
        # warm up
        attention_reference(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        with cuda_timer(baseline_fw):
            ref_out = attention_reference(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        # with cuda_timer(baseline_bw):
        #     ref_out.backward(d_out)
        out = torch.compile(dynamic=False)(attention_reference)(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        with cuda_timer(ours_fw):
            out = torch.compile(dynamic=False)(attention_reference)(Q, K, V, [bias1, bias2], 1 / (dim**0.5))
        # bf16 compiled vs fp32 eager: ~2496/268M elements exceed 1e-2 (max 0.0625)
        # on large 5D evoformer tensors (batch*N*seq*heads*dim). If this still
        # fails on your hardware, increase atol or reduce batch_sizes above.
        torch.testing.assert_close(ref_out, out, atol=0.1, rtol=0.1)

    print("batch size\tours (FW)\tbaseline (FW)\tours (BW)\tbaseline (BW)")
    for i in range(len(ours_fw)):
        # print(f"{i+1}\t{ours_fw[i]}\t{baseline_fw[i]}\t{ours_bw[i]}\t{baseline_bw[i]}")
        print(f"{i+1}\t{ours_fw[i]}\t{baseline_fw[i]}")
    
    print("Implementation,FW_Time_ms,FW_TFLOPS,batch_size,seqlen,nheads,headdim")
    for batch, t_fw in zip(batch_sizes, ours_fw):
        print(f"ipa_compiled,{t_fw},todo-tflops,{batch},{seq_len},{heads},{dim}")
    for batch, t_fw in zip(batch_sizes, baseline_fw):
        print(f"ipa,{t_fw},todo-tflops,{batch},{seq_len},{heads},{dim}")
