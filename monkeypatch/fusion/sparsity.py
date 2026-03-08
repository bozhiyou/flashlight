from .. import _monkey as monkey

from typing import List, Any, Sequence, Callable

import collections
import contextlib
import dataclasses
import functools
import itertools

import torch
import torch._inductor.config
from torch._inductor import ir, lowering as L
from torch._inductor.virtualized import V
from torch._inductor.kernel.bmm import tuned_bmm
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.loop_body import LoopBody
from torch._inductor.codegen.common import SizeArg
from torch._inductor.codegen.triton import TritonScheduling, TritonKernel, TritonKernelOverrides
from torch._inductor.codegen.simd import IterationRangesRoot, IterationRangesEntry, EnableReduction, DisableReduction
from torch._inductor.optimize_indexing import indexing_dtype_strength_reduction
from torch._inductor.utils import ceildiv, sympy_index_symbol, sympy_product, IndentedBuffer, sympy_subs, VarRanges
from torch._inductor.sizevars import SimplifyIndexing


import sympy
from sympy import Expr
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._ordered_set import OrderedSet

from torch._inductor.virtualized import ops, OpsWrapper, OpsValue

# add 'fusion' to comma-separated TORCH_LOG to enable
fusion_log = torch._logging.getArtifactLogger('torch._inductor', "fusion")  # fusion_log.debug(...)


# Unregister original torch.where lowering
# torch.ops.aten.bmm: torch._ops.OpOverloadPacket
for overload in torch.ops.aten.where.overloads():
    other_fn = getattr(torch.ops.aten.where, overload)
    L.lowerings.pop(other_fn)

@dataclasses.dataclass
class ConditionalPointwise(ir.Pointwise):
    def make_loader(self):
        # Make zero-element loops into a no-op
        if self.is_zero_elements():
            # TODO @bozhiyou torch.where(torch.tensor([], dtype=bool), a, b) returns torch.tensor([], dtype=<promoted_dtype(a, b)>)
            return functools.partial(ir.nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def store_output(self, output_name, indexer, vars):
        loader = self.make_loader()
        return ops.store(output_name, indexer(vars), loader(vars))



def make_pointwise(
    fn,
    override_return_dtype=None,
    override_device=None,
    override_fn_when_input_bool=None,
    override_fn_when_gpu_float64=None,
    allow_alpha=False,
    triton_fallback=None,
):
    def inner(*inputs: List[ir.TensorBox], alpha=None):
        if triton_fallback is not None and any(map(ir.is_triton, inputs)):
            assert not allow_alpha  # not implemented
            return triton_fallback(*inputs)

        inputs = L.promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = L.mul(inputs[-1], alpha)
        else:
            assert alpha is None
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()
        is_gpu_device = L.is_gpu(L.decode_device(inputs[0].get_device()).type)

        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(
                other.get_size()
            ), f"ndim mismatch {fn} {ranges} {other.get_size()}"

        # in tracing, we will annotate pointwise nodes that correspond to the output of
        # a pointwise node that would have been run in eager. intermediary pointwise nodes
        # during decompositions are not annotated.
        emulate_precision_casts = (
            V.graph is not None
            and getattr(V.graph, "current_node", None) is not None
            and V.graph.current_node.meta is not None
            and V.graph.current_node.meta.get("low_precision_pointwise_barrier", False)
            and dtype in (torch.bfloat16, torch.float16)
        )

        def inner_fn(index):
            ops.bozhi(1996)
            assert len(index) == len(ranges), f"wrong ndim {index} {ranges}"
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif (
                override_fn_when_gpu_float64
                and is_gpu_device
                and dtype == torch.float64
            ):
                return override_fn_when_gpu_float64(*[load(index) for load in loaders])
            else:
                inputs_loaded = []
                for load in loaders:
                    out = load(index)
                    if emulate_precision_casts:
                        downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                        out = ops.to_dtype(downcast, dtype)
                    inputs_loaded.append(out)

                out = fn(*inputs_loaded)
                if emulate_precision_casts:
                    # fp16/bf16 kernels are computed in fp32. Casting down to fp16/bf16 here,
                    # then upcasting again, to emulate casts that eager would do.
                    downcast = ops.to_dtype(out, dtype, use_compute_types=False)
                    return ops.to_dtype(downcast, dtype)
                return out

        if not override_device:
            device = None
            for i in inputs:
                if L.is_gpu(i.get_device().type):
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()

        device = override_device or device

        return ConditionalPointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=ranges,
        )

    return inner



@L.register_lowering(torch.ops.aten.where, broadcast=False, type_promotion_kind=None)
def where(cond, a, b):
    def fn(*args):
        return ops.where(*args)

    if isinstance(a, (float, int)):
        a = L.constant_like(a)(b)
    if isinstance(b, (float, int)):
        b = L.constant_like(b)(a)

    args = [cond, a, b]
    dtype = L.get_promoted_dtype(
        args[1], args[2], type_promotion_kind=L.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    indices = [i for i, x in enumerate(args) if isinstance(x, ir.TensorBox)]
    for i, x in zip(indices, L.broadcast_tensors(*[args[i] for i in indices])):
        args[i] = x
    for i in range(len(args)):
        if isinstance(args[i], ir.Constant):
            args[i] = ir.ExpandView.create(args[i], list(args[indices[0]].get_size()))
    result = make_pointwise(fn, override_return_dtype=dtype)(
        args[0], L.to_dtype(args[1], dtype), L.to_dtype(args[2], dtype)
    )
    result.realize()
    return result