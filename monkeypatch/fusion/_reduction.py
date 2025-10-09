
"""
Extended IR for bmm, which used to be templates for block indexing
Etended interpretation of original reduction and support new reduction IR  
"""
from .. import _monkey as monkey

import dataclasses
from typing import Any, Union, Tuple, Callable

import torch
import torch._inductor.config
from torch._inductor import ir, scheduler, lowering as L
from torch._inductor.virtualized import V, ops, OpsValue
from torch.utils._ordered_set import OrderedSet

from torch._inductor.scheduler import BaseSchedulerNode, SchedulerNode, FusedSchedulerNode, OutputNode, SchedulerBuffer
from torch._inductor.ops_handler import ReductionType, KernelFormatterHandler
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.triton import TritonScheduling, TritonKernelOverrides, TritonKernel, TritonCSEVariable
from torch._inductor.codegen.triton import triton_acc_type
from torch._inductor.codegen.simd import constant_repr
from torch._inductor.dependencies import Dep, MemoryDep, WeakDep
from torch._inductor.utils import reduction_num_outputs, is_welford_reduction, IndentedBuffer, sympy_product

import sympy
from sympy import Expr
from torch.utils._sympy.functions import FloorDiv, ModularIndexing
from torch.utils._ordered_set import OrderedSet

from torch._inductor.virtualized import ops, OpsWrapper, OpsValue

# add 'fusion' to comma-separated TORCH_LOG to enable
fusion_log = torch._logging.getArtifactLogger('torch._inductor', "fusion")  # fusion_log.debug(...)


#####
# Extended IR
#####

# bmm implementation using tl.dot leverages Triton's ability to use Tensor Cores. These hardware units inherently perform float16 or bfloat16 multiplications but accumulate the results in float32 for higher precision and performance.
# The original triton_compute_type function's upcasting behavior was controlled by a global configuration flag (config.triton.codegen_upcast_to_fp32), which made it conditional. Your patch makes this upcasting mandatory for float16/bfloat16, ensuring that the generated code for operations like your generalized bmm correctly uses float32 for accumulation, thereby producing more performant and numerically stable kernels.
torch._inductor.config.triton.codegen_upcast_to_fp32 = False

import torch._inductor.codegen.triton
_triton_compute_type = torch._inductor.codegen.triton.triton_compute_type
def triton_compute_type(dtype: torch.dtype):
    """Modifies torch._inductor.codegen.triton.triton_compute_type, which determines the Triton data type to be used for computations,
    to unconditionally upcast float16 and bfloat16 data types to float32
    for computations within Triton kernels.

    Motivation is to model the behavior of Tensor Cores performing f16*f16 + f32 -> f32 or bf16*bf16 + f32 -> f32 operations.
    """
    triton_type_name = str(dtype).split(".")[-1]
    if triton_type_name in ("float16", "bfloat16"):
        return "tl.float32"  # tl.dot output will be promoted to tl.float32  # TODO @bozhiyou only for tl.dot; for sum etc. still fp16/bf16?
    return _triton_compute_type(dtype)
torch._inductor.codegen.triton.triton_compute_type = triton_compute_type



@dataclasses.dataclass
class ReductionExt(ir.Reduction):
    """
    ReductionExt performs reduction on every block and accumulates the result. Loop-carried accumulator is a scalar and no final reduction required.
        (x0+x1+x2)+(x3+x4+x5)
    Reduction (the original Inductor IR) performs multi-lane reduction. Loop-carried accumulator is a vector and final reduction is required.
        (x0+x3, x2+x4, x3+x5) -> (x0+x3)+(x2+x4)+(x3+x5)
    """
    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: list[Expr],
        reduction_ranges: list[Expr],
        reduction_type: str,
        reduction_hint = ir.ReductionHint.DEFAULT,
        input_node: None|ir.IRNode|tuple[ir.IRNode, ...] = None,
    ):
        """
        + instanciate using cls rather than Reduction to allow subclassing
        """
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel <= 1 or (
            isinstance(reduction_numel, sympy.Integer)
            and V.graph.sizevars.size_hint(reduction_numel)
            < torch._inductor.config.unroll_reductions_threshold
            and sympy_product(ranges) != 1
        ):
            super().create(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                reduction_hint,
                input_node,
            )

        # triton doesn't support reduce to single element well, so break it up
        hint, split = cls.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            reduction_numel,
            input_node,
        )
        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ir.ReductionHint.DEFAULT:
            reduction_hint = hint
        if split == -1:
            assert input_node is not None
            new_ranges, new_reduction_ranges = ir.extract_input_node_reduction_ranges(
                input_node  # type: ignore[arg-type]
            )
            assert new_ranges is not None
            assert new_reduction_ranges is not None
            return cls.create_multilayer_existing_ranges(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                new_ranges,
                new_reduction_ranges,
                reduction_type,
                reduction_hint,
            )
        elif split > 1:
            # triton doesn't support reduce to single element well, so break it up
            return cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
            )

        return ir.TensorBox.create(
            cls(
                device,
                dst_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                src_dtype,
                reduction_hint,
            )
        )

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        """
        # + interpreted as block_reduction
        + interpreted as single-lane reduction
        """
        # value = ops.block_reduction(
        value = ops.reductionx(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
            contraction=True,
        )
        return ops.store_reduction(output_name, indexer(vars), value)


@monkey.patch(TritonKernel)
def reductionx(
    self: TritonKernel,
    dtype: torch.dtype,
    src_dtype: torch.dtype,
    reduction_type: ReductionType,
    value: Union[CSEVariable, Tuple[CSEVariable, ...]],
    # extended args
    multilane=True,     # i.e. RBLOCK > 1
    contraction=False,      # no r dim in result if True
    writeback_later=False,
    modification: int|None = None,
) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
    """Extend PyTorch Inductor's fusion capabilities to handle reduction fusion.
    
    A classic example is the softmax operation, which can be expressed as:
    1. max_val = max(x, dim=-1)
    2. exp_x = exp(x - max_val)
    3. sum_exp = sum(exp_x, dim=-1)
    4. output = exp_x / sum_exp
    Here, the sum reduction (step 3) depends on the result of the max reduction (step 1). Without this patch, Inductor would generate at least two separate loops: one for the max reduction and another for the sum reduction, with intermediate operations.
    This patch allows Inductor to fuse these steps into a single, highly efficient "online softmax" kernel. This is achieved by introducing a more flexible `reductionx` operation that can incorporate the logic of a dependent operation directly into its reduction loop.
    
    Original reduction handler uses multi-lane (parallel reduction over e.g. RBLOCK), if not persistent reduction (kernel level flag). This patch controls persistent reduction (multilane=False) at loop level.
    For non-persistent reduction, the original handler always performs a final reduction step, which is not necessary for contraction operation (e.g. matmul).
    With the control of `modification` and `writeback_later`, this patch enables reduction fusion.

    New args:
    - multilane (bool): When True (default), it behaves like the original reduction, performing a parallel reduction over a block of loop-carried data. When False, it performs a single-lane reduction, where the accumulator is a scalar, suitable for dot products.
    - contraction (bool): When True, it signifies that the reduction dimension is fully consumed and no final cross-thread reduction is needed. This is key for implementing matrix multiplication with block-level operations.
    - writeback_later (bool): When True, the result of the reduction's combine step is not immediately written back to the accumulator. This provides a hook to insert other operations.
    - modification (int): An identifier for a subgraph (the "update function") that should be executed on the accumulator before the final combine step. This is the mechanism for fusing dependent reductions, where executing the subgraph on accumulator reflects the update of its dependency.
    """
    assert self == V.kernel
    assert isinstance(self, TritonKernel)

    assert self.inside_reduction
    if not contraction and multilane:
        masks = OrderedSet(f"{tree.prefix}mask" for tree in self.range_trees)
        self.filter_masks(masks)
        masks = sorted(masks)
        if self._load_mask:
            masks.append(self._load_mask)
    else:
        masks = []
    reduction_range_prefix = self.range_trees[-1].prefix

    if not contraction and multilane:
        # Say we have
        #     tmp0 = ops.constant(1, torch.int64)
        #     tmp1 = ops.reduction(torch.int64, torch.int64, "sum", tmp0)
        # tmp0 in the triton code is either a scalar, or single-element tensor
        # so if we emit tl.sum directly, it will only give 1 instead of RBLOCK * 1
        # To avoid this, we broadcast to the expected shape first.
        dense_size_str = self.dense_size_str()
        value = self._map_tuple_or_scalar(
            lambda v: self.cse.generate(
                self.compute, f"tl.broadcast_to({v}, {dense_size_str})"
            ),
            value,
        )

    dim: int
    root_op: str

    def final_reduction(value):
        use_helper = reduction_type in {"any", "max", "min", "prod"}
        module = "triton_helpers" if use_helper else "tl"
        if reduction_type in {"max", "min"}:
            return self.reduction_resize(
                f"{module}.{reduction_type}2({value}, {dim})"
            )
        return self.reduction_resize(f"{module}.{reduction_type}({value}, {dim})")

    def final_argreduce(buffer, result_var, value, index):
        buffer.splice(
            f"""\
            _, {result_var}_tmp = triton_helpers.{root_op}_with_index({value}, {index}, {dim})
            {result_var} = {self.reduction_resize(f'{result_var}_tmp')}
            """
        )

    cache_key = (src_dtype, reduction_type, value)
    if cache_key in self.cse.reduction_cache:
        return self.cse.reduction_cache[cache_key]

    dim = self.triton_tensor_ndim() - 1
    acc_type = triton_acc_type(src_dtype)
    result_var: Any = self.cse.newvar()
    result_var.mask_vars = OrderedSet(var for var in masks if var[0] != "r")
    cond = " & ".join(masks)

    def where_cond(tval, fval):
        if not cond:
            return tval
        return TritonKernelOverrides.where(cond, tval, fval)

    if self.persistent_reduction:
        default = ir.Reduction.default_value(reduction_type, src_dtype)
        default = self._map_tuple_or_scalar(constant_repr, default)

        def _mask_value(value, default):
            return self.cse.generate(self.compute, where_cond(value, default))

        if isinstance(value, tuple):
            masked_value = [_mask_value(v, d) for v, d in zip(value, default)]
        else:
            masked_value = _mask_value(value, default)

        if reduction_type in {"argmax", "argmin"}:
            accumulator_index = str(
                self.cse.generate(
                    self.compute,
                    f"tl.broadcast_to({reduction_range_prefix}index, {masked_value}.shape)",
                )
            )
            root_op = {"argmax": "max", "argmin": "min"}[reduction_type]
            final_argreduce(
                self.compute, result_var, masked_value, accumulator_index
            )
        elif reduction_type == "welford_reduce":
            # For persistent reductions, don't bother with
            # welford's algorithm since it uses more registers, and
            # taking two reductions doesn't increase memory usage.
            result_var = self.welford_reduce_fallback(dtype, value)
        elif reduction_type == "welford_combine":
            mean, m2, weight = masked_value
            welford = f"triton_helpers.welford({mean}, {m2}, {weight}, {dim})"
            mean, m2, weight = (self.cse.newvar() for _ in range(3))
            self.compute.writeline(f"{mean}, {m2}, {weight} = {welford}")

            result_var = tuple(
                self.cse.generate(self.compute, self.reduction_resize(var_name))
                for var_name in (mean, m2, weight)
            )
        else:
            result_var = self.cse.generate(
                self.compute, final_reduction(masked_value)
            )
    else:
        accumulator = f"_{result_var}" if not contraction and multilane else result_var
        default = ir.Reduction.default_accumulator(reduction_type, src_dtype)
        default = self._map_tuple_or_scalar(constant_repr, default)
        prefix = self.body if self.range_trees[-1].is_loop else self.loads  # this is a one-shot loop TODO @bozhiyou persistent reduction?
        if not isinstance(default, tuple):
            if not contraction and multilane:
                prefix.writeline(
                    f"{accumulator} = tl.full({self.dense_size_str()}, {default}, {acc_type})"
                )
            else:
                if self.triton_tensor_ndim() == 1:
                    prefix.writeline(
                        f"{accumulator} = triton_helpers.promote_to_tensor({default})"
                    )
                else:
                    prefix.writeline(
                        f"{accumulator} = tl.full([{', '.join(self.dense_size_list()[:-1] + (['1'] if not contraction else []))}], {default}, {acc_type})"
                    )

        if reduction_type in {"argmax", "argmin"}:
            accumulator_index = f"_{result_var}_index"
            long_max = torch.iinfo(torch.int64).max
            prefix.writeline(
                f"{accumulator_index} = tl.full({self.dense_size_str()}, {long_max}, tl.int64)"
            )
            root_op = {"argmax": "max", "argmin": "min"}[reduction_type]

            self.compute.splice(
                f"""\
            {accumulator}_next, {accumulator_index}_next = triton_helpers.{root_op}imum_with_index(
                {accumulator}, {accumulator_index}, {value}, {reduction_range_prefix}index
            )
            {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
            {accumulator_index} = {where_cond(f'{accumulator_index}_next', accumulator_index)}
            """
            )
            final_argreduce(self.suffix, result_var, accumulator, accumulator_index)
        elif is_welford_reduction(reduction_type):
            accumulator = f"{result_var}_mean"
            accumulator_m2 = f"{result_var}_m2"
            accumulator_weight = f"{result_var}_weight"
            prefix.writeline(
                f"{accumulator} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )
            prefix.writeline(
                f"{accumulator_m2} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )
            prefix.writeline(
                f"{accumulator_weight} = tl.zeros({self.dense_size_str()}, {acc_type})"
            )

            if reduction_type == "welford_combine":
                mean, m2, weight = value
                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_combine(
                    {accumulator}, {accumulator_m2}, {accumulator_weight},
                    {mean}, {m2}, {weight}
                )
                """
                )
            else:
                assert reduction_type == "welford_reduce"
                self.compute.splice(
                    f"""\
                {accumulator}_next, {accumulator_m2}_next, {accumulator_weight}_next = triton_helpers.welford_reduce(
                    {value}, {accumulator}, {accumulator_m2}, {accumulator_weight}, roffset == 0
                )
                """
                )

            self.compute.splice(
                f"""\
            {accumulator} = {where_cond(f'{accumulator}_next', accumulator)}
            {accumulator_m2} = {where_cond(f'{accumulator_m2}_next', accumulator_m2)}
            {accumulator_weight} = {where_cond(f'{accumulator_weight}_next', accumulator_weight)}
            """
            )

            result_mean = result_var
            result_m2 = self.cse.newvar()
            result_weight = self.cse.newvar()
            self.suffix.splice(
                f"""\
            {result_mean}_tmp, {result_m2}_tmp, {result_weight}_tmp = triton_helpers.welford(
                {accumulator}, {accumulator_m2}, {accumulator_weight}, {dim}
            )
            {result_mean} = {self.reduction_resize(f'{result_mean}_tmp')}
            {result_m2} = {self.reduction_resize(f'{result_m2}_tmp')}
            {result_weight} = {self.reduction_resize(f'{result_weight}_tmp')}
            """
            )
            result_var = result_mean, result_m2, result_weight
        else:
            if modification is not None:
                updated_accumulator = self.overrides.modification(modification[0], accumulator, **modification[1])
                self.compute.writeline(
                        f"{accumulator} = {updated_accumulator}"
                    )

            if not contraction and not multilane:
                if reduction_type in {"max", "min"}:
                    # NOTE @bozhiyou not using triton_helpers.max2 here
                    value = self.reduction_resize(getattr(ops, reduction_type)(value, dim))
                else:
                    value = final_reduction(value)
            combine_fn = ir.get_reduction_combine_fn(reduction_type, src_dtype)
            updated = combine_fn(accumulator, value)

            if not writeback_later:
                if not contraction and multilane:
                    self.compute.writeline(
                        f"{accumulator} = {where_cond(updated, accumulator)}"
                    )
                else:
                    self.compute.writeline(
                        f"{accumulator} = {updated}"
                    )
            else:
                self.cse.reduction_cache[ops._unwrap(updated)] = result_var
                self.stores.writeline(
                    f"{accumulator} = {where_cond(updated, accumulator)}"
                )
            if not contraction and multilane:
                if src_dtype == torch.bool:
                    # This is only really used for aten.any. It changes the
                    # final reduction of a non-persistent reduction from
                    #     tmp5 = triton_helpers.max(_tmp5, 1)[:, None]
                    # to
                    #     tmp5 = triton_helpers.max(_tmp5.to(tl.int8), 1)[:, None].to(tl.int1)
                    # which is needed because tl.reduce doesn't support tl.int1
                    accumulator = f"{accumulator}.to(tl.int8)"
                    result_type = triton_compute_type(dtype)
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}.to({result_type})"
                    )
                else:
                    self.suffix.writeline(
                        f"{result_var} = {final_reduction(accumulator)}"
                    )

    self.cse.reduction_cache[cache_key] = result_var

    if isinstance(result_var, tuple):
        assert all(isinstance(x, TritonCSEVariable) for x in result_var)
        self.outside_loop_vars |= OrderedSet(result_var)
    else:
        assert isinstance(result_var, TritonCSEVariable)
        self.outside_loop_vars.add(result_var)

    return result_var if not writeback_later else ops._unwrap(updated)


@monkey.patch(SchedulerNode)
def codegen(self, index_vars) -> None:
    """
    + extend interpretation of reduction
    """
    @monkey.patch(V.ops)
    def reductionx(
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: Union[CSEVariable, Tuple[CSEVariable, ...]],
        **kwargs
    ) -> Union[CSEVariable, Tuple[CSEVariable, ...]]:
        V.kernel.num_reduction += 1
        return V.kernel.reductionx(dtype, src_dtype, reduction_type, value, **kwargs)
    return monkey.fallback(self, index_vars)

