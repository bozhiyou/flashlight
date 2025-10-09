"""
At a high level, patches in this library transform PyTorch Inductor into a more powerful kernel fusion engine, capable of generating highly efficient, monolithic kernels for complex patterns that would otherwise result in multiple, less efficient kernels.

Core ideas:

- Reduction Fusion: a mechanism to fuse a reduction with another reduction that depends on its output. This transformation can, for example, fuse the components of a softmax operation (which involves max, sub, exp, sum, div) into a single "online softmax" kernel, similar to what's in FlashAttention. This is achieved through clever use of homomorphic properties of operations to rewrite the computation graph.

- Generalizing Matrix Multiplication: Instead of treating matrix multiplication (bmm, mm) as a special-cased template, modeling it as a form of generalized reduction (ReductionExt). This is a more general and flexible abstraction that allows bmm to participate in the fusion system more naturally with other pointwise and reduction operations.

- Enhanced Scheduling and Tiling: To make these advanced fusions possible, significant enhancements to Inductor's scheduling and code generation logic, particularly for the Triton backend, include more flexible tiling strategies and more powerful indexing capabilities.

monkeypatch/fusion/
├── __init__.py
├── bmm/
│   ├── __init__.py
│   ├── ir.py          # ReductionExt IR definition
│   └── lowering.py    # BMM lowering logic and related FX patterns
├── codegen/
│   ├── __init__.py
│   └── triton.py      # Patches and helpers for Triton code generation
├── scheduler/
│   ├── __init__.py
│   ├── rules.py       # Fusion rule patches (e.g., can_fuse_vertical)
│   └── indexing.py    # Patches for indexing, tiling, and range handling
└── passes/
    ├── __init__.py
    ├── dependent_reduction.py  # Core logic for dependent reduction fusion
    └── fx_patterns.py          # FX-level graph patterns
"""


##
# debug breakpoint
##
from .. import _monkey as monkey
import torch
from torch._inductor import ir, lowering as L, scheduler
from torch._inductor.virtualized import V
from torch._inductor.loop_body import LoopBody
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.simd import IterationRangesRoot
from torch._inductor.sizevars import SizeVarAllocator


@monkey.patch(ir.ExpandView)
def __post_init__(self):
    super(ir.ExpandView, self).__post_init__()
    pass

@monkey.patch(ir.ComputedBuffer)
def __post_init__(self):
    super(ir.ComputedBuffer, self).__post_init__()
    pass

# @monkey.patch(ir.Reduction)
# def store_reduction(self: ir.Reduction, output_name, indexer, vars, reduction_vars):
#     """
#     reduction opsvalue flow
#     """
#     # load
#     value = self.inner_fn(vars, reduction_vars)
#     # reduction
#     value = ops.reduction(
#         self.dtype,
#         self.src_dtype,
#         self.reduction_type,
#         value
#     )
#     # store
#     value = ops.store_reduction(output_name, indexer(vars), value)
#     return value


@monkey.property(IterationRangesRoot)
def tensor_dim(self):
    return self._tensor_dim

@tensor_dim.setter
def tensor_dim(self, value):
    self._tensor_dim = value

@monkey.property(LoopBody)
def indexing_exprs(self):
    return self._indexing_exprs

@indexing_exprs.setter
def indexing_exprs(self, value):
    self._indexing_exprs = value

@monkey.property(TritonKernel)
def args(self):
    return self._args

@args.setter
def args(self, value):
    self._args = value

@monkey.property(TritonKernel)
def numels(self):
    return self._numel

@numels.setter
def numels(self, value):
    self._numel = value


from torch._inductor.bounds import BoundVars
@monkey.property(BoundVars)
def replacement_vals(self):
    return self._replacement_vals

@replacement_vals.setter
def replacement_vals(self, value):
    self._replacement_vals = value


@monkey.property(BoundVars)
def _bounds(self):
    return self._replacement_vals

@_bounds.setter
def _bounds(self, value):
    self._replacement_vals = value

@monkey.property(SizeVarAllocator)
def inv_precomputed_replacements(self):
    return self._inv_precomputed_replacements

@inv_precomputed_replacements.setter
def inv_precomputed_replacements(self, value):
    self._inv_precomputed_replacements = value

from torch._inductor.codegen.wrapper import WrapperCodeGen
@monkey.property(WrapperCodeGen)
def computed_sizes(self):
    return self._computed_sizes

@computed_sizes.setter
def computed_sizes(self, value):
    self._computed_sizes = value


