"""
Runtime autotune
blockreduction as a new heuristic type
"""
from .. import _monkey as monkey
from ._common import TRITON_MAX_RBLOCK

import torch
import torch._inductor.config
from torch._inductor.utils import ceildiv
import torch._inductor.runtime.triton_heuristics
from torch._inductor.runtime.triton_heuristics import cached_autotune, get_max_y_grid
from torch._inductor.runtime.hints import HeuristicType
from torch._inductor.wrapper_benchmark import _kernel_category_choices

import triton

@monkey.patch(torch._inductor.runtime.triton_heuristics)
def grid(*numels):
    """Helper function to compute triton grids"""
    if len(numels) == 1:
        xnumel, ynumel, znumel = numels[0], None, None
    elif len(numels) == 2:
        xnumel, ynumel, znumel = numels[1], numels[0], None
    elif len(numels) == 3:
        xnumel, ynumel, znumel = numels[2], numels[1], numels[0]
    else:
        raise AssertionError(f"invalid size for numels {len(numels)}")

    def get_grid_dim(numel, block):
        if numel is None:
            return 1
        if block is None:
            return numel
        return ceildiv(numel, block)

    import operator
    def grid_fn(meta):
        x_grid = xnumel
        xblock = meta.get("XBLOCK", 1)
        if isinstance(xblock, tuple):  # {number of elements: number of blocks}
            for ne, nb in xblock:
                assert x_grid % ne == 0, f"{x_grid} is not multiple of {ne}! {xnumel=} {xblock=}"
                x_grid //= ne
                x_grid *= nb
        else:
            x_grid = get_grid_dim(xnumel, xblock)
        # x_grid = get_grid_dim(xnumel, meta.get("XBLOCK", 1))
        # for k, v in meta.items():
        #     if k.startswith('xnumbl'):
        #         x_grid *= v
        y_grid = get_grid_dim(ynumel, meta.get("YBLOCK", None))

        max_y_grid = get_max_y_grid()
        if znumel is None:
            div = ceildiv(y_grid, max_y_grid)
            y_grid = ceildiv(y_grid, div)
            z_grid = div
        else:
            z_grid = get_grid_dim(znumel, meta.get("ZBLOCK", None))
            torch._check(
                y_grid <= max_y_grid,
                lambda: f"Generated y grid beyond 2^16 ({y_grid}) not supported with z dimension present. File issue",
            )

        return (
            x_grid,
            y_grid,
            z_grid,
        )

    setattr(grid_fn, "grid_fn_str", f"grid{numels}")  # noqa: B010

    return grid_fn


def _blockreduction_configs(
    *,
    size_hints: dict[str, int],
    inductor_meta={},
):
    f"""
    Config space from {torch._inductor.kernel.flex_attention} and {torch._inductor.kernel.mm_common}.
    """
    from torch._inductor.kernel.flex_attention import _get_default_config_fwd
    # (BLOCK_M, BLOCK_N, num_warps, num_stages)
    configs: list[tuple[int, int, int, int]] = [(128, 64, 4, 3), (128, 64, 8, 3)]  # TODO @bozhiyou default to max or 1?
    # configs = [(x, r, w, s) for x in (128, 64) for r in (16, 32, 64, 128) for w in (4, 8) for s in (2, 3)]
    # configs.append(_get_default_config_fwd(query))
    if torch._inductor.config.max_autotune:
        configs += [
            (128, 64, 4, 3),
            (128, 128, 4, 3),
            (128, 128, 8, 2),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
        ]

    if 'block_args' in inductor_meta:
        block_hints = inductor_meta['block_args']
        # 'block_args':
        #   # _numel: number of grouped elements, _numel = prod(_numbl_)
        #   'xnumel': 16384,
        #
        #   # _numbl_: nnn means _numbl_ = ceildiv(nnn, _BLOCK_)
        #   'xnumbl0': 16384,
        #   'RBLOCK1': None, 'RBLOCK': None
        def block_config(xblock, rblock):
            block = {'x': xblock, 'r': rblock}
            c = {"XBLOCK": ()}
            for arg, v in block_hints.items():
                if 'numbl' in arg:
                    prefix, suffix = arg.split('numbl')
                    c[arg] = ceildiv(v, block[prefix])
                    c[f"{prefix.upper()}BLOCK{suffix}"] = block[prefix]
                    c[f"{prefix.upper()}BLOCK"] += ((v, c[arg]),)
                if 'BLOCK' in arg:
                    if v <= TRITON_MAX_RBLOCK:
                        c[arg] = v
                    else:
                        c[arg] = block[arg[0].lower()]
            return c

        return [
            triton.Config({  # keys must be kernel args
                **block_config(XBLOCK, RBLOCK),
                # "XBLOCK": inductor_meta['block_args']["xnumel"],  # force XBLOCK to 1 to recalculate number of blocks
                "RBLOCK": RBLOCK,
            }, num_stages=num_stages, num_warps=num_warps)
            for (XBLOCK, RBLOCK, num_warps, num_stages) in configs
        ]

    return [
        triton.Config({
            "XBLOCK": XBLOCK,
            "RBLOCK": RBLOCK,
        }, num_stages=num_stages, num_warps=num_warps)
        for (XBLOCK, RBLOCK, num_warps, num_stages) in configs
    ]


@monkey.patch(torch._inductor.runtime.triton_heuristics)
def blockreduction(
    size_hints: dict[str, int],
    reduction_hint=False,
    triton_meta={},
    filename=None,
    inductor_meta={},
):
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    configs = _blockreduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )

_kernel_category_choices.append('blockreduction')




