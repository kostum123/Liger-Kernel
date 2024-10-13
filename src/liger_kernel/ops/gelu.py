import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import (
    calculate_settings,
    compare_version,
    ensure_contiguous,
)

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def _gelu_tanh_forward_kernel(
    a, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    gelu_a = 0.5 * a_row * (1 + tanh_result)
    tl.store(c + col_offsets, gelu_a, mask=mask)


@triton.jit
def _gelu_tanh_backward_kernel(
    dc, a, da, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    da += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)

    # recomputation to save memory
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)

    # Gradient w.r.t. a can be computed with:
    # 0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2))
    # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * a_row
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    )
    da_row = dc_row * (term1 + term2)

    tl.store(da + col_offsets, da_row, mask=mask)


@triton.jit
def _gelu_exact_forward_kernel(
    a, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)

    # exact form of GELU is computed with:
    # 0.5 * a * (1 + erf(a / sqrt(2)))
    sqrt_2 = 1.4142135623730951  # sqrt(2)
    erf_arg = a_row / sqrt_2
    erf_result = tl.libdevice.erf(erf_arg)
    gelu_a = 0.5 * a_row * (1 + erf_result)
    tl.store(c + col_offsets, gelu_a, mask=mask)


@triton.jit
def _gelu_exact_backward_kernel(
    dc, a, da, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    da += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)

    # recomputation to save memory
    sqrt_2 = 1.4142135623730951  # sqrt(2)
    erf_arg = a_row / sqrt_2
    erf_result = tl.libdevice.erf(erf_arg)

    # Gradient w.r.t. a can be computed with:
    # 0.5 * (1 + erf(a / sqrt(2))) + 0.5 * a * (2 / sqrt(pi)) * exp(-a^2 / 2)
    term1 = 0.5 * (1 + erf_result)
    exp_arg = -0.5 * a_row * a_row
    exp_result = tl.libdevice.exp(exp_arg)
    term2 = 0.5 * a_row * (2 / 3.141592653589793) * exp_result
    da_row = dc_row * (term1 + term2)

    tl.store(da + col_offsets, da_row, mask=mask)


def gelu_forward(a, exact=False):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    if exact:
        _gelu_exact_forward_kernel[(n_rows,)](
            a,
            c,
            c.stride(-2),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        _gelu_tanh_forward_kernel[(n_rows,)](
            a,
            c,
            c.stride(-2),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return a, c.view(*ori_shape)


def gelu_backward(a, dc, exact=False):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    a = a.view(-1, n_cols)
    n_rows = dc.shape[0]
    da = torch.empty_like(dc)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    if exact:
        _gelu_exact_backward_kernel[(n_rows,)](
            dc,
            a,
            da,
            dc.stride(-2),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        _gelu_tanh_backward_kernel[(n_rows,)](
            dc,
            a,
            da,
            dc.stride(-2),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return da.view(*ori_shape)


class LigerGELUFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, exact=False):
        a, c = gelu_forward(a, exact)
        ctx.save_for_backward(a)
        ctx.exact = exact
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, = ctx.saved_tensors
        da = gelu_backward(a, dc, ctx.exact)
        return da, None
