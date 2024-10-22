import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, ensure_contiguous


@triton.jit
def _relu_squared_forward_kernel(
    x_ptr, y_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    x_ptr += program_id * stride
    y_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    x_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    y_row = tl.maximum(x_row, 0) ** 2
    tl.store(y_ptr + col_offsets, y_row, mask=mask)


@triton.jit
def _relu_squared_backward_kernel(
    dy_ptr, x_ptr, dx_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dy_ptr += program_id * stride
    x_ptr += program_id * stride
    dx_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dy_row = tl.load(dy_ptr + col_offsets, mask=mask, other=0)
    x_row = tl.load(x_ptr + col_offsets, mask=mask, other=0).to(tl.float32)

    dx_row = dy_row * 2 * tl.maximum(x_row, 0)
    tl.store(dx_ptr + col_offsets, dx_row, mask=mask)


def relu_squared_forward(x):
    ori_shape = x.shape

    n_cols = ori_shape[-1]
    x = x.view(-1, n_cols)
    y = torch.empty_like(x)
    n_rows = x.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _relu_squared_forward_kernel[(n_rows,)](
        x,
        y,
        y.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return x, y.view(*ori_shape)


def relu_squared_backward(x, dy):
    ori_shape = dy.shape
    n_cols = ori_shape[-1]
    dy = dy.view(-1, n_cols)
    n_rows = dy.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    dx = torch.empty_like(dy)

    _relu_squared_backward_kernel[(n_rows,)](
        dy,
        x,
        dx,
        dx.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return dx.view(*ori_shape)


class LigerReLUSquaredFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, x):
        x, y = relu_squared_forward(x)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = relu_squared_backward(x, dy)
        return dx
