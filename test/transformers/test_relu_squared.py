import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.activations import get_activation

from liger_kernel.ops.relu_squared import LigerReLUSquaredFunction
from liger_kernel.transformers.functional import liger_relu_squared
from liger_kernel.transformers.relu_squared import LigerReLUSquaredMLP

LLAMA_CONFIG = LlamaConfig(
    hidden_size=4096,
    intermediate_size=11008,
    hidden_act="relu_squared",
)
SLEEP_SECONDS = 0.1


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 2048, 4096, 11008),
        (2, 2048, 2048, 4096),
        (9, 41, 341, 4231),
        (6, 42, 256, 2048),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-0, 2e-6),
        pytest.param(
            torch.bfloat16,
            1e4,
            6e-3,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
def test_correctness(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    G = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)

    llama_mlp = LlamaMLP(config=LLAMA_CONFIG).to("cuda").to(dtype)
    llama_mlp.gate_proj.weight.data = G.T
    llama_mlp.up_proj.weight.data = U.T
    llama_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerReLUSquaredMLP(config=LLAMA_CONFIG).to("cuda").to(dtype)
    liger_mlp.gate_proj.weight.data = G.T
    liger_mlp.up_proj.weight.data = U.T
    liger_mlp.down_proj.weight.data = D.T

    y1 = llama_mlp(x1)
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol) is True

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert (
        torch.allclose(
            llama_mlp.gate_proj.weight.grad,
            liger_mlp.gate_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )
    assert (
        torch.allclose(
            llama_mlp.up_proj.weight.grad,
            liger_mlp.up_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )
    assert (
        torch.allclose(
            llama_mlp.down_proj.weight.grad,
            liger_mlp.down_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol) is True


@pytest.mark.parametrize(
    "bsz, seq_len, size",
    [
        (2, 2, 8),
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-0, 2e-6),
        (torch.bfloat16, 1e4, 6e-3),
    ],
)
def test_correctness_functional(bsz, seq_len, size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, size, device="cuda", dtype=dtype)
    _b = torch.randn(bsz, seq_len, size, device="cuda", dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    b1 = _b.clone().requires_grad_(True)
    b2 = _b.clone().requires_grad_(True)

    y1 = liger_relu_squared(x1, b1)
    y2 = LigerReLUSquaredFunction.apply(x2, b2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y1)

    y1.backward(grad_output)
    y2.backward(grad_output)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size",
    [
        (2, 2048, 4096, 11008),
        (2, 2048, 2048, 4096),
        (9, 41, 341, 4231),
        (6, 42, 256, 2048),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-0, 2e-6),
        pytest.param(
            torch.bfloat16,
            1e4,
            6e-3,
            marks=pytest.mark.skipif(
                not torch.cuda.is_bf16_supported(), reason="bfloat16 not supported on this GPU"
            ),
        ),
    ],
)
def test_correctness_against_transformers(bsz, seq_len, hidden_size, intermediate_size, dtype, atol, rtol):
    _input = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)

    x1 = _input.clone().requires_grad_(True)
    x2 = _input.clone().requires_grad_(True)

    G = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
    U = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype)
    D = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype)

    llama_mlp = LlamaMLP(config=LLAMA_CONFIG).to("cuda").to(dtype)
    llama_mlp.gate_proj.weight.data = G.T
    llama_mlp.up_proj.weight.data = U.T
    llama_mlp.down_proj.weight.data = D.T

    liger_mlp = LigerReLUSquaredMLP(config=LLAMA_CONFIG).to("cuda").to(dtype)
    liger_mlp.gate_proj.weight.data = G.T
    liger_mlp.up_proj.weight.data = U.T
    liger_mlp.down_proj.weight.data = D.T

    transformers_activation = get_activation("relu2")

    y1 = transformers_activation(llama_mlp(x1))
    y2 = liger_mlp(x2)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol) is True

    dy = torch.randn_like(y1)

    y1.backward(dy.clone(), retain_graph=True)
    y2.backward(dy.clone(), retain_graph=True)

    assert (
        torch.allclose(
            llama_mlp.gate_proj.weight.grad,
            liger_mlp.gate_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )
    assert (
        torch.allclose(
            llama_mlp.up_proj.weight.grad,
            liger_mlp.up_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )
    assert (
        torch.allclose(
            llama_mlp.down_proj.weight.grad,
            liger_mlp.down_proj.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        is True
    )

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol) is True
