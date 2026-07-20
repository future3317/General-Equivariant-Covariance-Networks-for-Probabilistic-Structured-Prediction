"""Regression tests for algebra-preserving backbone optimizations."""

import importlib.util

import pytest
import torch

from models.backbone import EquivariantMessagePassing
from compatibility.e3nn import o3


def _reference_layer_forward(
    layer,
    node_features,
    edge_source,
    edge_target,
    edge_attributes,
    edge_embedding,
    edge_weight,
):
    weights = layer.fc(edge_embedding)
    messages = layer.tp(
        node_features[edge_source], edge_attributes, weights
    ) * edge_weight.unsqueeze(-1)
    summed = messages.new_zeros((node_features.size(0), messages.size(-1)))
    counts = messages.new_zeros(node_features.size(0))
    for edge_index, destination in enumerate(edge_target.tolist()):
        summed[destination] = summed[destination] + messages[edge_index]
        counts[destination] += 1
    aggregated = summed / torch.sqrt(counts + 1e-8).unsqueeze(-1)
    return layer.act(layer.self_interaction(aggregated) + layer.skip(node_features))


def test_shared_degree_and_native_index_add_match_reference():
    torch.manual_seed(17)
    layer = EquivariantMessagePassing(
        o3.Irreps("3x0e + 2x1o"),
        o3.Irreps("4x0e + 2x1o"),
        o3.Irreps("0e + 1o"),
        num_basis=4,
    ).double()
    node_features = layer.irreps_node_input.randn(6, -1, dtype=torch.float64)
    node_features.requires_grad_(True)
    edge_source = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 4])
    edge_target = torch.tensor([1, 2, 3, 4, 5, 0, 3, 5, 1])
    edge_vectors = torch.randn(edge_source.numel(), 3, dtype=torch.float64)
    edge_attributes = o3.spherical_harmonics(
        layer.irreps_edge_attr,
        edge_vectors,
        normalize=True,
        normalization="component",
    )
    edge_embedding = torch.randn(edge_source.numel(), 4, dtype=torch.float64)
    edge_weight = torch.rand(edge_source.numel(), dtype=torch.float64)
    counts = torch.bincount(edge_target, minlength=node_features.size(0)).double()
    node_norm = torch.rsqrt(counts + 1e-8)

    optimized = layer(
        node_features,
        edge_source,
        edge_target,
        edge_attributes,
        edge_embedding,
        edge_weight,
        node_norm,
    )
    reference = _reference_layer_forward(
        layer,
        node_features,
        edge_source,
        edge_target,
        edge_attributes,
        edge_embedding,
        edge_weight,
    )
    # The vectorized reduction and explicit reference loop need not be bitwise
    # identical because floating-point addition order may differ.
    torch.testing.assert_close(optimized, reference, atol=2e-8, rtol=5e-6)

    optimized_gradient = torch.autograd.grad(
        optimized.square().sum(), node_features, retain_graph=True
    )[0]
    reference_gradient = torch.autograd.grad(reference.square().sum(), node_features)[0]
    torch.testing.assert_close(
        optimized_gradient, reference_gradient, atol=2e-8, rtol=5e-6
    )


def test_cueq_matches_e3nn_weight_layout_and_gradients():
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    irreps_in = o3.Irreps("3x0e + 2x1o")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    edge_irreps = o3.Irreps("0e + 1o")
    e3nn_layer = EquivariantMessagePassing(
        irreps_in, irreps_out, edge_irreps, num_basis=4, tp_backend="e3nn"
    ).double()
    cueq_layer = EquivariantMessagePassing(
        irreps_in, irreps_out, edge_irreps, num_basis=4, tp_backend="cueq"
    ).double()
    assert e3nn_layer.tp.weight_numel == cueq_layer.tp.weight_numel

    x1_e3nn = irreps_in.randn(7, -1, dtype=torch.float64).requires_grad_(True)
    x2_e3nn = edge_irreps.randn(7, -1, dtype=torch.float64).requires_grad_(True)
    weights_e3nn = torch.randn(
        7, e3nn_layer.tp.weight_numel, dtype=torch.float64, requires_grad=True
    )
    output_e3nn = e3nn_layer.tp(x1_e3nn, x2_e3nn, weights_e3nn)
    gradients_e3nn = torch.autograd.grad(
        output_e3nn.square().sum(), (x1_e3nn, x2_e3nn, weights_e3nn)
    )

    x1_cueq = x1_e3nn.detach().clone().requires_grad_(True)
    x2_cueq = x2_e3nn.detach().clone().requires_grad_(True)
    weights_cueq = weights_e3nn.detach().clone().requires_grad_(True)
    output_cueq = cueq_layer.tp(x1_cueq, x2_cueq, weights_cueq)
    gradients_cueq = torch.autograd.grad(
        output_cueq.square().sum(), (x1_cueq, x2_cueq, weights_cueq)
    )

    torch.testing.assert_close(output_cueq, output_e3nn, atol=1e-10, rtol=1e-10)
    for cueq_gradient, e3nn_gradient in zip(gradients_cueq, gradients_e3nn):
        torch.testing.assert_close(cueq_gradient, e3nn_gradient, atol=1e-10, rtol=1e-10)


def test_cueq_fused_backend_does_not_silently_fallback():
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if importlib.util.find_spec("cuequivariance_ops_torch") is not None:
        pytest.skip("native cuEquivariance ops are installed")
    with pytest.raises(RuntimeError, match="cuequivariance_ops_torch"):
        EquivariantMessagePassing(
            o3.Irreps("2x0e"),
            o3.Irreps("2x0e"),
            o3.Irreps("0e"),
            num_basis=2,
            tp_backend="cueq",
            cueq_method="fused_tp",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.skipif(
    importlib.util.find_spec("cuequivariance_ops_torch") is None,
    reason="native cuEquivariance ops are unavailable",
)
def test_cueq_fused_cuda_matches_e3nn_forward_and_gradients():
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    irreps_in = o3.Irreps("3x0e + 2x1o")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    edge_irreps = o3.Irreps("0e + 1o")
    e3nn_layer = EquivariantMessagePassing(
        irreps_in, irreps_out, edge_irreps, num_basis=4, tp_backend="e3nn"
    ).cuda()
    cueq_layer = EquivariantMessagePassing(
        irreps_in,
        irreps_out,
        edge_irreps,
        num_basis=4,
        tp_backend="cueq",
        cueq_method="fused_tp",
    ).cuda()
    assert e3nn_layer.tp.weight_numel == cueq_layer.tp.weight_numel

    inputs = irreps_in.randn(11, -1, device="cuda").requires_grad_(True)
    edges = edge_irreps.randn(11, -1, device="cuda").requires_grad_(True)
    weights = torch.randn(
        11,
        e3nn_layer.tp.weight_numel,
        device="cuda",
        requires_grad=True,
    )
    e3nn_output = e3nn_layer.tp(inputs, edges, weights)
    e3nn_gradients = torch.autograd.grad(
        e3nn_output.square().sum(), (inputs, edges, weights)
    )

    cueq_inputs = inputs.detach().clone().requires_grad_(True)
    cueq_edges = edges.detach().clone().requires_grad_(True)
    cueq_weights = weights.detach().clone().requires_grad_(True)
    cueq_output = cueq_layer.tp(cueq_inputs, cueq_edges, cueq_weights)
    cueq_gradients = torch.autograd.grad(
        cueq_output.square().sum(), (cueq_inputs, cueq_edges, cueq_weights)
    )

    torch.testing.assert_close(cueq_output, e3nn_output, atol=2e-6, rtol=2e-5)
    for cueq_gradient, e3nn_gradient in zip(cueq_gradients, e3nn_gradients):
        torch.testing.assert_close(cueq_gradient, e3nn_gradient, atol=2e-5, rtol=2e-5)
