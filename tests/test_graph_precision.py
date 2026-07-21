"""Tests for graph-structured equivariant precision models."""

import math

import torch
from e3nn import o3

from data.itop_dataset import ITOP_INDEPENDENT_GRAPH
from distributions import GaussianNLL, StudentTNLL
from equivcompiler import (
    FeatureSpec,
    FirstFeasible,
    FullCovariance,
    GraphPrecision,
    IsotypicBlockCovariance,
    LowRankCovariance,
    plan_readout,
)
from representations import (
    EquivariantOutputGraph,
    O3IrrepsSpec,
)
from spd_maps import GraphStructuredPrecisionMap


EDGES = ((0, 1), (1, 2), (1, 3))


def _graph() -> EquivariantOutputGraph:
    return EquivariantOutputGraph(num_nodes=4, edges=EDGES, node_irrep="1o")


def _symmetric_params(batch: int = 2) -> torch.Tensor:
    graph = _graph()
    raw = torch.randn(batch, graph.num_potentials, 3, 3)
    return 0.5 * (raw + raw.transpose(-1, -2))


def test_graph_structure_incidence_and_parameter_count():
    graph = _graph()
    incidence = graph.incidence_matrix()
    assert incidence.shape == (3, 4)
    assert torch.allclose(incidence.sum(-1), torch.zeros(3))
    assert graph.output_dim == 12
    assert graph.num_potentials * 6 == 42
    assert graph.is_tree


def test_graph_precision_is_spd_and_covariance_is_dense():
    graph = _graph()
    spd_map = GraphStructuredPrecisionMap(graph)
    params = _symmetric_params()
    precision = spd_map.precision(params)
    covariance = spd_map(params)
    assert torch.linalg.eigvalsh(precision).min().item() > 0.0
    assert torch.linalg.eigvalsh(covariance).min().item() > 0.0
    identity = precision @ covariance
    assert torch.allclose(identity, torch.eye(graph.output_dim), atol=2e-5, rtol=2e-5)
    # End nodes are not adjacent, yet marginalization generally couples them.
    assert covariance[..., 0:3, 9:12].abs().max().item() > 1e-6


def test_local_precision_action_and_logdet_match_dense_algebra():
    graph = _graph()
    spd_map = GraphStructuredPrecisionMap(graph)
    params = _symmetric_params()
    residual = torch.randn(2, graph.output_dim)
    precision = spd_map.precision(params)
    expected_action = torch.einsum("bi,bij,bj->b", residual, precision, residual)
    expected_logdet = -torch.linalg.slogdet(precision).logabsdet
    assert torch.allclose(
        spd_map.precision_action(params, residual),
        expected_action,
        atol=2e-5,
        rtol=2e-5,
    )
    assert torch.allclose(spd_map.logdet(params), expected_logdet, atol=2e-5, rtol=2e-5)


def test_joint_graph_statistics_reuse_local_exponentials():
    class CountingGraphMap(GraphStructuredPrecisionMap):
        local_calls = 0

        def local_precisions(self, params):
            self.local_calls += 1
            return super().local_precisions(params)

    graph = _graph()
    spd_map = CountingGraphMap(graph)
    params = _symmetric_params()
    residual = torch.randn(2, graph.output_dim)
    logdet, quadratic = spd_map.statistics(params, residual)
    assert spd_map.local_calls == 1
    precision = spd_map.precision(params)
    expected_logdet = -torch.linalg.slogdet(precision).logabsdet
    expected_quadratic = torch.einsum("bi,bij,bj->b", residual, precision, residual)
    torch.testing.assert_close(logdet, expected_logdet, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(quadratic, expected_quadratic, atol=2e-5, rtol=2e-5)


def test_tree_statistics_match_dense_gradients():
    graph = _graph()
    spd_map = GraphStructuredPrecisionMap(graph).double()
    params_tree = _symmetric_params().double().requires_grad_(True)
    params_dense = params_tree.detach().clone().requires_grad_(True)
    residual_tree = torch.randn(2, graph.output_dim, dtype=torch.float64)
    residual_tree.requires_grad_(True)
    residual_dense = residual_tree.detach().clone().requires_grad_(True)

    tree_logdet, tree_quadratic = spd_map.statistics(params_tree, residual_tree)
    dense_precision = spd_map.precision(params_dense)
    dense_logdet = -torch.linalg.slogdet(dense_precision).logabsdet
    dense_quadratic = torch.einsum(
        "bi,bij,bj->b", residual_dense, dense_precision, residual_dense
    )
    (tree_logdet + tree_quadratic).sum().backward()
    (dense_logdet + dense_quadratic).sum().backward()

    torch.testing.assert_close(tree_logdet, dense_logdet, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(tree_quadratic, dense_quadratic, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        params_tree.grad, params_dense.grad, atol=2e-9, rtol=2e-9
    )
    torch.testing.assert_close(
        residual_tree.grad, residual_dense.grad, atol=1e-10, rtol=1e-10
    )


def test_cyclic_graph_uses_dense_logdet_path():
    graph = EquivariantOutputGraph(
        num_nodes=3,
        edges=((0, 1), (1, 2), (2, 0)),
        node_irrep="1o",
    )
    assert not graph.is_tree
    spd_map = GraphStructuredPrecisionMap(graph)
    raw = torch.randn(2, graph.num_potentials, 3, 3)
    params = 0.5 * (raw + raw.transpose(-1, -2))
    precision = spd_map.precision(params)
    expected = -torch.linalg.slogdet(precision).logabsdet
    torch.testing.assert_close(spd_map.logdet(params), expected, atol=2e-5, rtol=2e-5)


def test_edgeless_graph_uses_exact_independent_joint_logdet():
    graph = EquivariantOutputGraph(num_nodes=15, edges=(), node_irrep="1o")
    spd_map = GraphStructuredPrecisionMap(graph)
    raw = torch.randn(2, graph.num_potentials, 3, 3, dtype=torch.float64)
    params = 0.5 * (raw + raw.transpose(-1, -2))
    precision = spd_map.precision(params)
    expected = -torch.linalg.slogdet(precision).logabsdet
    torch.testing.assert_close(spd_map.logdet(params), expected, atol=1e-10, rtol=1e-10)


def test_itop_independent_joint_family_has_fifteen_full_3x3_blocks():
    plan = GraphPrecision(ITOP_INDEPENDENT_GRAPH).compile(
        O3IrrepsSpec(ITOP_INDEPENDENT_GRAPH.output_irreps)
    )
    assert plan.parameter_count == 15 * 6
    assert plan.graph.num_edges == 0


def test_graph_precision_equivariance():
    torch.manual_seed(4)
    graph = _graph()
    spd_map = GraphStructuredPrecisionMap(graph)
    params = _symmetric_params(batch=1)
    rotation = o3.rand_matrix()
    transformed_params = rotation @ params @ rotation.T
    representation = graph.representation_matrix(rotation)

    precision = spd_map.precision(params)
    transformed_precision = spd_map.precision(transformed_params)
    expected_precision = representation @ precision @ representation.T
    assert torch.allclose(
        transformed_precision, expected_precision, atol=2e-4, rtol=2e-4
    )

    covariance = spd_map(params)
    transformed_covariance = spd_map(transformed_params)
    expected_covariance = representation @ covariance @ representation.T
    assert torch.allclose(
        transformed_covariance, expected_covariance, atol=2e-4, rtol=2e-4
    )


def test_precision_coordinate_gaussian_and_student_t_objectives():
    graph = _graph()
    spd_map = GraphStructuredPrecisionMap(graph)
    params = _symmetric_params()
    mean = torch.randn(2, graph.output_dim)
    target = torch.randn_like(mean)
    residual = target - mean
    precision = spd_map.precision(params)
    quad = torch.einsum("bi,bij,bj->b", residual, precision, residual)
    logdet_precision = torch.linalg.slogdet(precision).logabsdet

    gaussian, _ = GaussianNLL()(mean, params, target, spd_map)
    expected_gaussian = (
        0.5 * graph.output_dim * math.log(2.0 * math.pi)
        - 0.5 * logdet_precision
        + 0.5 * quad
    ).mean()
    assert torch.allclose(gaussian, expected_gaussian, atol=2e-5, rtol=2e-5)

    student, _ = StudentTNLL(nu=7.0)(mean, params, target, spd_map)
    assert torch.isfinite(student)


def test_graph_precision_gradients_and_sampling():
    graph = _graph()
    spd_map = GraphStructuredPrecisionMap(graph)
    params = _symmetric_params().requires_grad_()
    residual = torch.randn(2, graph.output_dim)
    loss = (
        spd_map.logdet(params).mean()
        + spd_map.precision_action(params, residual).mean()
    )
    loss.backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()

    samples = spd_map.sample(torch.zeros(2, graph.output_dim), params.detach(), 5)
    assert samples.shape == (2, graph.output_dim, 5)
    assert torch.isfinite(samples).all()


def test_compiler_selects_graph_precision_for_itop_budget():
    graph = EquivariantOutputGraph(
        num_nodes=15,
        edges=tuple((index, index + 1) for index in range(14)),
        node_irrep="1o",
    )
    seed = o3.Irreps("8x0e + 4x1o + 4x2e")
    feature = FeatureSpec.from_irreps(seed, scope="node")
    compilation = plan_readout(
        feature,
        output=graph.output_irreps,
        covariance=FirstFeasible(
            192,
            (
                FullCovariance(),
                GraphPrecision(graph),
                LowRankCovariance(8),
                IsotypicBlockCovariance(),
            ),
        ),
    ).compilation
    assert compilation.covariance_mode == "graph"
    assert compilation.output_spec.dim == 45
    assert compilation.output_spec.symmetric_square().operator_dim == 1035
    assert compilation.covariance_parameter_count == 174
    assert compilation.canonical_plan.depth == 1
    assert compilation.active_plan.depth == 0


def test_compiled_graph_head_and_precision_are_equivariant():
    torch.manual_seed(5)
    graph = _graph()
    seed = o3.Irreps("8x0e + 4x1o + 4x2e")
    compilation = plan_readout(
        FeatureSpec.from_irreps(seed, scope="node"),
        output=graph.output_irreps,
        covariance=GraphPrecision(graph),
    ).compilation
    head = compilation.build_head()
    spd_map = compilation.build_spd_map()
    features = seed.randn(5, -1)
    batch = torch.tensor([0, 0, 0, 1, 1])
    rotation = o3.rand_matrix()
    transformed_features = features @ seed.D_from_matrix(rotation).T

    mean, params = head(features, batch)
    transformed_mean, transformed_params = head(transformed_features, batch)
    output_representation = graph.representation_matrix(rotation)
    expected_mean = mean @ output_representation.T
    assert torch.allclose(transformed_mean, expected_mean, atol=2e-4, rtol=2e-4)

    parameter_representation = compilation.operator_family.parameter_irreps
    expected_params = params @ parameter_representation.D_from_matrix(rotation).T
    assert torch.allclose(transformed_params, expected_params, atol=2e-4, rtol=2e-4)
    precision = spd_map.precision(params)
    transformed_precision = spd_map.precision(transformed_params)
    expected_precision = output_representation @ precision @ output_representation.T
    assert torch.allclose(
        transformed_precision, expected_precision, atol=3e-4, rtol=3e-4
    )
