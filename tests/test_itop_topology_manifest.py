import json
from types import SimpleNamespace

import pytest

from data.itop_dataset import ITOP_OUTPUT_GRAPH, ITOP_SKELETON_EDGES
from scripts.itop_topology_manifest import (
    generate_degree_matched_tree,
    topology_manifest,
)
from scripts.train_itop import _topology_graph


def test_degree_matched_tree_preserves_true_skeleton_degree_sequence():
    true_edges = ((0, 1), (1, 2), (1, 3), (3, 4))
    generated = generate_degree_matched_tree(true_edges, num_nodes=5, seed=17)
    assert len(generated) == 4
    degrees = [0] * 5
    for source, target in generated:
        degrees[source] += 1
        degrees[target] += 1
    expected = [0] * 5
    for source, target in true_edges:
        expected[source] += 1
        expected[target] += 1
    assert sorted(degrees) == sorted(expected)
    assert len({tuple(sorted(edge)) for edge in generated}) == 4


def test_topology_manifest_is_seeded_and_unfiltered():
    true_edges = ((0, 1), (1, 2), (1, 3), (3, 4))
    first = topology_manifest(true_edges, num_nodes=5, count=3, seed=91)
    second = topology_manifest(true_edges, num_nodes=5, count=3, seed=91)
    assert first == second
    assert [item["topology_seed"] for item in first] == [91, 92, 93]
    assert all(item["degree_sequence"] == [1, 1, 1, 2, 3] for item in first)
    assert all(
        item["sampler"] == "uniform_labeled_degree_sequence_prufer"
        for item in first
    )
    assert any(item["edges"] != item["reference_edges"] for item in first)


def test_prufer_sampler_is_deterministic_and_preserves_labeled_degrees():
    true_edges = ((0, 1), (1, 2), (1, 3), (3, 4))
    first = generate_degree_matched_tree(true_edges, num_nodes=5, seed=17)
    second = generate_degree_matched_tree(true_edges, num_nodes=5, seed=17)
    assert first == second
    expected = [0] * 5
    for source, target in true_edges:
        expected[source] += 1
        expected[target] += 1
    actual = [0] * 5
    for source, target in first:
        actual[source] += 1
        actual[target] += 1
    assert actual == expected


def test_degree_matched_tree_rejects_non_tree_reference():
    with pytest.raises(ValueError, match="tree"):
        generate_degree_matched_tree(((0, 1), (1, 2)), num_nodes=4, seed=3)


def test_training_resolves_manifest_topology_without_outcome_selection(tmp_path):
    records = topology_manifest(ITOP_SKELETON_EDGES, num_nodes=15, count=1, seed=2026)
    manifest = {
        "outcome_filtered": False,
        "records": records,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    graph = _topology_graph(
        SimpleNamespace(
            model="shuffled_graph_student_t",
            topology_manifest=str(path),
            topology_index=0,
        )
    )

    assert graph.edges == tuple(tuple(edge) for edge in records[0]["edges"])
    assert graph.edges != ITOP_OUTPUT_GRAPH.edges
