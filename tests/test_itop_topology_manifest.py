import pytest

from scripts.itop_topology_manifest import (
    generate_degree_matched_tree,
    topology_manifest,
)


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


def test_degree_matched_tree_rejects_non_tree_reference():
    with pytest.raises(ValueError, match="tree"):
        generate_degree_matched_tree(((0, 1), (1, 2)), num_nodes=4, seed=3)
