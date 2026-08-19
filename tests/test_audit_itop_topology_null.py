from scripts.audit_itop_topology_null import edge_overlap


def test_edge_overlap_is_undirected_and_counts_shared_edges() -> None:
    reference = [[0, 1], [1, 2], [2, 3]]
    candidate = [[1, 0], [2, 4], [3, 2]]

    assert edge_overlap(reference, candidate) == 2
