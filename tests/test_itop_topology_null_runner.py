import json
from pathlib import Path

import pytest

from scripts.run_itop_topology_null import _load_manifest


def test_topology_null_runner_rejects_outcome_filtered_manifest(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"outcome_filtered": True, "records": [{"index": 0}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outcome-filtered"):
        _load_manifest(path)


def test_topology_null_runner_requires_contiguous_manifest_indices(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"outcome_filtered": False, "records": [{"index": 1}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contiguous"):
        _load_manifest(path)
