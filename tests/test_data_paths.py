from pathlib import Path

import pytest

from data.paths import DATA_ROOT_ENV, data_root, dataset_dir


def test_configured_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    assert data_root() == tmp_path
    assert dataset_dir(None, "ITOP") == tmp_path / "ITOP"


def test_explicit_dataset_directory_does_not_require_environment(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    explicit = Path("custom/dataset")
    assert dataset_dir(explicit, "ITOP") == explicit


def test_missing_data_root_has_actionable_error(monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match=DATA_ROOT_ENV):
        data_root()
