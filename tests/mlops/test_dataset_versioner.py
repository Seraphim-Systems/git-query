"""Unit tests for DatasetVersioner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.recommender.mlops.dataset_versioner import DatasetVersioner


def _make_df(n: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "name": [f"repo_{i}" for i in range(n)],
            "stars": rng.integers(0, 10000, n).tolist(),
            "language": rng.choice(["python", "go", "rust"], n).tolist(),
        }
    )


class TestComputeVersion:
    def test_same_df_same_version(self):
        versioner = DatasetVersioner()
        df = _make_df()
        assert versioner.compute_version(df) == versioner.compute_version(df)

    def test_different_df_different_version(self):
        versioner = DatasetVersioner()
        df1 = _make_df(seed=0)
        df2 = _make_df(seed=99)
        assert versioner.compute_version(df1) != versioner.compute_version(df2)

    def test_version_changes_when_row_added(self):
        versioner = DatasetVersioner()
        df = _make_df(n=10)
        extra_row = pd.DataFrame([{"name": "new_repo", "stars": 9999, "language": "java"}])
        df_extended = pd.concat([df, extra_row], ignore_index=True)
        assert versioner.compute_version(df) != versioner.compute_version(df_extended)

    def test_version_is_short_hex_string(self):
        versioner = DatasetVersioner()
        version = versioner.compute_version(_make_df())
        assert len(version) == 8
        assert all(c in "0123456789abcdef" for c in version)

    def test_custom_digest_length(self):
        versioner = DatasetVersioner(digest_length=12)
        version = versioner.compute_version(_make_df())
        assert len(version) == 12


class TestSaveLoadMetadata:
    def test_save_load_roundtrip(self, tmp_path):
        versioner = DatasetVersioner()
        df = _make_df()
        version = versioner.compute_version(df)

        path = tmp_path / "meta.json"
        versioner.save_version_metadata(version, df, path)
        meta = versioner.load_version_metadata(path)

        assert meta["version"] == version
        assert meta["row_count"] == len(df)
        assert meta["col_count"] == len(df.columns)
        assert "timestamp" in meta
        assert "feature_cols" in meta

    def test_save_creates_parent_dirs(self, tmp_path):
        versioner = DatasetVersioner()
        df = _make_df()
        version = versioner.compute_version(df)

        deep_path = tmp_path / "a" / "b" / "c" / "meta.json"
        versioner.save_version_metadata(version, df, deep_path)
        assert deep_path.exists()

    def test_extra_metadata_saved(self, tmp_path):
        versioner = DatasetVersioner()
        df = _make_df()
        version = versioner.compute_version(df)
        path = tmp_path / "meta.json"

        versioner.save_version_metadata(version, df, path, extra={"experiment": "run_42"})
        meta = versioner.load_version_metadata(path)
        assert meta["experiment"] == "run_42"

    def test_feature_cols_sorted(self, tmp_path):
        versioner = DatasetVersioner()
        df = _make_df()
        version = versioner.compute_version(df)
        path = tmp_path / "meta.json"
        versioner.save_version_metadata(version, df, path)
        meta = versioner.load_version_metadata(path)
        assert meta["feature_cols"] == sorted(df.columns.tolist())


class TestVersionsMatch:
    def test_identical_versions_match(self):
        versioner = DatasetVersioner()
        df = _make_df()
        v = versioner.compute_version(df)
        assert versioner.versions_match(v, v)

    def test_different_versions_do_not_match(self):
        versioner = DatasetVersioner()
        v1 = versioner.compute_version(_make_df(seed=1))
        v2 = versioner.compute_version(_make_df(seed=2))
        assert not versioner.versions_match(v1, v2)
