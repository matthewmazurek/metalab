"""
Tests for the Capture interface.

Tests cover:
- capture.data() for structured results
- Stepped metrics conversion to data entries
- capture.metric() and capture.log_metrics()
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metalab.capture.capture import Capture
from metalab.store import FileStore


class TestCaptureData:
    """Tests for capture.data() structured results."""

    def test_data_stores_list(self, tmp_path: Path) -> None:
        """capture.data() should store list data."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.data("scores", [0.1, 0.2, 0.3])

        # Check results were captured
        assert len(capture.results) == 1
        assert capture.results[0]["name"] == "scores"
        assert capture.results[0]["data"] == [0.1, 0.2, 0.3]
        assert capture.results[0]["dtype"] is None
        assert capture.results[0]["shape"] is None

        capture.finalize()

    def test_data_stores_dict(self, tmp_path: Path) -> None:
        """capture.data() should store dict data."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        data = {"gene_a": 0.8, "gene_b": 0.6}
        capture.data("gene_scores", data)

        assert len(capture.results) == 1
        assert capture.results[0]["name"] == "gene_scores"
        assert capture.results[0]["data"] == data

        capture.finalize()

    def test_data_stores_numpy_array(self, tmp_path: Path) -> None:
        """capture.data() should store numpy arrays with metadata."""
        pytest.importorskip("numpy")
        import numpy as np

        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        capture.data("transition_matrix", matrix)

        assert len(capture.results) == 1
        result = capture.results[0]
        assert result["name"] == "transition_matrix"
        assert result["data"] == [[1.0, 2.0], [3.0, 4.0]]
        assert result["dtype"] == "float64"
        assert result["shape"] == [2, 2]

        capture.finalize()

    def test_data_with_metadata(self, tmp_path: Path) -> None:
        """capture.data() should store optional metadata."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.data("embeddings", [1, 2, 3], metadata={"dim": 3})

        assert capture.results[0]["metadata"] == {"dim": 3}

        capture.finalize()

    def test_multiple_data_entries(self, tmp_path: Path) -> None:
        """Multiple capture.data() calls should accumulate."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.data("data_a", [1, 2])
        capture.data("data_b", {"key": "value"})
        capture.data("data_c", [[1, 2], [3, 4]])

        assert len(capture.results) == 3
        assert capture.results[0]["name"] == "data_a"
        assert capture.results[1]["name"] == "data_b"
        assert capture.results[2]["name"] == "data_c"

        capture.finalize()


class TestSteppedMetricsConversion:
    """Tests for stepped metrics conversion to data entries."""

    def test_stepped_metrics_converted_to_data(self, tmp_path: Path) -> None:
        """Stepped metrics should be converted to data entries at finalize."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        # Log some stepped metrics
        capture.metric("loss", 0.9, step=0)
        capture.metric("loss", 0.5, step=10)
        capture.metric("loss", 0.1, step=20)

        # Before finalize, results should be empty
        assert len(capture.results) == 0
        assert len(capture._stepped_metrics) == 3

        # After finalize, stepped metrics should be converted to data
        summary = capture.finalize()

        # Should have one data entry for "loss"
        assert len(capture.results) == 1
        result = capture.results[0]
        assert result["name"] == "loss"
        assert result["data"] == [(0, 0.9), (10, 0.5), (20, 0.1)]
        assert result["metadata"] == {"type": "stepped_metric"}
        assert result["shape"] == [3, 2]

    def test_multiple_stepped_metrics_grouped(self, tmp_path: Path) -> None:
        """Multiple stepped metric names should create separate data entries."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        # Log stepped metrics for two different names
        capture.metric("loss", 0.9, step=0)
        capture.metric("accuracy", 0.1, step=0)
        capture.metric("loss", 0.5, step=10)
        capture.metric("accuracy", 0.5, step=10)

        capture.finalize()

        # Should have two data entries
        assert len(capture.results) == 2
        names = {r["name"] for r in capture.results}
        assert names == {"loss", "accuracy"}

    def test_stepped_metrics_sorted_by_step(self, tmp_path: Path) -> None:
        """Stepped metrics should be sorted by step number."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        # Log out of order
        capture.metric("loss", 0.1, step=20)
        capture.metric("loss", 0.9, step=0)
        capture.metric("loss", 0.5, step=10)

        capture.finalize()

        result = capture.results[0]
        # Should be sorted by step
        assert result["data"] == [(0, 0.9), (10, 0.5), (20, 0.1)]


class TestCaptureMetrics:
    """Tests for capture.metric() and capture.log_metrics()."""

    def test_metric_stores_scalar(self, tmp_path: Path) -> None:
        """capture.metric() should store scalar values."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.metric("accuracy", 0.95)
        capture.metric("loss", 0.05)

        assert capture.metrics == {"accuracy": 0.95, "loss": 0.05}

        capture.finalize()

    def test_log_metrics_batch(self, tmp_path: Path) -> None:
        """capture.log_metrics() should store multiple metrics."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.log_metrics({"a": 1, "b": 2, "c": 3})

        assert capture.metrics == {"a": 1, "b": 2, "c": 3}

        capture.finalize()

    def test_log_metrics_with_step(self, tmp_path: Path) -> None:
        """capture.log_metrics() with step should use stepped metrics."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.log_metrics({"loss": 0.9, "accuracy": 0.1}, step=0)

        # Scalar metrics should be empty
        assert capture.metrics == {}
        # Stepped metrics should have entries
        assert len(capture._stepped_metrics) == 2

        capture.finalize()


class TestCaptureSummary:
    """Tests for capture.finalize() and _get_summary()."""

    def test_summary_includes_results(self, tmp_path: Path) -> None:
        """Summary should include results in addition to metrics and artifacts."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.metric("accuracy", 0.95)
        capture.data("matrix", [[1, 2], [3, 4]])

        summary = capture.finalize()

        assert "metrics" in summary
        assert "results" in summary
        assert "artifacts" in summary
        assert "stepped_metrics" in summary

        assert summary["metrics"] == {"accuracy": 0.95}
        assert len(summary["results"]) == 1
        assert summary["results"][0]["name"] == "matrix"

    def test_finalize_idempotent(self, tmp_path: Path) -> None:
        """finalize() should be idempotent."""
        store = FileStore(tmp_path)
        capture = Capture(store=store, run_id="test_run_123")

        capture.metric("accuracy", 0.95)

        summary1 = capture.finalize()
        summary2 = capture.finalize()

        assert summary1 == summary2
