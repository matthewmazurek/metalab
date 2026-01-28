"""Unit tests for derived metrics module."""

from __future__ import annotations

import pytest

from metalab.derived import (
    DerivedMetricFn,
    compute_derived_for_run,
    get_func_ref,
    import_derived_metric,
)
from metalab.types import Metric


# Test functions for import/export
def sample_derived_metric(run) -> dict[str, Metric]:
    """Sample derived metric function for testing."""
    return {"test_metric": 42}


def metric_with_params(run) -> dict[str, Metric]:
    """Derived metric that uses params."""
    lr = run.params.get("lr", 1.0)
    return {"scaled_value": 100 * lr}


class TestImportDerivedMetric:
    """Tests for import_derived_metric function."""

    def test_import_valid_reference(self):
        """Import a function by reference."""
        ref = "tests.unit.test_derived:sample_derived_metric"
        func = import_derived_metric(ref)
        assert callable(func)
        assert func.__name__ == "sample_derived_metric"

    def test_import_empty_reference_raises(self):
        """Empty reference should raise ValueError."""
        with pytest.raises(ValueError, match="Empty reference"):
            import_derived_metric("")

    def test_import_invalid_format_raises(self):
        """Invalid format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid reference format"):
            import_derived_metric("no_colon_here")

    def test_import_nonexistent_module_raises(self):
        """Nonexistent module should raise ImportError."""
        with pytest.raises(ImportError):
            import_derived_metric("nonexistent.module:func")

    def test_import_nonexistent_function_raises(self):
        """Nonexistent function should raise AttributeError."""
        with pytest.raises(AttributeError):
            import_derived_metric("tests.unit.test_derived:nonexistent_func")


class TestGetFuncRef:
    """Tests for get_func_ref function."""

    def test_get_ref_for_module_function(self):
        """Get reference for a module-level function."""
        ref = get_func_ref(sample_derived_metric)
        assert ref == "tests.unit.test_derived:sample_derived_metric"

    def test_lambda_raises_error(self):
        """Lambda functions should raise ValueError."""
        with pytest.raises(ValueError, match="not lambdas"):
            get_func_ref(lambda run: {"x": 1})

    def test_local_function_raises_error(self):
        """Local functions should raise ValueError."""

        def local_func(run):
            return {"x": 1}

        with pytest.raises(ValueError, match="not lambdas or local functions"):
            get_func_ref(local_func)

    def test_round_trip(self):
        """Import reference from get_func_ref should return same function."""
        ref = get_func_ref(sample_derived_metric)
        imported = import_derived_metric(ref)
        assert imported is sample_derived_metric


class TestComputeDerivedForRun:
    """Tests for compute_derived_for_run function."""

    def test_empty_metrics_list(self, mock_run):
        """Empty metrics list returns empty dict."""
        result = compute_derived_for_run(mock_run, [])
        assert result == {}

    def test_single_metric(self, mock_run):
        """Single metric function works."""
        result = compute_derived_for_run(mock_run, [sample_derived_metric])
        assert result == {"test_metric": 42}

    def test_multiple_metrics_merged(self, mock_run):
        """Multiple metrics are merged."""

        def metric1(run) -> dict[str, Metric]:
            return {"a": 1}

        def metric2(run) -> dict[str, Metric]:
            return {"b": 2}

        result = compute_derived_for_run(mock_run, [metric1, metric2])
        assert result == {"a": 1, "b": 2}

    def test_failing_metric_logged_and_skipped(self, mock_run, caplog):
        """Failing metric is logged but doesn't stop computation."""

        def failing_metric(run) -> dict[str, Metric]:
            raise RuntimeError("Test error")

        def working_metric(run) -> dict[str, Metric]:
            return {"works": True}

        result = compute_derived_for_run(mock_run, [failing_metric, working_metric])
        assert result == {"works": True}
        assert "Test error" in caplog.text

    def test_metric_accesses_params(self, mock_run):
        """Metric can access run params."""
        mock_run._params = {"lr": 0.5}
        result = compute_derived_for_run(mock_run, [metric_with_params])
        assert result == {"scaled_value": 50.0}


# Fixtures


class MockRun:
    """Mock Run object for testing."""

    def __init__(self):
        self.run_id = "test_run_123"
        self._params = {}
        self._metrics = {}
        self._artifacts = {}

    @property
    def params(self):
        return self._params

    @property
    def metrics(self):
        return self._metrics

    def artifact(self, name: str):
        if name not in self._artifacts:
            raise FileNotFoundError(f"Artifact '{name}' not found")
        return self._artifacts[name]


@pytest.fixture
def mock_run():
    """Create a mock Run object."""
    return MockRun()
