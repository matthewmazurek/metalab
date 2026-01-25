"""Tests for metalab.operation module."""

import pytest

import metalab
from metalab.operation import operation, OperationWrapper


class TestSignatureInspection:
    """Test that operations can use flexible signatures."""

    def test_minimal_signature(self):
        """Operation can have just capture parameter."""
        @operation(name="minimal")
        def minimal_op(capture):
            capture.metric("test", 1)

        assert isinstance(minimal_op, OperationWrapper)
        assert minimal_op.name == "minimal"

    def test_partial_signature(self):
        """Operation can have subset of parameters."""
        @operation(name="partial")
        def partial_op(params, seeds, capture):
            pass

        assert isinstance(partial_op, OperationWrapper)

    def test_full_signature(self):
        """Operation can have all parameters (backward compatible)."""
        @operation(name="full")
        def full_op(context, params, seeds, runtime, capture):
            pass

        assert isinstance(full_op, OperationWrapper)

    def test_invalid_parameter_raises(self):
        """Operation with unknown parameter name raises ValueError."""
        with pytest.raises(ValueError, match="invalid parameter"):
            @operation(name="bad")
            def bad_op(params, unknown_param, capture):
                pass

    def test_only_injects_requested_params(self):
        """Operation only receives the parameters it requests."""
        received = {}

        @operation(name="check_injection")
        def check_op(params, capture):
            received["params"] = params
            received["capture"] = capture
            # Should not receive context, seeds, or runtime

        # Create mock objects
        from unittest.mock import MagicMock

        mock_context = MagicMock(name="context")
        mock_params = {"key": "value"}
        mock_seeds = MagicMock(name="seeds")
        mock_runtime = MagicMock(name="runtime")
        mock_capture = MagicMock(name="capture")

        # Run the operation
        check_op.run(
            context=mock_context,
            params=mock_params,
            seeds=mock_seeds,
            runtime=mock_runtime,
            capture=mock_capture,
        )

        # Verify only requested params were received
        assert received["params"] is mock_params
        assert received["capture"] is mock_capture
        assert "context" not in received
        assert "seeds" not in received
        assert "runtime" not in received


class TestOperationMetadata:
    """Test operation metadata and properties."""

    def test_name_and_version(self):
        """Operation has correct name and version."""
        @operation(name="test_op", version="1.2.3")
        def my_op(capture):
            pass

        assert my_op.name == "test_op"
        assert my_op.version == "1.2.3"

    def test_default_version(self):
        """Operation defaults to version 0.0.0."""
        @operation(name="test_op")
        def my_op(capture):
            pass

        assert my_op.version == "0.0.0"

    def test_code_hash_is_stable(self):
        """Code hash is deterministic."""
        @operation(name="test_op")
        def my_op(capture):
            x = 1 + 1
            return x

        hash1 = my_op.code_hash
        hash2 = my_op.code_hash

        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_repr(self):
        """Operation has readable repr."""
        @operation(name="test_op", version="1.0")
        def my_op(capture):
            pass

        assert repr(my_op) == "Operation(test_op:1.0)"
