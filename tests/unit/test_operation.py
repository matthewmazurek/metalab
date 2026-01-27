"""Tests for metalab.operation module."""

import pytest

import metalab
from metalab.operation import OperationWrapper, operation


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

    def test_custom_name(self):
        """Operation can have a custom name."""

        @operation(name="custom_name")
        def my_op(capture):
            pass

        assert my_op.name == "custom_name"

    def test_default_name_from_function(self):
        """Operation defaults to function name."""

        @operation
        def my_operation(capture):
            pass

        assert my_operation.name == "my_operation"

    def test_bare_decorator(self):
        """Operation works as bare decorator without parentheses."""

        @operation
        def bare_op(capture):
            pass

        assert isinstance(bare_op, OperationWrapper)
        assert bare_op.name == "bare_op"

    def test_empty_parentheses(self):
        """Operation works with empty parentheses."""

        @operation()
        def empty_parens_op(capture):
            pass

        assert isinstance(empty_parens_op, OperationWrapper)
        assert empty_parens_op.name == "empty_parens_op"

    def test_code_hash_is_stable(self):
        """Code hash is deterministic."""

        @operation
        def my_op(capture):
            x = 1 + 1
            return x

        hash1 = my_op.code_hash
        hash2 = my_op.code_hash

        assert hash1 == hash2
        assert len(hash1) == 16  # Truncated to 16 chars

    def test_repr(self):
        """Operation has readable repr."""

        @operation(name="test_op")
        def my_op(capture):
            pass

        assert repr(my_op) == "Operation(test_op)"
