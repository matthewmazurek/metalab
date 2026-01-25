"""Tests for parameter generation."""

from __future__ import annotations

import pytest

from metalab.params import (
    choice,
    grid,
    loguniform_int,
    manual,
    random,
    uniform,
    with_resolver,
)


class TestGrid:
    """Tests for GridSource."""

    def test_grid_single_param(self):
        """Single parameter grid."""
        source = grid(x=[1, 2, 3])
        cases = list(source)
        assert len(cases) == 3
        assert [c.params["x"] for c in cases] == [1, 2, 3]

    def test_grid_multiple_params(self):
        """Multiple parameters: Cartesian product."""
        source = grid(a=[1, 2], b=["x", "y"])
        cases = list(source)
        assert len(cases) == 4
        params_set = {(c.params["a"], c.params["b"]) for c in cases}
        assert params_set == {(1, "x"), (1, "y"), (2, "x"), (2, "y")}

    def test_grid_empty(self):
        """Empty grid yields single empty case."""
        source = grid()
        cases = list(source)
        assert len(cases) == 1
        assert cases[0].params == {}

    def test_grid_case_ids_unique(self):
        """Each case should have a unique ID."""
        source = grid(x=[1, 2, 3])
        cases = list(source)
        ids = [c.case_id for c in cases]
        assert len(ids) == len(set(ids))

    def test_grid_len(self):
        """len() should return product of value counts."""
        source = grid(a=[1, 2], b=[3, 4, 5])
        assert len(source) == 6


class TestRandom:
    """Tests for RandomSource."""

    def test_random_replayable(self):
        """Same seed should produce same samples."""
        space = {"x": uniform(0, 1), "y": uniform(0, 1)}
        source1 = random(space, n_trials=5, seed=42)
        source2 = random(space, n_trials=5, seed=42)

        cases1 = list(source1)
        cases2 = list(source2)

        for c1, c2 in zip(cases1, cases2):
            assert c1.params == c2.params

    def test_random_different_seeds(self):
        """Different seeds should produce different samples."""
        space = {"x": uniform(0, 1)}
        source1 = random(space, n_trials=10, seed=42)
        source2 = random(space, n_trials=10, seed=123)

        cases1 = list(source1)
        cases2 = list(source2)

        # Very unlikely to be identical
        values1 = [c.params["x"] for c in cases1]
        values2 = [c.params["x"] for c in cases2]
        assert values1 != values2

    def test_random_uniform(self):
        """Uniform distribution should be within bounds."""
        source = random({"x": uniform(10, 20)}, n_trials=100, seed=42)
        for case in source:
            assert 10 <= case.params["x"] <= 20

    def test_random_loguniform_int(self):
        """LogUniform int should produce integers in range."""
        source = random({"x": loguniform_int(1, 1000)}, n_trials=50, seed=42)
        for case in source:
            assert isinstance(case.params["x"], int)
            assert 1 <= case.params["x"] <= 1000

    def test_random_choice(self):
        """Choice should pick from options."""
        options = ["a", "b", "c"]
        source = random({"x": choice(options)}, n_trials=50, seed=42)
        for case in source:
            assert case.params["x"] in options

    def test_random_len(self):
        """len() should return n_trials."""
        source = random({"x": uniform(0, 1)}, n_trials=20, seed=42)
        assert len(source) == 20


class TestManual:
    """Tests for ManualSource."""

    def test_manual_basic(self):
        """Manual source yields exact cases."""
        cases_input = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
        ]
        source = manual(cases_input)
        cases = list(source)
        assert len(cases) == 2
        assert cases[0].params == {"a": 1, "b": "x"}
        assert cases[1].params == {"a": 2, "b": "y"}

    def test_manual_with_tags(self):
        """Tags should be applied to all cases."""
        source = manual([{"x": 1}], tags=["test"])
        cases = list(source)
        assert "test" in cases[0].tags

    def test_manual_len(self):
        """len() should return number of cases."""
        source = manual([{"a": i} for i in range(5)])
        assert len(source) == 5


class TestResolver:
    """Tests for parameter resolution."""

    def test_with_resolver_basic(self):
        """Resolver should transform params."""

        def double_x(ctx, params):
            return {"x": params["x"] * 2}

        source = with_resolver(grid(x=[1, 2, 3]), resolver=double_x)
        cases = list(source)
        assert [c.params["x"] for c in cases] == [2, 4, 6]

    def test_with_resolver_case_id_updated(self):
        """Resolved params should have different case_id."""
        original = grid(x=[1])
        resolved = with_resolver(original, resolver=lambda ctx, p: {"x": p["x"] + 1})

        orig_cases = list(original)
        res_cases = list(resolved)

        # Different params means different case_id
        assert orig_cases[0].case_id != res_cases[0].case_id

    def test_with_resolver_context_meta(self):
        """Resolver should receive context metadata."""
        received_ctx = {}

        def capture_ctx(ctx, params):
            received_ctx.update(ctx)
            return params

        source = with_resolver(
            grid(x=[1]),
            resolver=capture_ctx,
            context_meta={"dataset": "test"},
        )
        list(source)  # Consume to trigger resolver

        assert received_ctx == {"dataset": "test"}
