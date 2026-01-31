"""Tests for parameter generation."""

from __future__ import annotations

import pytest

from metalab.params import (
    choice,
    grid,
    loguniform,
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


class TestRandomIndexing:
    """Tests for RandomSource indexing (SLURM array support)."""

    def test_random_getitem_matches_iteration(self):
        """source[i] should equal list(source)[i] for all indices."""
        space = {"x": uniform(0, 1), "y": choice(["a", "b", "c"])}
        source = random(space, n_trials=20, seed=42)
        cases_list = list(source)

        for i in range(len(source)):
            indexed = source[i]
            iterated = cases_list[i]
            assert indexed.params == iterated.params
            assert indexed.case_id == iterated.case_id
            assert indexed.tags == iterated.tags

    def test_random_getitem_deterministic(self):
        """Accessing same index multiple times returns identical results."""
        source = random({"x": uniform(0, 100)}, n_trials=10, seed=42)

        # Access indices in random order, multiple times
        for _ in range(3):
            val_5 = source[5].params["x"]
            val_2 = source[2].params["x"]
            val_9 = source[9].params["x"]

            # All accesses to same index should be identical
            assert source[5].params["x"] == val_5
            assert source[2].params["x"] == val_2
            assert source[9].params["x"] == val_9

    def test_random_getitem_independence(self):
        """Accessing one index doesn't affect other indices."""
        source = random({"x": uniform(0, 1)}, n_trials=10, seed=42)

        # Get value at index 5 without accessing anything else
        val_5_first = source[5].params["x"]

        # Access many other indices
        for i in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
            _ = source[i]

        # Index 5 should still return the same value
        val_5_after = source[5].params["x"]
        assert val_5_first == val_5_after

    def test_random_getitem_negative_index(self):
        """Negative indexing should work."""
        source = random({"x": uniform(0, 1)}, n_trials=5, seed=42)
        assert source[-1].params == source[4].params
        assert source[-2].params == source[3].params
        assert source[-5].params == source[0].params

    def test_random_getitem_out_of_bounds(self):
        """Out of bounds should raise IndexError."""
        source = random({"x": uniform(0, 1)}, n_trials=5, seed=42)
        with pytest.raises(IndexError):
            source[5]
        with pytest.raises(IndexError):
            source[-6]

    def test_random_different_seeds_different_values(self):
        """Different seeds should produce different indexed values."""
        space = {"x": uniform(0, 1)}
        source1 = random(space, n_trials=10, seed=42)
        source2 = random(space, n_trials=10, seed=123)

        # Very unlikely to have same value at same index with different seeds
        assert source1[0].params["x"] != source2[0].params["x"]
        assert source1[5].params["x"] != source2[5].params["x"]

    def test_random_manifest_roundtrip_preserves_ordering(self):
        """Serialization and deserialization preserves indexing order."""
        from metalab.manifest import deserialize_param_source

        space = {
            "lr": loguniform_int(1, 1000),
            "dropout": uniform(0.1, 0.5),
            "opt": choice(["adam", "sgd", "rmsprop"]),
        }
        source = random(space, n_trials=50, seed=42)
        manifest = source.to_manifest_dict()
        restored = deserialize_param_source(manifest)

        # Verify same length
        assert len(restored) == len(source)

        # Verify all indices match (check a sample of indices for speed)
        for i in [0, 1, 10, 25, 49]:
            assert restored[i].params == source[i].params
            assert restored[i].case_id == source[i].case_id

    def test_random_all_distributions_roundtrip(self):
        """All distribution types should serialize/deserialize correctly."""
        from metalab.params.random import randint
        from metalab.manifest import deserialize_param_source

        space = {
            "a": uniform(0.0, 1.0),
            "b": loguniform(0.001, 1.0),
            "c": loguniform_int(1, 100),
            "d": randint(0, 10),
            "e": choice([True, False, None]),
        }
        source = random(space, n_trials=20, seed=99)
        manifest = source.to_manifest_dict()
        restored = deserialize_param_source(manifest)

        # Check that restored source produces same results
        for i in range(len(source)):
            assert restored[i].params == source[i].params


class TestGridIndexing:
    """Tests for GridSource indexing (SLURM array support)."""

    def test_grid_getitem_matches_iteration(self):
        """source[i] should equal list(source)[i] for all indices."""
        source = grid(a=[1, 2, 3], b=["x", "y"], c=[True, False])
        cases_list = list(source)

        for i in range(len(source)):
            indexed = source[i]
            iterated = cases_list[i]
            assert indexed.params == iterated.params
            assert indexed.case_id == iterated.case_id

    def test_grid_getitem_single_param(self):
        """Single parameter grid indexing."""
        source = grid(x=[10, 20, 30])
        assert source[0].params == {"x": 10}
        assert source[1].params == {"x": 20}
        assert source[2].params == {"x": 30}

    def test_grid_getitem_empty(self):
        """Empty grid indexing."""
        source = grid()
        assert source[0].params == {}
        with pytest.raises(IndexError):
            source[1]

    def test_grid_getitem_negative_index(self):
        """Negative indexing should work."""
        source = grid(x=[1, 2, 3])
        assert source[-1].params == {"x": 3}
        assert source[-2].params == {"x": 2}
        assert source[-3].params == {"x": 1}

    def test_grid_getitem_out_of_bounds(self):
        """Out of bounds should raise IndexError."""
        source = grid(x=[1, 2])
        with pytest.raises(IndexError):
            source[2]
        with pytest.raises(IndexError):
            source[-3]

    def test_grid_manifest_roundtrip_preserves_ordering(self):
        """Serialization and deserialization preserves indexing order."""
        from metalab.manifest import deserialize_param_source

        source = grid(alpha=[0.1, 0.01], beta=[1, 2, 3], gamma=["a", "b"])
        manifest = source.to_manifest_dict()
        restored = deserialize_param_source(manifest)

        # Verify same length
        assert len(restored) == len(source)

        # Verify all indices match
        for i in range(len(source)):
            assert restored[i].params == source[i].params
            assert restored[i].case_id == source[i].case_id


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


class TestManualIndexing:
    """Tests for ManualSource indexing (SLURM array support)."""

    def test_manual_getitem_matches_iteration(self):
        """source[i] should equal list(source)[i] for all indices."""
        cases_input = [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
        ]
        source = manual(cases_input)
        cases_list = list(source)

        for i in range(len(source)):
            indexed = source[i]
            iterated = cases_list[i]
            assert indexed.params == iterated.params
            assert indexed.case_id == iterated.case_id

    def test_manual_getitem_negative_index(self):
        """Negative indexing should work."""
        source = manual([{"x": 1}, {"x": 2}, {"x": 3}])
        assert source[-1].params == {"x": 3}
        assert source[-2].params == {"x": 2}

    def test_manual_getitem_out_of_bounds(self):
        """Out of bounds should raise IndexError."""
        source = manual([{"x": 1}])
        with pytest.raises(IndexError):
            source[1]

    def test_manual_manifest_roundtrip_preserves_ordering(self):
        """Serialization and deserialization preserves indexing order."""
        from metalab.manifest import deserialize_param_source

        cases_input = [
            {"lr": 0.01, "batch": 32},
            {"lr": 0.1, "batch": 64},
            {"lr": 0.001, "batch": 128},
        ]
        source = manual(cases_input, tags=["test"])
        manifest = source.to_manifest_dict()
        restored = deserialize_param_source(manifest)

        assert len(restored) == len(source)
        for i in range(len(source)):
            assert restored[i].params == source[i].params
            assert restored[i].case_id == source[i].case_id


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
