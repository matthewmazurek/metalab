"""Tests for identity computation."""

from __future__ import annotations

from metalab._ids import (
    compute_run_id,
    fingerprint_context,
    fingerprint_params,
    fingerprint_seeds,
)
from metalab.seeds.bundle import SeedBundle


class TestFingerprintParams:
    """Tests for fingerprint_params()."""

    def test_fingerprint_params_deterministic(self):
        """Same params should produce same fingerprint."""
        params = {"learning_rate": 0.01, "batch_size": 32}
        fp1 = fingerprint_params(params)
        fp2 = fingerprint_params(params)
        assert fp1 == fp2

    def test_fingerprint_params_order_independent(self):
        """Dict ordering should not affect fingerprint."""
        params1 = {"a": 1, "b": 2}
        params2 = {"b": 2, "a": 1}
        assert fingerprint_params(params1) == fingerprint_params(params2)

    def test_fingerprint_params_different_values(self):
        """Different values should produce different fingerprints."""
        fp1 = fingerprint_params({"x": 1})
        fp2 = fingerprint_params({"x": 2})
        assert fp1 != fp2


class TestFingerprintContext:
    """Tests for fingerprint_context()."""

    def test_fingerprint_context_dict(self):
        """Dict context should fingerprint correctly."""
        ctx = {"dataset": "train.csv", "version": "1.0"}
        fp = fingerprint_context(ctx)
        assert len(fp) == 16

    def test_fingerprint_context_dataclass(self):
        """Dataclass context should fingerprint correctly."""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MyContext:
            name: str
            value: int

        ctx = MyContext(name="test", value=42)
        fp = fingerprint_context(ctx)
        assert len(fp) == 16

    def test_fingerprint_context_deterministic(self):
        """Same context should produce same fingerprint."""
        ctx = {"key": "value"}
        assert fingerprint_context(ctx) == fingerprint_context(ctx)


class TestFingerprintSeeds:
    """Tests for fingerprint_seeds()."""

    def test_fingerprint_seeds_deterministic(self):
        """Same bundle should produce same fingerprint."""
        bundle = SeedBundle(root_seed=42, replicate_index=0)
        fp1 = fingerprint_seeds(bundle)
        fp2 = fingerprint_seeds(bundle)
        assert fp1 == fp2

    def test_fingerprint_seeds_different_root(self):
        """Different root seeds should produce different fingerprints."""
        b1 = SeedBundle(root_seed=42, replicate_index=0)
        b2 = SeedBundle(root_seed=123, replicate_index=0)
        assert fingerprint_seeds(b1) != fingerprint_seeds(b2)

    def test_fingerprint_seeds_different_replicate(self):
        """Different replicates should produce different fingerprints."""
        b1 = SeedBundle(root_seed=42, replicate_index=0)
        b2 = SeedBundle(root_seed=42, replicate_index=1)
        assert fingerprint_seeds(b1) != fingerprint_seeds(b2)


class TestComputeRunId:
    """Tests for compute_run_id()."""

    def test_compute_run_id_deterministic(self):
        """Same inputs should produce same run_id."""
        run_id1 = compute_run_id(
            experiment_id="test:1.0",
            context_fp="abc123",
            params_fp="def456",
            seed_fp="ghi789",
        )
        run_id2 = compute_run_id(
            experiment_id="test:1.0",
            context_fp="abc123",
            params_fp="def456",
            seed_fp="ghi789",
        )
        assert run_id1 == run_id2

    def test_compute_run_id_length(self):
        """Run ID should be 16 hex characters."""
        run_id = compute_run_id(
            experiment_id="test:1.0",
            context_fp="abc",
            params_fp="def",
            seed_fp="ghi",
        )
        assert len(run_id) == 16
        assert all(c in "0123456789abcdef" for c in run_id)

    def test_compute_run_id_different_experiment(self):
        """Different experiment should produce different run_id."""
        run_id1 = compute_run_id("exp1:1.0", "ctx", "params", "seed")
        run_id2 = compute_run_id("exp2:1.0", "ctx", "params", "seed")
        assert run_id1 != run_id2

    def test_compute_run_id_different_params(self):
        """Different params fingerprint should produce different run_id."""
        run_id1 = compute_run_id("exp:1.0", "ctx", "params1", "seed")
        run_id2 = compute_run_id("exp:1.0", "ctx", "params2", "seed")
        assert run_id1 != run_id2

    def test_compute_run_id_with_code_fp(self):
        """Code fingerprint should affect run_id."""
        run_id1 = compute_run_id("exp:1.0", "ctx", "params", "seed", code_fp="code1")
        run_id2 = compute_run_id("exp:1.0", "ctx", "params", "seed", code_fp="code2")
        run_id3 = compute_run_id("exp:1.0", "ctx", "params", "seed", code_fp=None)
        assert run_id1 != run_id2
        assert run_id1 != run_id3
