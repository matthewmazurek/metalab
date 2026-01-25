"""Tests for seed management."""

from __future__ import annotations

from metalab.seeds import SeedBundle, SeedPlan, seeds


class TestSeedBundle:
    """Tests for SeedBundle."""

    def test_derive_deterministic(self):
        """Same name should produce same derived seed."""
        bundle = SeedBundle(root_seed=42, replicate_index=0)
        seed1 = bundle.derive("test")
        seed2 = bundle.derive("test")
        assert seed1 == seed2

    def test_derive_different_names(self):
        """Different names should produce different seeds."""
        bundle = SeedBundle(root_seed=42, replicate_index=0)
        seed1 = bundle.derive("sampling")
        seed2 = bundle.derive("initialization")
        assert seed1 != seed2

    def test_derive_different_roots(self):
        """Different root seeds should produce different derived seeds."""
        b1 = SeedBundle(root_seed=42, replicate_index=0)
        b2 = SeedBundle(root_seed=123, replicate_index=0)
        assert b1.derive("test") != b2.derive("test")

    def test_derive_different_replicates(self):
        """Different replicates should produce different derived seeds."""
        b1 = SeedBundle(root_seed=42, replicate_index=0)
        b2 = SeedBundle(root_seed=42, replicate_index=1)
        assert b1.derive("test") != b2.derive("test")

    def test_rng_produces_stdlib_random(self):
        """rng() should return stdlib Random instance."""
        import random

        bundle = SeedBundle(root_seed=42)
        rng = bundle.rng("test")
        assert isinstance(rng, random.Random)

    def test_rng_reproducible(self):
        """Same seed bundle should produce same random sequence."""
        b1 = SeedBundle(root_seed=42)
        b2 = SeedBundle(root_seed=42)

        rng1 = b1.rng("test")
        rng2 = b2.rng("test")

        values1 = [rng1.random() for _ in range(10)]
        values2 = [rng2.random() for _ in range(10)]
        assert values1 == values2

    def test_numpy_requires_numpy(self):
        """numpy() should work when numpy is installed."""
        try:
            import numpy as np

            bundle = SeedBundle(root_seed=42)
            rng = bundle.numpy("test")
            assert isinstance(rng, np.random.Generator)
        except ImportError:
            pass  # numpy not installed, skip

    def test_numpy_reproducible(self):
        """Same seed should produce same numpy sequence."""
        try:
            import numpy as np

            b1 = SeedBundle(root_seed=42)
            b2 = SeedBundle(root_seed=42)

            rng1 = b1.numpy("test")
            rng2 = b2.numpy("test")

            values1 = rng1.random(10)
            values2 = rng2.random(10)
            assert np.allclose(values1, values2)
        except ImportError:
            pass  # numpy not installed, skip

    def test_to_dict_from_dict(self):
        """Serialization round-trip should preserve data."""
        bundle = SeedBundle(root_seed=42, replicate_index=5)
        data = bundle.to_dict()
        restored = SeedBundle.from_dict(data)
        assert restored.root_seed == bundle.root_seed
        assert restored.replicate_index == bundle.replicate_index


class TestSeedPlan:
    """Tests for SeedPlan."""

    def test_seeds_factory(self):
        """seeds() factory should create SeedPlan."""
        plan = seeds(base=42, replicates=3)
        assert isinstance(plan, SeedPlan)
        assert plan.base_seed == 42
        assert plan.replicates == 3

    def test_iteration(self):
        """Iterating should yield SeedBundles."""
        plan = seeds(base=42, replicates=3)
        bundles = list(plan)
        assert len(bundles) == 3
        for i, bundle in enumerate(bundles):
            assert isinstance(bundle, SeedBundle)
            assert bundle.root_seed == 42
            assert bundle.replicate_index == i

    def test_len(self):
        """len() should return replicates."""
        plan = seeds(base=42, replicates=5)
        assert len(plan) == 5

    def test_getitem(self):
        """Indexing should return specific bundle."""
        plan = seeds(base=42, replicates=3)
        bundle = plan[1]
        assert bundle.root_seed == 42
        assert bundle.replicate_index == 1

    def test_single_replicate(self):
        """Single replicate should work."""
        plan = seeds(base=42, replicates=1)
        bundles = list(plan)
        assert len(bundles) == 1
        assert bundles[0].replicate_index == 0
