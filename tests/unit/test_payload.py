"""Tests for RunPayload."""

import pytest

from metalab.executor.payload import RunPayload
from metalab.seeds.bundle import SeedBundle


class TestMakeLogLabel:
    """Tests for make_log_label method."""

    def test_basic_params(self):
        """Creates label from string params and replicate index."""
        payload = RunPayload(
            run_id="abc123",
            experiment_id="test:1.0",
            context_spec={},
            params_resolved={"algorithm": "gd", "problem": "rastrigin"},
            seed_bundle=SeedBundle(root_seed=42, replicate_index=0),
            store_locator="/tmp/store",
        )
        label = payload.make_log_label()
        assert label == "gd_rastrigin_r0"

    def test_numeric_params(self):
        """Numeric params include key name."""
        payload = RunPayload(
            run_id="abc123",
            experiment_id="test:1.0",
            context_spec={},
            params_resolved={"dim": 10, "lr": 0.01},
            seed_bundle=SeedBundle(root_seed=42, replicate_index=2),
            store_locator="/tmp/store",
        )
        label = payload.make_log_label()
        assert label == "dim10_lr0.01_r2"

    def test_skips_internal_params(self):
        """Params starting with underscore are skipped."""
        payload = RunPayload(
            run_id="abc123",
            experiment_id="test:1.0",
            context_spec={},
            params_resolved={"_internal": "hidden", "visible": "shown"},
            seed_bundle=SeedBundle(root_seed=42, replicate_index=1),
            store_locator="/tmp/store",
        )
        label = payload.make_log_label()
        assert label == "shown_r1"
        assert "hidden" not in label
        assert "_internal" not in label

    def test_max_params_limit(self):
        """Only first max_params values included."""
        payload = RunPayload(
            run_id="abc123",
            experiment_id="test:1.0",
            context_spec={},
            params_resolved={"a": "1", "b": "2", "c": "3", "d": "4"},
            seed_bundle=SeedBundle(root_seed=42, replicate_index=0),
            store_locator="/tmp/store",
        )
        label = payload.make_log_label(max_params=2)
        assert label == "1_2_r0"

    def test_bool_params(self):
        """Bool True includes key name, False is skipped."""
        payload = RunPayload(
            run_id="abc123",
            experiment_id="test:1.0",
            context_spec={},
            params_resolved={"debug": True, "quiet": False},
            seed_bundle=SeedBundle(root_seed=42, replicate_index=0),
            store_locator="/tmp/store",
        )
        label = payload.make_log_label()
        assert "debug" in label
        assert "quiet" not in label
