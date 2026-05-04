"""Tiny smoke-test experiment for the filesystem-native HPC runner.

Run with:
    uv run metalab run examples/hpc_smoke.py --store /tmp/metalab-hpc-smoke --workers 2
"""

from __future__ import annotations

import time

import metalab


@metalab.context_spec
class SmokeContext:
    baseline: float = 10.0
    sleep_seconds: float = 30.0


@metalab.operation
def score_case(context: SmokeContext, params, seeds, capture):
    """Produce a deterministic scalar metric and one small JSON artifact."""
    rng = seeds.rng()
    time.sleep(context.sleep_seconds)

    score = context.baseline + params["x"] * params["scale"] + rng.random()

    capture.log(
        f"scored x={params['x']} scale={params['scale']} "
        f"replicate={seeds.replicate_index}"
    )
    capture.metric("score", score)
    capture.metric("x_scaled", params["x"] * params["scale"])
    capture.artifact(
        "inputs",
        {
            "x": params["x"],
            "scale": params["scale"],
            "replicate": seeds.replicate_index,
            "score": score,
        },
    )


exp = metalab.Experiment(
    name="hpc_smoke",
    version="1",
    context=SmokeContext(),
    operation=score_case,
    params=metalab.grid(
        x=[1, 2, 3],
        scale=[0.1, 1.0],
    ),
    seeds=metalab.seeds(base=2026, replicates=2),
)


def get_experiment():
    return exp
