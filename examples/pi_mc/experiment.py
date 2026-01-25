"""
Goal example: Monte Carlo π estimation.
This is a minimal, domain-agnostic demo of the intended metalab user API.

Key features demonstrated:
- @metalab.operation: operation as a function (no user subclasses required)
- metalab.grid: parameter generation (cartesian product)
- metalab.seeds: explicit seed plan with replicates
- capture.metric / capture.artifact: metrics + artifacts
- automatic artifact serializer selection (no forced "policy")
- results handle with .table() and .load()

Hints (not used directly here) are included at bottom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import metalab


@dataclass(frozen=True)
class EmptyContextSpec:
    """
    Minimal context spec. In real use this would include dataset/resource IDs,
    checksums, configuration fragments, etc. It must be serializable.
    """

    name: str = "empty"
    version: str = "1.0"
    data: dict[str, Any] = None  # type: ignore[assignment]


@metalab.operation(name="pi_mc")
def pi_monte_carlo(
    context: Any,  # intentionally unused in this example
    params: dict[str, Any],
    seeds: metalab.SeedBundle,
    capture: metalab.Capture,
    runtime: metalab.Runtime | None = None,
) -> metalab.RunRecord:
    """
    Estimate π by sampling points uniformly in [0,1]^2.

    Params:
      - n_samples: int
      - store_points: bool (optional)
    """
    n = int(params["n_samples"])
    store_points = bool(params.get("store_points", False))

    # Explicit RNG derived from SeedBundle (no global RNG use).
    rng = seeds.numpy()  # or seeds.rng("numpy")

    x = rng.random(n)
    y = rng.random(n)
    inside = (x * x + y * y) <= 1.0
    pi_est = float(4.0 * inside.mean())

    # Metrics (flat scalars)
    capture.metric("pi_estimate", pi_est)
    capture.metric("n_samples", n)
    capture.metric("inside_frac", float(inside.mean()))

    # Optional artifact (large-ish): store raw points as a bundle
    if store_points:
        points = {
            "x": x.astype(np.float32),
            "y": y.astype(np.float32),
            "inside": inside,
        }

        # No explicit format required: serializer registry should pick something reasonable
        # (e.g., npz or parquet-like depending on supported serializers).
        capture.artifact(
            name="points",
            obj=points,
            kind="array_bundle",
            metadata={"n_samples": n},
        )

    # Return a small RunRecord (framework can also populate many fields automatically).
    return metalab.RunRecord.success()


def build_experiment() -> metalab.Experiment:
    # Context: serializable manifest, reconstructable on workers
    context_spec = EmptyContextSpec(data={})

    # Param sweep: cartesian product
    params = metalab.grid(
        n_samples=[1_000, 10_000, 100_000],
        store_points=[False],  # flip True to exercise artifact storage
    )

    # Seeds: base seed + replicates (replicate index becomes part of run identity)
    seeds = metalab.seeds(base=42, replicates=3)

    exp = metalab.Experiment(
        name="pi_mc",
        version="0.1",
        context=context_spec,  # the spec, not the built context object
        operation=pi_monte_carlo,
        params=params,
        seeds=seeds,
        tags=["example", "monte_carlo"],
    )
    return exp


# --- Hints / optional API surface (not used directly above) -------------------


def _hints_not_used_directly() -> None:
    # 1) Random params (replayable sampling)
    _ = metalab.random(
        space={
            "n_samples": metalab.loguniform_int(1_000, 1_000_000),
            "store_points": metalab.choice([False, True]),
        },
        n_trials=20,
        seed=123,
    )

    # 2) Param resolution (conditional/derived params)
    # (User supplies a resolver; metalab calls it before running.)
    def resolver(context_meta: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
        # Example: derive a parameter from context metadata
        out = dict(raw)
        out["n_samples"] = int(out["n_samples"])
        return out

    _ = metalab.with_resolver(metalab.grid(n_samples=[1000]), resolver=resolver)

    # 3) Alternate executors (processes / ray / arc), same experiment
    # metalab.run(exp, executor="processes", max_workers=8)
    # metalab.run(exp, executor="ray", address="auto")
    # metalab.run(exp, executor="arc", queue="short", cpus=4, memory="16G")

    # 4) File-sink artifacts (user generates file, capture uploads/records it)
    # with capture.file_sink("plot.png") as path:
    #     make_plot_and_save(path)
    # capture.from_path("plot", "/tmp/plot.png", kind="image")

    # 5) Event hooks
    def on_event(evt: metalab.Event) -> None:
        # evt.kind: "run_started" | "artifact_saved" | "run_finished" | "run_failed" | ...
        # evt.run_id, evt.payload, evt.timestamp, ...
        pass

    # metalab.run(exp, on_event=on_event, progress=True)

    return
