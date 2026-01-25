"""
How to run the experiment (goal UX).

Intended usage:
  uv run python examples/pi_mc/run.py

This script demonstrates:
- one-liner metalab.run facade
- threads executor selection by string
- file store specified as path
- resume/dedupe
- result handle usage (table + artifact loading)
"""

from __future__ import annotations

import metalab
from examples.pi_mc.experiment import build_experiment


def main() -> None:
    exp = build_experiment()

    # Run with a very small surface area.
    # Under the hood this should construct:
    # - FileStore at ./runs
    # - ThreadExecutor(max_workers=8)
    # - Runner(resume=True, progress=True)
    result = metalab.run(
        exp,
        store="./runs",
        executor="threads",
        max_workers=8,
        resume=True,
        progress=True,
        # on_event=... (optional)
    )

    # Results handle: tabular view of run records
    df = result.table(as_dataframe=True)  # pandas DataFrame
    print(df[["run_id", "status", "pi_estimate", "n_samples"]].sort_values("n_samples"))

    # Example: load an artifact (only exists if store_points=True)
    # run_id = df.loc[df["status"] == "success", "run_id"].iloc[0]
    # points = result.load(run_id, "points")
    # print(points.keys())


if __name__ == "__main__":
    main()
