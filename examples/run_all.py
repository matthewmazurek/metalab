#!/usr/bin/env python
"""Run all metalab examples and report timing.

Usage:
    uv run python examples/run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

EXAMPLES = [
    ("pi_mc", "examples/pi_mc/run.py"),
    ("curve_fit", "examples/curve_fit/run.py"),
    ("hypersearch", "examples/hypersearch/run.py"),
    ("optbench", "examples/optbench/run.py"),
]


def run_example(name: str, path: str) -> tuple[bool, float]:
    """Run an example and return (success, elapsed_seconds)."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(
        [sys.executable, path],
        cwd=Path(__file__).parent.parent,
    )
    elapsed = time.time() - start

    return result.returncode == 0, elapsed


def main():
    print("Running all metalab examples...")
    print(f"Python: {sys.executable}")

    results = []
    total_start = time.time()

    for name, path in EXAMPLES:
        success, elapsed = run_example(name, path)
        results.append((name, success, elapsed))

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Example':<15} {'Status':<10} {'Time':>10}")
    print("-" * 35)
    for name, success, elapsed in results:
        status = "OK" if success else "FAILED"
        print(f"{name:<15} {status:<10} {elapsed:>8.1f}s")
    print("-" * 35)
    print(f"{'Total':<15} {'':<10} {total_elapsed:>8.1f}s")

    # Exit with error if any failed
    if not all(success for _, success, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
