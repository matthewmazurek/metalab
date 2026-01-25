#!/usr/bin/env python3
"""
Run the optimization benchmark experiment.

Usage:
    # Light test (quick verification)
    uv run python examples/optbench/run.py --intensity light

    # Medium stress test
    uv run python examples/optbench/run.py --intensity medium --workers 8

    # Heavy stress test (warning: many runs)
    uv run python examples/optbench/run.py --intensity heavy --workers 16

    # Extreme stress test (finding breaking points)
    uv run python examples/optbench/run.py --intensity extreme --workers 32

    # Targeted investigation
    uv run python examples/optbench/run.py --targeted adam rosenbrock

    # Random hyperparameter search
    uv run python examples/optbench/run.py --random --n-trials 200

This script demonstrates:
- Configurable experiment intensity
- Progress tracking with metalab.progress API (rich progress bar)
- Result analysis with metalab.display_results
- Error handling and recovery
"""

from __future__ import annotations

import argparse
from datetime import datetime

import metalab
from examples.optbench.experiment import (
    build_experiment,
    build_random_search_experiment,
    build_targeted_experiment,
)

# Check for rich availability
try:
    from rich.console import Console

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def count_grid_runs(exp: metalab.Experiment) -> int:
    """Count total number of runs in an experiment."""
    n_params = sum(1 for _ in exp.params)
    n_seeds = sum(1 for _ in exp.seeds)
    return n_params * n_seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the optimization benchmark experiment"
    )
    parser.add_argument(
        "--intensity",
        choices=["light", "medium", "heavy", "extreme"],
        default="light",
        help="Experiment intensity (default: light)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument(
        "--store",
        default="./runs/optbench",
        help="Storage directory (default: ./runs/optbench)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (re-run all)",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich progress bar (use simple output)",
    )
    parser.add_argument(
        "--targeted",
        nargs=2,
        metavar=("ALGORITHM", "PROBLEM"),
        help="Run targeted experiment for specific algorithm and problem",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random hyperparameter search",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for random search (default: 100)",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        help="Override number of replicates",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count runs without executing",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt for large experiments",
    )

    args = parser.parse_args()

    # Setup console for output
    console = Console() if HAS_RICH and not args.no_rich else None

    def print_msg(msg: str, style: str = "") -> None:
        if console:
            console.print(msg)
        else:
            # Strip rich markup for plain output
            import re

            plain = re.sub(r"\[/?[^\]]+\]", "", msg)
            print(plain)

    # Build experiment
    if args.targeted:
        algorithm, problem = args.targeted
        print_msg(
            f"Building targeted experiment: [cyan]{algorithm}[/cyan] on [cyan]{problem}[/cyan]"
        )
        exp = build_targeted_experiment(
            algorithm=algorithm,
            problem=problem,
            replicates=args.replicates or 10,
        )
    elif args.random:
        print_msg(
            f"Building random search experiment with [cyan]{args.n_trials}[/cyan] trials"
        )
        exp = build_random_search_experiment(
            n_trials=args.n_trials,
            replicates=args.replicates or 3,
        )
    else:
        print_msg(f"Building experiment with intensity: [cyan]{args.intensity}[/cyan]")
        exp = build_experiment(intensity=args.intensity)

    # Count runs
    n_runs = count_grid_runs(exp)
    print_msg(f"Experiment: [bold]{exp.name}[/bold] v{exp.version}")
    print_msg(f"Total runs: [bold]{n_runs:,}[/bold]")

    if args.dry_run:
        print_msg("\n[yellow][Dry run - not executing][/yellow]")
        print_msg("\nSample parameter combinations:")
        for i, param_case in enumerate(exp.params):
            if i >= 5:
                remaining = n_runs // max(1, sum(1 for _ in exp.seeds)) - 5
                print_msg(f"  ... and {remaining} more")
                break
            print_msg(f"  {param_case.params}")
        return

    # Warn about large experiments
    if n_runs > 1000 and not args.yes:
        print_msg(f"\n[yellow]WARNING: This will execute {n_runs:,} runs![/yellow]")
        print_msg("Consider using --intensity light for initial testing.")
        print_msg("Use --yes to skip this prompt.")
        try:
            response = input("Continue? [y/N] ")
            if response.lower() != "y":
                print_msg("Aborted.")
                return
        except EOFError:
            print_msg("\nNo TTY available. Use --yes to skip confirmation.")
            return

    print_msg(f"\nStarting with [cyan]{args.workers}[/cyan] workers...")
    print_msg(f"Store: {args.store}")
    print_msg(f"Resume: {not args.no_resume}")

    # Create progress tracker using metalab API
    progress_style = "simple" if args.no_rich else "auto"
    tracker = metalab.create_progress_tracker(
        total=n_runs,
        title=f"{exp.name} ({args.intensity})" if not args.targeted else f"{exp.name}",
        style=progress_style,
        display_metrics=["best_f", "converged"],
    )

    # Run experiment with progress tracker
    with tracker:
        result = metalab.run(
            exp,
            store=args.store,
            executor="threads",
            max_workers=args.workers,
            resume=not args.no_resume,
            progress=True,
            on_event=tracker,
        )

    # Display results using metalab API
    metalab.display_results(
        result,
        console=console,
        show_by_group=["algorithm", "problem"],
    )

    # Save results table
    df = result.table(as_dataframe=True)
    if not df.empty:
        csv_path = (
            f"{args.store}/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df.to_csv(csv_path, index=False)
        print_msg(f"\n[dim]Results saved to: {csv_path}[/dim]")


if __name__ == "__main__":
    main()
