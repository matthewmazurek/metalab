"""
Display utilities for experiment results.

Provides rich formatting for result tables and summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metalab.result import Results

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def display_results(
    result: "Results",
    console: Any | None = None,
    show_by_group: list[str] | None = None,
) -> None:
    """
    Display experiment results in a formatted table.

    If rich is available, displays beautiful tables and panels.
    Otherwise falls back to plain text output.

    Args:
        result: The Results object.
        console: Optional rich Console instance.
        show_by_group: Column names to group results by (e.g., ["algorithm", "problem"]).

    Example:
        handle = metalab.run(exp)
        results = handle.result()
        display_results(results, show_by_group=["algorithm"])
    """
    df = result.table(as_dataframe=True)

    if df.empty:
        if HAS_RICH and console:
            console.print("[yellow]No results to display.[/yellow]")
        else:
            print("No results to display.")
        return

    if HAS_RICH:
        _display_rich(df, console, show_by_group)
    else:
        _display_simple(df, show_by_group)


def _display_rich(df: Any, console: Any | None, groups: list[str] | None) -> None:
    """Display results using rich formatting."""
    if console is None:
        console = Console()

    # Summary table
    summary = Table(title="Results Summary", show_header=True, header_style="bold")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")

    summary.add_row("Total runs", str(len(df)))
    summary.add_row("Successful", f"[green]{(df['status'] == 'success').sum()}[/green]")
    summary.add_row("Failed", f"[red]{(df['status'] == 'failed').sum()}[/red]")

    success_df = df[df["status"] == "success"]

    if not success_df.empty and "duration_ms" in success_df.columns:
        summary.add_row("", "")
        summary.add_row("Avg duration", f"{success_df['duration_ms'].mean():.1f}ms")
        summary.add_row("Total time", f"{success_df['duration_ms'].sum()/1000:.1f}s")

    console.print(summary)

    # Group breakdowns
    if groups and not success_df.empty:
        for group_col in groups:
            if group_col in success_df.columns:
                _display_group_table(console, success_df, group_col)

    # Best result
    if not success_df.empty:
        # Find the best metric column (common patterns)
        metric_cols = [
            c
            for c in success_df.columns
            if c in ("best_f", "loss", "error", "score", "accuracy", "f1")
        ]

        if metric_cols:
            metric_col = metric_cols[0]
            # For accuracy/score, higher is better; for loss/error, lower is better
            if metric_col in ("accuracy", "score", "f1"):
                best_idx = success_df[metric_col].idxmax()
            else:
                best_idx = success_df[metric_col].idxmin()

            best_row = success_df.loc[best_idx]
            _display_best_result(console, best_row, metric_col)


def _display_group_table(console: Any, df: Any, group_col: str) -> None:
    """Display a breakdown table for a grouping column."""
    # Find a metric column
    metric_cols = [
        c
        for c in df.columns
        if c in ("best_f", "loss", "error", "score", "accuracy", "f1")
    ]

    if not metric_cols:
        return

    metric_col = metric_cols[0]

    table = Table(
        title=f"Performance by {group_col.title()}",
        show_header=True,
    )
    table.add_column(group_col.title(), style="cyan")
    table.add_column(f"Mean {metric_col}", justify="right")
    table.add_column(f"Best {metric_col}", justify="right", style="green")
    table.add_column("Runs", justify="right")

    stats = df.groupby(group_col)[metric_col].agg(["mean", "min", "count"])
    for group_val, row in stats.iterrows():
        table.add_row(
            str(group_val),
            f"{row['mean']:.2e}",
            f"{row['min']:.2e}",
            str(int(row["count"])),
        )

    console.print()
    console.print(table)


def _display_best_result(console: Any, best_row: Any, metric_col: str) -> None:
    """Display the best result in a panel."""
    lines = [f"Run ID: {best_row['run_id'][:24]}..."]

    # Add relevant columns
    for col in ["algorithm", "problem", "model", "config"]:
        if col in best_row.index:
            lines.append(f"{col.title()}: {best_row[col]}")

    if "dim" in best_row.index:
        lines.append(f"Dimension: {best_row['dim']}")

    lines.append(f"{metric_col}: [bold green]{best_row[metric_col]:.6e}[/bold green]")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold]Best Result[/bold]",
            border_style="green",
        )
    )


def _display_simple(df: Any, groups: list[str] | None) -> None:
    """Display results using plain text."""
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nTotal runs: {len(df)}")
    print(f"  Success: {(df['status'] == 'success').sum()}")
    print(f"  Failed: {(df['status'] == 'failed').sum()}")

    success_df = df[df["status"] == "success"]

    if not success_df.empty and "duration_ms" in success_df.columns:
        print(
            f"\nTiming: mean={success_df['duration_ms'].mean():.1f}ms, "
            f"total={success_df['duration_ms'].sum()/1000:.1f}s"
        )

    # Group breakdowns
    if groups and not success_df.empty:
        metric_cols = [
            c
            for c in success_df.columns
            if c in ("best_f", "loss", "error", "score", "accuracy")
        ]
        if metric_cols:
            metric_col = metric_cols[0]
            for group_col in groups:
                if group_col in success_df.columns:
                    print(f"\nBy {group_col}:")
                    stats = success_df.groupby(group_col)[metric_col].agg(
                        ["mean", "min"]
                    )
                    for group_val, row in stats.iterrows():
                        print(
                            f"  {group_val}: mean={row['mean']:.2e}, best={row['min']:.2e}"
                        )
