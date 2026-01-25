"""
Rich progress tracker for beautiful terminal output.

Requires the 'rich' package: pip install metalab[rich]
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from metalab.progress.base import MetricDisplay, normalize_display_metrics

if TYPE_CHECKING:
    from metalab.events import Event


class RichProgressTracker:
    """
    Beautiful progress tracking with rich.
    
    Displays a live-updating progress bar with:
    - Spinner and progress bar
    - Success/failed/skipped/running counters
    - Recent activity feed
    - Time elapsed and ETA
    
    Example:
        from metalab.progress import RichProgressTracker
        
        with RichProgressTracker(total=100, title="My Experiment") as tracker:
            result = metalab.run(exp, on_event=tracker, progress=False)
    """

    def __init__(
        self,
        total: int = 0,
        title: str = "Experiment",
        show_recent: int = 5,
        console: Console | None = None,
        display_metrics: list[str | MetricDisplay] | None = None,
    ) -> None:
        """
        Initialize the rich progress tracker.
        
        Args:
            total: Total number of runs expected.
            title: Title for the progress display.
            show_recent: Number of recent events to show.
            console: Rich console instance (created if None).
            display_metrics: Metrics to display in recent activity. Accepts:
                - Strings: "metric_name" or "metric_name:format_spec"
                - MetricDisplay instances for full control
                If None, only duration is shown.
        """
        self.total = total
        self.title = title
        self.show_recent = show_recent
        self._display_metrics = normalize_display_metrics(display_metrics)
        
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.in_progress: dict[str, dict[str, Any]] = {}
        self.recent_events: deque[tuple[str, str, str]] = deque(maxlen=show_recent)
        self.start_time = time.time()

        self.console = console or Console()
        self._owns_console = console is None  # Track if we created the console
        
        # Main progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=False,
        )

        self.task_id = None
        self.live: Live | None = None

    def _make_display(self) -> Group:
        """Create the rich display layout."""
        # Stats table
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(style="green", justify="right")
        stats_table.add_column(style="dim")
        stats_table.add_column(style="red", justify="right")
        stats_table.add_column(style="dim")
        stats_table.add_column(style="yellow", justify="right")
        stats_table.add_column(style="dim")
        stats_table.add_column(style="cyan", justify="right")
        stats_table.add_column(style="dim")

        stats_table.add_row(
            str(self.completed), "success",
            str(self.failed), "failed",
            str(self.skipped), "skipped",
            str(len(self.in_progress)), "running",
        )

        # Recent events table
        events_table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            collapse_padding=True,
        )
        events_table.add_column("Status", width=8)
        events_table.add_column("Run ID", width=14)
        events_table.add_column("Details", overflow="ellipsis")

        for status, run_id, details in self.recent_events:
            if status == "success":
                status_text = Text("✓ done", style="green")
            elif status == "failed":
                status_text = Text("✗ fail", style="red")
            elif status == "skip":
                status_text = Text("→ skip", style="yellow")
            else:
                status_text = Text("● run", style="blue")

            events_table.add_row(status_text, run_id[:12] + "…", details)

        # Combine into panels
        components = [self.progress]

        if self.recent_events:
            components.append(
                Panel(
                    events_table,
                    title="[dim]Recent Activity[/dim]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )

        components.append(
            Panel(
                stats_table,
                title=f"[bold]{self.title}[/bold]",
                border_style="blue",
                padding=(0, 1),
            )
        )

        return Group(*components)

    def _format_run_details(self, duration_ms: int, metrics: dict[str, Any]) -> str:
        """Format run details for the recent activity display."""
        parts = [f"{duration_ms}ms"]
        
        # Add configured metrics if available
        for metric in self._display_metrics:
            if metric.name in metrics:
                value = metrics[metric.name]
                formatted = metric.format_value(value)
                parts.append(f"{metric.label}={formatted}")
        
        return " | ".join(parts)

    def __call__(self, event: "Event") -> None:
        """Handle metalab events."""
        from metalab.events import EventKind
        
        if event.kind == EventKind.RUN_STARTED:
            self.in_progress[event.run_id] = {
                "start_time": time.time(),
            }

        elif event.kind == EventKind.RUN_FINISHED:
            self.completed += 1
            duration = event.payload.get("duration_ms", 0) if event.payload else 0
            metrics = event.payload.get("metrics", {}) if event.payload else {}
            
            # Format details with optional metrics display
            details = self._format_run_details(duration, metrics)
            self.recent_events.appendleft(("success", event.run_id, details))
            self.in_progress.pop(event.run_id, None)
            if self.task_id is not None:
                self.progress.advance(self.task_id)

        elif event.kind == EventKind.RUN_FAILED:
            self.failed += 1
            error = event.payload.get("error", "unknown")[:35] if event.payload else "unknown"
            self.recent_events.appendleft(
                ("failed", event.run_id, error)
            )
            self.in_progress.pop(event.run_id, None)
            if self.task_id is not None:
                self.progress.advance(self.task_id)

        elif event.kind == EventKind.RUN_SKIPPED:
            self.skipped += 1
            reason = event.payload.get("reason", "cached")[:35] if event.payload else "cached"
            self.recent_events.appendleft(
                ("skip", event.run_id, reason)
            )

        elif event.kind == EventKind.PROGRESS:
            if event.payload:
                new_total = event.payload.get("total", self.total)
                if new_total != self.total:
                    self.total = new_total
                    if self.task_id is not None:
                        self.progress.update(self.task_id, total=self.total)

        # Update live display
        if self.live is not None:
            self.live.update(self._make_display())

    def __enter__(self) -> "RichProgressTracker":
        """Start the live display."""
        self.task_id = self.progress.add_task(
            "Running experiments",
            total=self.total or 100,
        )
        self.live = Live(
            self._make_display(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self.live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Stop the live display."""
        if self.live is not None:
            self.live.__exit__(*args)
        
        # Print final summary
        elapsed = time.time() - self.start_time
        self.console.print()
        self.console.print(Panel(
            f"[green]✓ Completed[/green] in [bold]{elapsed:.1f}s[/bold]\n"
            f"  Success: [green]{self.completed}[/green]\n"
            f"  Failed: [red]{self.failed}[/red]\n"
            f"  Skipped: [yellow]{self.skipped}[/yellow]",
            title="Execution Complete",
            border_style="green",
        ))

    def get_console(self) -> Console:
        """Get the rich Console instance for output routing."""
        return self.console
