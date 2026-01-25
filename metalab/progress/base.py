"""
Base classes for progress tracking.

Provides the ProgressTracker protocol and SimpleProgressTracker implementation.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from metalab.events import Event


@dataclass
class Progress:
    """
    Configuration for progress display during experiment execution.

    Pass this to `metalab.run()` for customized progress tracking without
    needing to manually create and wire up a progress tracker.

    Attributes:
        title: Title for the progress display (defaults to experiment name).
        style: Progress style: "auto" (detect rich), "rich", or "simple".
        display_metrics: Metrics to show in progress output. Accepts:
            - Strings: "metric_name" or "metric_name:format_spec"
            - MetricDisplay instances for full control

    Example:
        # Simple progress
        result = metalab.run(exp, progress=True)

        # Customized progress
        result = metalab.run(
            exp,
            progress=metalab.Progress(
                title="Gene Perturbation",
                display_metrics=["gene", "perturbation_value:>8.0f"],
            ),
        )
    """

    title: str | None = None
    style: Literal["auto", "rich", "simple"] = "auto"
    display_metrics: list[str | "MetricDisplay"] | None = None


@dataclass
class MetricDisplay:
    """
    Configuration for how a metric should be displayed in progress output.
    
    Supports three ways to specify metrics:
    
    1. String (name only) - uses default formatting:
       "best_f"
    
    2. String with format spec (f-string style):
       "best_f:.2f"      # 2 decimal places
       "best_f:>8.2f"    # Right-align, 8 chars, 2 decimals
       "loss:.2e"        # Scientific notation
    
    3. MetricDisplay instance for full control:
       MetricDisplay("best_f", format=".2f", label="best")
       MetricDisplay("converged", formatter=lambda v: "yes" if v else "no")
    
    Attributes:
        name: The metric key to look up in the metrics dict.
        format: Python format spec (e.g., ".2f", ">8.2e"). Applied to numeric values.
        label: Display label. Defaults to name if not provided.
        formatter: Custom formatter function. Takes the value, returns a string.
            If provided, takes precedence over format.
    
    Example:
        from metalab.progress import MetricDisplay, create_progress_tracker
        
        tracker = create_progress_tracker(
            display_metrics=[
                "duration_ms",                    # Default formatting
                "best_f:.2f",                     # Format spec in string
                MetricDisplay("loss", format=".4e"),
                MetricDisplay(
                    "converged",
                    formatter=lambda v: "[green]✓[/green]" if v else "[red]✗[/red]"
                ),
            ],
        )
    """
    
    name: str
    format: str | None = None
    label: str | None = None
    formatter: Callable[[Any], str] | None = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        if self.label is None:
            self.label = self.name
    
    def format_value(self, value: Any) -> str:
        """Format a metric value for display."""
        # Custom formatter takes precedence
        if self.formatter is not None:
            return self.formatter(value)
        
        # Apply format spec to numeric values
        if self.format is not None and isinstance(value, (int, float)):
            return f"{value:{self.format}}"
        
        # Default formatting by type
        if isinstance(value, bool):
            return "✓" if value else "✗"
        if isinstance(value, float):
            # Use scientific notation for very small/large values
            if abs(value) < 0.001 or abs(value) > 10000:
                return f"{value:.2e}"
            return f"{value:.4g}"
        
        return str(value)
    
    @classmethod
    def normalize(cls, spec: "str | MetricDisplay") -> "MetricDisplay":
        """
        Normalize a metric specification to a MetricDisplay instance.
        
        Args:
            spec: Either a string (with optional format spec) or MetricDisplay.
        
        Returns:
            A MetricDisplay instance.
        
        Examples:
            MetricDisplay.normalize("best_f")        # MetricDisplay(name="best_f")
            MetricDisplay.normalize("best_f:.2f")    # MetricDisplay(name="best_f", format=".2f")
            MetricDisplay.normalize(md)              # Returns md unchanged
        """
        if isinstance(spec, cls):
            return spec
        
        if not isinstance(spec, str):
            raise TypeError(
                f"display_metrics must be strings or MetricDisplay instances, "
                f"got {type(spec).__name__}"
            )
        
        # Parse "name:format" syntax
        # Match: name followed by optional :format_spec
        # Format spec can include alignment (<>^), width, precision, type
        match = re.match(r"^([^:]+)(?::(.+))?$", spec)
        if match:
            name, fmt = match.groups()
            return cls(name=name.strip(), format=fmt)
        
        return cls(name=spec)


def normalize_display_metrics(
    display_metrics: list[str | MetricDisplay] | None,
) -> list[MetricDisplay]:
    """
    Normalize a list of metric specifications to MetricDisplay instances.
    
    Args:
        display_metrics: List of strings or MetricDisplay instances, or None.
    
    Returns:
        List of MetricDisplay instances (empty list if input is None).
    """
    if display_metrics is None:
        return []
    return [MetricDisplay.normalize(spec) for spec in display_metrics]


@runtime_checkable
class ProgressTracker(Protocol):
    """
    Protocol for progress trackers.
    
    Progress trackers receive events from the experiment runner and
    display progress information to the user.
    
    Trackers should be usable as context managers for setup/teardown.
    """
    
    total: int
    completed: int
    failed: int
    skipped: int
    
    def __call__(self, event: Event) -> None:
        """Handle an event from the runner."""
        ...
    
    def __enter__(self) -> ProgressTracker:
        """Enter the context (start display)."""
        ...
    
    def __exit__(self, *args: Any) -> None:
        """Exit the context (cleanup display)."""
        ...


class SimpleProgressTracker:
    """
    Simple text-based progress tracker.
    
    Outputs progress updates to stdout without any fancy formatting.
    Used as a fallback when rich is not available.
    
    Example:
        with SimpleProgressTracker(total=100) as tracker:
            result = metalab.run(exp, on_event=tracker)
    """

    def __init__(
        self,
        total: int = 0,
        title: str = "Experiment",
        show_failures: bool = True,
        update_interval: int = 1,
        display_metrics: list[str | MetricDisplay] | None = None,
    ) -> None:
        """
        Initialize the simple progress tracker.
        
        Args:
            total: Total number of runs expected.
            title: Title for the progress display.
            show_failures: Whether to print failure details.
            update_interval: Print progress every N completions.
            display_metrics: Metrics to display on completion. Accepts:
                - Strings: "metric_name" or "metric_name:format_spec"
                - MetricDisplay instances for full control
                If None, only duration is shown.
        """
        self.total = total
        self.title = title
        self.show_failures = show_failures
        self.update_interval = update_interval
        self._display_metrics = normalize_display_metrics(display_metrics)
        
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = time.time()
        self._update_counter = 0
        self._last_metrics: dict[str, Any] = {}

    def __call__(self, event: "Event") -> None:
        """Handle metalab events."""
        from metalab.events import EventKind
        
        if event.kind == EventKind.RUN_FINISHED:
            self.completed += 1
            # Capture metrics for display
            if event.payload:
                self._last_metrics = event.payload.get("metrics", {})
            self._update_counter += 1
            if self._update_counter >= self.update_interval:
                self._print_progress()
                self._update_counter = 0
                
        elif event.kind == EventKind.RUN_FAILED:
            self.failed += 1
            if self.show_failures:
                error = event.payload.get("error", "unknown")[:50] if event.payload else "unknown"
                print(f"  [FAIL] {event.run_id[:12]}... ({error})")
            self._print_progress()
            
        elif event.kind == EventKind.RUN_SKIPPED:
            self.skipped += 1
            
        elif event.kind == EventKind.PROGRESS:
            if event.payload:
                self.total = event.payload.get("total", self.total)

    def _print_progress(self) -> None:
        """Print current progress to stdout."""
        elapsed = time.time() - self.start_time
        done = self.completed + self.failed
        remaining = self.total - done - self.skipped

        if done > 0:
            eta = (elapsed / done) * remaining
            if eta < 60:
                eta_str = f"{eta:.0f}s"
            elif eta < 3600:
                eta_str = f"{eta/60:.1f}m"
            else:
                eta_str = f"{eta/3600:.1f}h"
        else:
            eta_str = "?"

        pct = (done / self.total * 100) if self.total > 0 else 0
        
        # Build metrics display string
        metrics_str = ""
        if self._display_metrics and self._last_metrics:
            metrics_parts = []
            for metric in self._display_metrics:
                if metric.name in self._last_metrics:
                    val = self._last_metrics[metric.name]
                    formatted = metric.format_value(val)
                    metrics_parts.append(f"{metric.label}={formatted}")
            if metrics_parts:
                metrics_str = " | " + " ".join(metrics_parts)
        
        print(
            f"  [{self.title}] {done}/{self.total} ({pct:.0f}%) "
            f"ok={self.completed} fail={self.failed} skip={self.skipped} "
            f"ETA: {eta_str}{metrics_str}"
        )

    def __enter__(self) -> "SimpleProgressTracker":
        """Start tracking."""
        print(f"\n{self.title}")
        print("-" * 60)
        return self

    def __exit__(self, *args: Any) -> None:
        """Finish tracking."""
        elapsed = time.time() - self.start_time
        print("-" * 60)
        print(f"Completed in {elapsed:.1f}s")
        print(f"  Success: {self.completed}")
        print(f"  Failed: {self.failed}")
        print(f"  Skipped: {self.skipped}")
