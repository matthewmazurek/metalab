"""
Base classes for progress tracking.

Provides the ProgressTracker protocol and SimpleProgressTracker implementation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from metalab.events import Event


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
        display_metrics: list[str] | None = None,
    ) -> None:
        """
        Initialize the simple progress tracker.
        
        Args:
            total: Total number of runs expected.
            title: Title for the progress display.
            show_failures: Whether to print failure details.
            update_interval: Print progress every N completions.
            display_metrics: List of metric names to display on completion.
                If None, only duration is shown. Metrics are displayed in order.
        """
        self.total = total
        self.title = title
        self.show_failures = show_failures
        self.update_interval = update_interval
        self.display_metrics = display_metrics or []
        
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
        if self.display_metrics and self._last_metrics:
            metrics_parts = []
            for name in self.display_metrics:
                if name in self._last_metrics:
                    val = self._last_metrics[name]
                    if isinstance(val, float):
                        if abs(val) < 0.001 or abs(val) > 10000:
                            metrics_parts.append(f"{name}={val:.2e}")
                        else:
                            metrics_parts.append(f"{name}={val:.4g}")
                    else:
                        metrics_parts.append(f"{name}={val}")
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
