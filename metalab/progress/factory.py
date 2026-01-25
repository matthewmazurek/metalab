"""
Factory function for creating progress trackers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from metalab.progress.base import MetricDisplay, ProgressTracker, SimpleProgressTracker

if TYPE_CHECKING:
    pass


def create_progress_tracker(
    total: int = 0,
    title: str = "Experiment",
    style: Literal["auto", "rich", "simple"] = "auto",
    display_metrics: list[str | MetricDisplay] | None = None,
    **kwargs,
) -> ProgressTracker:
    """
    Create a progress tracker.
    
    Factory function that creates the appropriate progress tracker
    based on the requested style and available dependencies.
    
    Args:
        total: Total number of runs expected.
        title: Title for the progress display.
        style: Progress style to use:
            - "auto": Use rich if available, otherwise simple
            - "rich": Use rich (raises ImportError if not available)
            - "simple": Use simple text output
        display_metrics: Metrics to display in recent activity. Accepts:
            - Strings: "metric_name" or "metric_name:format_spec"
            - MetricDisplay instances for full control over formatting
            If None, only duration is shown.
        **kwargs: Additional arguments passed to the tracker.
    
    Returns:
        A ProgressTracker instance.
    
    Raises:
        ImportError: If style="rich" but rich is not installed.
    
    Example:
        # Auto-select best available
        tracker = create_progress_tracker(total=100, title="My Experiment")
        
        # Force simple output
        tracker = create_progress_tracker(total=100, style="simple")
        
        # Display specific metrics with default formatting
        tracker = create_progress_tracker(
            total=100,
            display_metrics=["best_f", "converged"],
        )
        
        # Display metrics with custom format specs
        tracker = create_progress_tracker(
            total=100,
            display_metrics=[
                "best_f:.2f",      # 2 decimal places
                "loss:>8.2e",      # Right-align, 8 chars, scientific notation
                "converged",       # Default bool formatting (✓/✗)
            ],
        )
        
        # Full control with MetricDisplay
        from metalab.progress import MetricDisplay
        
        tracker = create_progress_tracker(
            total=100,
            display_metrics=[
                MetricDisplay("best_f", format=".2f", label="best"),
                MetricDisplay("converged", formatter=lambda v: "yes" if v else "no"),
            ],
        )
    """
    if style == "simple":
        return SimpleProgressTracker(
            total=total, title=title, display_metrics=display_metrics, **kwargs
        )
    
    if style == "rich":
        try:
            from metalab.progress.rich import RichProgressTracker
            return RichProgressTracker(
                total=total, title=title, display_metrics=display_metrics, **kwargs
            )
        except ImportError as e:
            raise ImportError(
                "Rich progress tracker requires the 'rich' package. "
                "Install it with: pip install metalab[rich]"
            ) from e
    
    # style == "auto"
    try:
        from metalab.progress.rich import RichProgressTracker
        return RichProgressTracker(
            total=total, title=title, display_metrics=display_metrics, **kwargs
        )
    except ImportError:
        return SimpleProgressTracker(
            total=total, title=title, display_metrics=display_metrics, **kwargs
        )
