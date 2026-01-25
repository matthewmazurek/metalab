"""
Factory function for creating progress trackers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from metalab.progress.base import ProgressTracker, SimpleProgressTracker

if TYPE_CHECKING:
    pass


def create_progress_tracker(
    total: int = 0,
    title: str = "Experiment",
    style: Literal["auto", "rich", "simple"] = "auto",
    display_metrics: list[str] | None = None,
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
        display_metrics: List of metric names to display in recent activity.
            For example, ["best_f", "converged"] will show these metrics
            alongside the run duration. If None, only duration is shown.
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
        
        # Display specific metrics in progress
        tracker = create_progress_tracker(
            total=100,
            title="Optimization",
            display_metrics=["best_f", "converged"],
        )
        
        # Use with metalab.run
        with create_progress_tracker(total=n_runs) as tracker:
            result = metalab.run(exp, on_event=tracker, progress=False)
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
