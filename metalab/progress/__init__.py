"""
Progress tracking module for metalab experiments.

Provides rich progress bars and status displays for experiment execution.
Falls back to simple text output if rich is not installed.

Example:
    import metalab
    from metalab.progress import create_progress_tracker, MetricDisplay

    exp = build_experiment()

    # Option 1: Simple boolean (auto-detects rich)
    result = metalab.run(exp, progress=True)

    # Option 2: Progress config object for customization
    result = metalab.run(
        exp,
        progress=metalab.Progress(
            title="My Experiment",
            display_metrics=["best_f:.2f", "converged"],
        ),
    )

    # Option 3: Create tracker manually for full control
    with create_progress_tracker(total=100, title="My Experiment") as tracker:
        result = metalab.run(exp, on_event=tracker, progress=False)
"""

from metalab.progress.base import (
    MetricDisplay,
    Progress,
    ProgressTracker,
    SimpleProgressTracker,
    normalize_display_metrics,
)
from metalab.progress.factory import create_progress_tracker

# Conditionally export RichProgressTracker
try:
    from metalab.progress.rich import RichProgressTracker
    HAS_RICH = True
except ImportError:
    RichProgressTracker = None  # type: ignore
    HAS_RICH = False

__all__ = [
    "MetricDisplay",
    "Progress",
    "ProgressTracker",
    "SimpleProgressTracker",
    "RichProgressTracker",
    "create_progress_tracker",
    "normalize_display_metrics",
    "HAS_RICH",
]
