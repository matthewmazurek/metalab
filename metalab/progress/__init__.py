"""
Progress tracking module for metalab experiments.

Provides rich progress bars and status displays for experiment execution.
Falls back to simple text output if rich is not installed.

Example:
    import metalab
    from metalab.progress import create_progress_tracker, MetricDisplay

    exp = build_experiment()
    
    # Option 1: Use with metalab.run() directly
    result = metalab.run(exp, progress="rich")  # or progress="simple" or progress=True
    
    # Option 2: Create tracker manually for more control
    with create_progress_tracker(total=100, title="My Experiment") as tracker:
        result = metalab.run(exp, on_event=tracker, progress=False)
    
    # Option 3: Custom metric formatting
    tracker = create_progress_tracker(
        total=100,
        display_metrics=[
            "best_f:.2f",                       # Format spec in string
            MetricDisplay("loss", format=".4e", label="L"),
        ],
    )
"""

from metalab.progress.base import (
    MetricDisplay,
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
    "ProgressTracker",
    "SimpleProgressTracker",
    "RichProgressTracker",
    "create_progress_tracker",
    "normalize_display_metrics",
    "HAS_RICH",
]
