"""
Progress tracking module for metalab experiments.

Provides rich progress bars and status displays for experiment execution.
Falls back to simple text output if rich is not installed.

Example:
    import metalab
    from metalab.progress import create_progress_tracker, MetricDisplay

    exp = build_experiment()

    # Option 1: Simple boolean (auto-detects rich)
    handle = metalab.run(exp, progress=True)
    results = handle.result()  # Shows live progress bar

    # Option 2: Progress config object for customization
    handle = metalab.run(
        exp,
        progress=metalab.Progress(
            title="My Experiment",
            display_metrics=["best_f:.2f", "converged"],
        ),
    )
    results = handle.result()

    # Option 3: Custom event handling
    def my_logger(event):
        print(f"{event.kind}: {event.run_id}")

    handle = metalab.run(exp, on_event=my_logger)
    results = handle.result()

    # Option 4: Manual tracker control (advanced)
    handle = metalab.run(exp)
    with create_progress_tracker(total=handle.status.total) as tracker:
        handle.set_event_callback(tracker)
        results = handle.result()
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
