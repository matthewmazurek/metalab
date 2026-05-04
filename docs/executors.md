# Custom Executors

Executors receive an `ExperimentPlan` and return a `RunHandle`.

The runner owns experiment policy:

- resolve the context fingerprint
- compute stable run IDs
- decide resume skips from canonical successful records
- write `manifest.json`
- write `planned_batch` and `skipped` events

Executors own submission mechanics. A local executor may turn the plan into
per-run payloads. A scheduler executor may write one compact array spec and map
worker indexes back to parameters.

```python
from metalab.executor.base import ExperimentPlan
from metalab.executor.handle import RunHandle


class MyExecutor:
    def submit_experiment(self, plan: ExperimentPlan) -> RunHandle:
        payloads = plan.to_payloads()
        ...
```

Contract:

- return a `RunHandle`
- execute only `plan.pending_entries`
- preserve `plan.job_id` in worker events and heartbeats
- call `execute_payload(...)` or emit equivalent canonical records/events
- never treat event or cache state as resume truth

Canonical run records remain the source of truth. Events, heartbeats, status
cache, and DuckDB indexes are accelerators.
