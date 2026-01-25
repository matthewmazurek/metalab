# AGENTS.md

## Purpose
metalab is a general experiment runner: (FrozenContext, Params, SeedBundle) -> RunRecord + Artifacts.
Backends (execution + storage) are pluggable. Domain logic stays in user Operations.

## Non-goals
- No domain-specific assumptions (ML/bio/etc.)
- No opinionated storage policy: users decide what to capture
- No hard dependency on distributed frameworks

## Core invariants (do not break)
- Context is treated as read-only; operations must not mutate it.
- Params inputs are immutable; resolve/derive into new objects.
- All randomness must be controlled via SeedBundle.
- run_id is stable and derived from experiment + context + params + seed fingerprints.
- Artifacts are emitted via capture; do not return large objects from Operation.run.
- Executor boundary payloads must be serializable or reconstructable via manifests.
- Core orchestration must remain backend-agnostic.

## Dev environment (uv)
- Python: 3.13
- Install: `uv sync`
- Test: `uv run pytest`

## Git workflow
- Branches: feat/*, fix/*, chore/*
- Commits: imperative, scoped when helpful
- PRs must include tests for new behavior and doc updates for API changes

## Contracts
- Operation.run(context, params, seeds, runtime, capture) -> RunRecord
- ParamSource.iter() -> ParamCase(params, case_id, tags?)
- ContextBuilder.build(ContextSpec) -> FrozenContext
- Executor.submit(payload) -> Future; Executor.gather(futures) -> [RunRecord]
- Store.put_run_record(record); Store.put_artifact(...) -> ArtifactDescriptor
- Capture.metric(s)/artifact/file/log

RunRecord required fields: run_id, experiment_id, status, context_fingerprint,
params_fingerprint, seed_fingerprint, timestamps, metrics, provenance, error?

## Testing (pytest)
- Unit: canonicalization, hashing, param generation
- Contract: seed discipline, context immutability, payload serialization round-trip
- Integration: end-to-end run with ThreadExecutor + FileStore, including resume/dedupe

No network. Avoid flaky timing assumptions. Use tmp_path.

## Adding plugins
- New ParamSource: metalab/params/, add unit tests
- New Executor: metalab/executor/, add integration test
- New Serializer: metalab/capture/serializers/, add round-trip test
- New Store: metalab/store/, add integration test (artifacts + run records)
