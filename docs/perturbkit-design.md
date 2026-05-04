# perturbkit Design Sketch

## Purpose

`perturbkit` is a proposed trajectory-first toolkit for analyzing gene
perturbation screens across cellular states.

The motivating use case is high-throughput gene perturbation screening run
through MetaLab and Dyno, followed by analysis of resultant cell trajectory
matrices. The broader goal is to make the analysis source-agnostic: the toolkit
should not assume Dyno, MetaLab, or any particular simulator produced the input.
Instead, it should accept standardized perturbation and trajectory data, then
provide an opinionated analysis workflow.

Core positioning:

```text
source-agnostic in, opinionated analysis out
```

## Non-Goals

- Do not run perturbation simulations.
- Do not perform general single-cell preprocessing.
- Do not replace trajectory inference tools.
- Do not become a generic gene perturbation analysis framework.
- Do not require MetaLab, Dyno, Streamlit, AnnData, or DuckDB as mandatory
  dependencies for the core analysis layer.
- Do not make the interactive app the canonical source of analysis logic.

The intended scope is specifically:

```text
gene perturbation effects on cell state trajectories
```

## Intended Users

- Computational researchers analyzing high-throughput perturbation screens.
- Wet-lab or nontechnical collaborators who need visual exploration and reports.
- Groups using trajectory simulators, trajectory inference tools, or experimental
  perturbation datasets that can be mapped into a common schema.
- Methods-paper readers who need reproducible examples, stable outputs, and
  documented assumptions.

## Design Principles

- Keep the core analysis engine independent from UI concerns.
- Use a stable internal data model after ingestion.
- Treat source-specific formats as adapters, not as internal truth.
- Make one default analysis workflow easy and well documented.
- Allow advanced users to swap scoring, alignment, aggregation, and visualization
  strategies without requiring a large plugin framework.
- Keep outputs portable, versioned, and inspectable.
- Prefer lazy loading for large trajectory matrices and eager loading only for
  summaries or explicitly requested subsets.
- Make the Streamlit app read analysis bundles rather than raw source data.

## High-Level Architecture

```text
Raw source
  MetaLab/Dyno, AnnData, Parquet, CSV, custom simulator
        |
        v
InputAdapter
        |
        v
PerturbationDataset
        |
        v
AnalysisPipeline
        |
        v
AnalysisBundle
        |
        v
Python API / CLI / Streamlit app / static report
```

## Core Concepts

### PerturbationDataset

The normalized in-memory or lazy representation of a perturbation trajectory
screen.

Expected concepts:

- perturbation identity, usually a gene or genetic construct
- observations, runs, cells, simulations, or replicates
- trajectory representation
- state labels or state probabilities
- control perturbation definitions
- time, pseudotime, or trajectory step index when available
- provenance describing the source format, tool versions, and analysis settings

The dataset should be constructible from multiple sources:

```python
import perturbkit as pk

dataset = pk.read_anndata("screen.h5ad")
dataset = pk.read_metalab("/path/to/run-store", adapter="dyno")
dataset = pk.read_bundle("analysis_bundle/")
dataset = pk.from_tables(obs=obs, trajectories=trajectories, perturbations=perturbations)
```

### AnalysisPipeline

An opinionated default workflow that converts a `PerturbationDataset` into
effect estimates, rankings, quality summaries, figures, and exports.

Possible stages:

1. Validate schema and required metadata.
2. Filter failed or low-quality observations.
3. Normalize or align trajectories.
4. Define controls and reference states.
5. Estimate perturbation effects.
6. Aggregate across replicates and seeds.
7. Quantify uncertainty.
8. Rank perturbations by state-specific or global effects.
9. Generate visual summaries.
10. Write an analysis bundle.

Example API:

```python
analysis = pk.analyze(
    dataset,
    perturbation_key="gene",
    state_key="cell_state",
    control="non_targeting",
    score="terminal_state_shift",
)

analysis.rank_genes(state="erythroid")
analysis.plot_state_effects()
analysis.write_bundle("bundle/")
```

### AnalysisBundle

A portable, versioned output directory produced by the analysis layer and
consumed by reports and the Streamlit app.

Tentative structure:

```text
analysis_bundle/
  manifest.json
  screen_metadata.parquet
  perturbations.parquet
  observations.parquet
  perturbation_effects.parquet
  state_effects.parquet
  trajectory_summaries.parquet
  replicate_quality.parquet
  rankings/
    genes.parquet
    states.parquet
  figures/
  report.html
  provenance/
    input_manifest.json
    analysis_config.yaml
    software_versions.json
```

The bundle is the primary interchange format for:

- static reports
- Streamlit app exploration
- archiving and sharing
- downstream analysis in notebooks
- paper examples and reproducibility material

## Interfaces

### Python API

Primary interface for computational users.

Responsibilities:

- load source data
- validate schema
- run analysis
- inspect derived tables
- generate plots
- write bundles

### CLI

Primary interface for reproducible workflows.

Possible commands:

```bash
perturbkit validate INPUT
perturbkit analyze INPUT --out bundle/
perturbkit report bundle/
perturbkit app bundle/
perturbkit export bundle/ --format csv
```

The CLI should make method parameters explicit and write them into bundle
provenance.

### Streamlit App

Friendly interface for nontechnical exploration.

Important constraint:

```text
The Streamlit app should open an AnalysisBundle, not raw arbitrary source data.
```

This keeps the app fast, stable, portable, and separate from analysis logic.

Expected app pages:

- QC overview
- perturbation ranking
- state-centric effects
- gene-centric effects
- perturbation comparison
- trajectory visualization
- export/download

### Static Report

Paper-friendly and collaborator-friendly HTML report generated from an
`AnalysisBundle`.

Expected contents:

- screen summary
- QC and replicate concordance
- top perturbations
- state transition effects
- trajectory deviation summaries
- uncertainty summaries
- method settings and software versions

## Tentative Package Structure

```text
perturbkit/
  __init__.py
  io/
    anndata.py
    metalab.py
    tables.py
    bundle.py
  schema/
    dataset.py
    trajectories.py
    effects.py
    provenance.py
  qc/
    validation.py
    replicate_quality.py
  transform/
    normalization.py
    alignment.py
    state_mapping.py
  effects/
    scoring.py
    aggregation.py
    uncertainty.py
  rank/
    genes.py
    states.py
  viz/
    heatmaps.py
    trajectories.py
    rankings.py
  report/
    html.py
    templates/
  app/
    streamlit_app.py
    pages/
  cli/
    main.py
```

## Swappable Strategy Points

The default workflow should be opinionated, but a few stages should eventually
be configurable:

- trajectory normalization
- trajectory alignment
- state mapping
- perturbation effect scoring
- replicate aggregation
- uncertainty estimation
- visualization style

These should begin as explicit string options or small callable protocols, not
as a broad plugin system.

Example:

```python
analysis = pk.analyze(
    dataset,
    score="trajectory_displacement",
    aggregation="bootstrap",
    reference="control",
)
```

## Candidate Effect Scores

Initial scoring methods to consider:

- terminal state shift
- trajectory displacement from control
- transition probability change
- state occupancy change over pseudotime
- distance to selected target state
- fate bias score
- replicate-consensus effect score

The first implementation should pick a small default set and document the
scientific estimand for each score.

## Relationship To MetaLab

MetaLab remains the experiment runner and canonical run-record system.
`perturbkit` should consume completed MetaLab outputs through an adapter.

MetaLab responsibilities:

- run high-throughput experiments
- preserve run records
- store artifacts
- optionally index/export records

`perturbkit` responsibilities:

- validate perturbation trajectory datasets
- normalize source outputs into a common schema
- estimate perturbation effects
- rank genes and states
- generate reports, exports, and interactive views

This separation keeps MetaLab domain-agnostic while allowing `perturbkit` to be
scientifically opinionated.

## Methods Paper Framing

The publishable contribution should not be framed as only a Streamlit app.
Instead:

> `perturbkit` provides a reproducible, source-agnostic workflow for summarizing
> gene perturbation screens as trajectory and cell-state effects, with
> programmatic, command-line, report, and interactive interfaces.

Possible paper components:

- formal data model
- default perturbation effect estimands
- replicate and control handling
- scalable analysis bundle format
- visual exploration interface
- case study using MetaLab and Dyno
- public example dataset

## Open Design Questions

1. What exact shapes should be supported for trajectory matrices?
   Examples: perturbation by state, cell by time by feature, latent coordinates
   over pseudotime, transition matrices, or state probability trajectories.

2. Should AnnData be a first-class adapter, an optional extra, or the preferred
   exchange format?

3. What is the primary default scientific estimand?
   Examples: terminal fate shift, trajectory deviation, transition probability
   change, or state occupancy change.

4. How should controls be represented?
   Single non-targeting control, multiple controls, matched controls, or
   user-defined reference groups.

5. Are perturbations always single-gene, or should the schema support
   combinatorial perturbations from the beginning?

6. How large are expected trajectory artifacts, and which operations must remain
   lazy or indexed?

7. Should the first prototype live inside the MetaLab repository, or should it
   begin as a sibling package with a MetaLab adapter?

## Suggested First Milestone

Build a minimal but coherent vertical slice:

1. Define `PerturbationDataset`.
2. Implement one table-based reader.
3. Implement one MetaLab/Dyno adapter.
4. Implement one default effect score.
5. Write an `AnalysisBundle`.
6. Generate one static HTML report.
7. Build a Streamlit app that opens only the bundle.
8. Add a small example dataset and tests around the schema and score.

The goal of the first milestone is not completeness. It is to prove that the
boundary between source ingestion, analysis, bundle output, and visual
exploration is healthy.
