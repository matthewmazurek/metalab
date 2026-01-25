# Gene Perturbation Example

This example demonstrates in silico gene perturbation analysis using [dynamo](https://dynamo-release.readthedocs.io/) with metalab for experiment orchestration.

Based on the dynamo perturbation tutorial: https://dynamo-release.readthedocs.io/en/stable/introduction/perturbation_tutorial/perturbation_tutorial.html

## Overview

The experiment perturbs key hematopoietic transcription factors (KLF1, SPI1, GATA1) in the scNT-seq hematopoiesis dataset and captures:

- **Streamline plots**: Visualizing velocity field after perturbation
- **State graphs**: Cell type transition network
- **Group graph matrices**: Transition probabilities between cell types

## Isolated Environment

This example uses `dynamo-release` which depends on deprecated versions of numpy (<2.0), pandas (<2.0), matplotlib (<3.9), and anndata (<0.11). To avoid polluting the main metalab environment, this example runs as a **uv workspace member** with its own isolated dependencies.

## Usage

All commands should be run from the **repository root**.

### Quick Test (Light)

```bash
uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --intensity light
```

Runs: 3 genes × 3 perturbation values × 1 replicate = **9 runs**

### Medium Experiment

```bash
uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --intensity medium --workers 4
```

Runs: 3 genes × 7 values × 2 replicates = **42 runs**

### Heavy Experiment

```bash
uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --intensity heavy --workers 8
```

Runs: 3 genes × 21 values × 3 replicates = **189 runs**

### Single Gene Investigation

```bash
uv run --package gene-perturbation-example python examples/gene_perturbation/run.py --gene KLF1
```

## Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `gene` | KLF1, SPI1, GATA1 | Target transcription factor |
| `perturbation_value` | [-1000, 1000] | Perturbation magnitude (negative = suppression, positive = activation) |
| `group_by` | cell_type | Grouping variable for state graph |

## Biological Background

From the dynamo documentation:

- **GATA1**: Master regulator of MEP (megakaryocyte-erythroid progenitor) lineage
- **SPI1/PU.1**: Master regulator of GMP (granulocyte-monocyte progenitor) lineage
- **KLF1**: Erythroid-specific factor; activation drives erythroid differentiation

GATA1 and SPI1 form a mutual inhibition network motif that specifies cell fate choice between these lineages.

## Artifacts

Each run generates:

1. **streamline_plot.png**: Velocity streamlines after perturbation
2. **state_graph.png**: Cell type transition network
3. **group_graph.npz**: Transition probability matrix between cell types
4. **summary.json**: Run metadata and summary statistics

## Metrics

| Metric | Description |
|--------|-------------|
| `n_cells` | Number of cells in dataset |
| `n_genes` | Number of genes |
| `n_groups` | Number of cell type groups |
| `diagonal_sum` | Sum of self-transitions |
| `off_diagonal_sum` | Sum of inter-group transitions |
| `max_transition` | Maximum transition probability |

## Example Analysis

After running experiments, you can analyze results:

```python
import metalab
from pathlib import Path

# Load results
result = metalab.ResultHandle(store=metalab.FileStore("./runs/gene_perturbation"))
df = result.table(as_dataframe=True)

# Compare transition patterns across genes
for gene in ["KLF1", "SPI1", "GATA1"]:
    gene_runs = df[df["gene"] == gene]
    print(f"\n{gene}:")
    print(f"  Mean diagonal sum: {gene_runs['diagonal_sum'].mean():.3f}")
    print(f"  Mean off-diagonal sum: {gene_runs['off_diagonal_sum'].mean():.3f}")

# Load a specific artifact
run_id = df.loc[df["gene"] == "KLF1", "run_id"].iloc[0]
group_graph = result.load(run_id, "group_graph")
print(f"Group graph shape: {group_graph['matrix'].shape}")
print(f"Labels: {group_graph['labels']}")
```
