# Preprocessing

When your experiment depends on preprocessed data, metalab provides `SeedBundle.for_preprocessing()` to ensure reproducibility across the full pipeline.

## Why a dedicated preprocessing seed?

Preprocessing often runs **before** `metalab.run()`, but if it involves randomness (train/test splits, shuffling, sampling), you need a reproducible seed that:

1. Doesn't collide with the experiment's replicate seeds
2. Changes when the base seed changes
3. Stays consistent across preprocessing reruns

## Minimal Example

```python
import metalab
from metalab import SeedBundle

BASE_SEED = 42

# ─── Preprocessing (runs once, before experiment) ───────────────────────
seeds = SeedBundle.for_preprocessing(BASE_SEED)
rng = seeds.numpy("train_test_split")

# Your preprocessing logic
train_idx, test_idx = train_test_split(
    range(len(data)), 
    test_size=0.2, 
    random_state=rng.integers(2**31)
)

# Include seed in filename for automatic cache invalidation
output_path = f"./cache/processed_seed{BASE_SEED}.h5ad"
save_processed(data, train_idx, test_idx, output_path)

# ─── Experiment (uses same base seed) ───────────────────────────────────
@metalab.context_spec
class DataContext:
    data_path: metalab.FilePath

@metalab.operation
def train_model(context, params, seeds, capture):
    data = load(str(context.data_path))
    rng = seeds.numpy("training")
    # ... training logic
    capture.metric("accuracy", acc)

exp = metalab.Experiment(
    name="my_experiment",
    operation=train_model,
    context=DataContext(data_path=metalab.FilePath(output_path)),
    params=metalab.grid(lr=[0.01, 0.001]),
    seeds=metalab.seeds(base=BASE_SEED, replicates=5),
)

metalab.run(exp)
```

## Best Practices

| Practice | Why |
|----------|-----|
| **Include seed in preprocessed filenames** | e.g., `processed_seed42.h5ad` — changing `BASE_SEED` automatically triggers cache miss and reprocessing |
| **Use the same `BASE_SEED` for both** | Ensures preprocessing and experiment share the same seed hierarchy |
| **Use named derivations** | `seeds.numpy("train_test_split")` vs `seeds.numpy()` — makes debugging easier and avoids accidental seed reuse |
| **Keep preprocessing deterministic** | Avoid hidden randomness (shuffling, sampling) outside the seeded RNG |

## Seed Collision Safety

`SeedBundle.for_preprocessing(42)` derives its root via `"42:preprocessing:0"`, while replicate bundles use `"42:replicate:0"`, `"42:replicate:1"`, etc. This guarantees no collision:

```python
# These produce different root seeds:
prep_bundle = SeedBundle.for_preprocessing(42)
rep0_bundle = metalab.seeds(base=42, replicates=1)[0]

assert prep_bundle.root_seed != rep0_bundle.root_seed  # ✓
```

## When to Use Preprocessing Seeds

Use `SeedBundle.for_preprocessing()` when your preprocessing involves:

- **Train/test/validation splits** — random partitioning of data
- **Subsampling** — selecting a random subset of records
- **Data augmentation** — generating synthetic variations
- **Shuffling** — randomizing order before batching
- **Imputation** — stochastic filling of missing values

If your preprocessing is fully deterministic (e.g., normalization, filtering by threshold), you don't need a preprocessing seed—but including `BASE_SEED` in the filename is still recommended for traceability.
