# Installation

metalab requires **Python 3.11 or higher**.

## Basic Installation

Install metalab using [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# Using uv
uv add metalab

# Using pip
pip install metalab
```

The base package has no required dependencies—it's lightweight by design.

## Optional Extras

metalab provides optional extras for extended functionality:

| Extra | Dependencies | Use Case |
|-------|-------------|----------|
| `numpy` | numpy ≥1.24 | Array serialization for `capture.data()` and `capture.artifact()` |
| `pandas` | pandas ≥2.0 | DataFrame export helpers (`results.to_dataframe()`, `results.to_csv()`) |
| `rich` | rich ≥13.0 | Rich progress bars and nicer CLI output |
| `postgres` | psycopg ≥3.0, psycopg-pool ≥3.0 | PostgreSQL store backend for large experiments |
| `full` | All of the above | Everything bundled together |

Install extras with brackets:

```bash
# Using uv
uv add metalab[numpy]
uv add metalab[pandas]
uv add metalab[rich]
uv add metalab[postgres]
uv add metalab[full]

# Using pip
pip install metalab[numpy]
pip install metalab[pandas]
pip install metalab[rich]
pip install metalab[postgres]
pip install metalab[full]
```

You can combine multiple extras:

```bash
uv add metalab[numpy,pandas,rich]
```

### When to Use Each Extra

- **numpy**: Required if your experiments capture numpy arrays via `capture.data()` or `capture.artifact()`
- **pandas**: Required if you want to export results as DataFrames with `results.to_dataframe()`
- **rich**: Recommended for better visual feedback during experiment runs
- **postgres**: Required for large-scale experiments where you need fast queries across many runs

For most users getting started, we recommend:

```bash
uv add metalab[pandas,rich]
```

This gives you nice progress output and easy DataFrame exports for analysis.

## Development Setup

To contribute to metalab or run the test suite, clone the repository and install with development dependencies:

```bash
git clone https://github.com/matthewmazurek/metalab.git
cd metalab

# Install all dependencies including dev tools
uv sync

# Run the test suite
uv run pytest
```

The development environment includes:

- **pytest** and **pytest-cov** for testing
- **numpy** and **pandas** for testing serializers and exports
- **rich** for testing progress displays
- **ruff** for linting
- **pyright** for type checking

## Verifying Installation

After installation, verify everything works:

```python
import metalab

print(metalab.__version__)  # Should print "0.1.0"
```

To check if optional extras are available:

```python
# Check numpy support
try:
    import numpy
    print("numpy available")
except ImportError:
    print("numpy not installed")

# Check pandas support
try:
    import pandas
    print("pandas available")
except ImportError:
    print("pandas not installed")
```

## Next Steps

Once installed, head to the [Quickstart](quickstart.md) to run your first experiment.
