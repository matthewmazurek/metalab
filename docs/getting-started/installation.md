# Installation

metalab requires **Python 3.11 or higher**.

## Basic Installation

Install metalab using [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# Using uv (recommended)
uv add git+https://github.com/matthewmazurek/metalab.git

# Using pip
pip install git+https://github.com/matthewmazurek/metalab.git
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
| `atlas` | metalab-atlas | Atlas web UI for browsing experiment results |
| `full` | All of the above | Everything bundled together |

Install extras with brackets:

```bash
# Using uv
uv add "metalab[numpy] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[pandas] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[rich] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[postgres] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[atlas] @ git+https://github.com/matthewmazurek/metalab.git"
uv add "metalab[full] @ git+https://github.com/matthewmazurek/metalab.git"

# Using pip
pip install "metalab[numpy] @ git+https://github.com/matthewmazurek/metalab.git"
pip install "metalab[pandas] @ git+https://github.com/matthewmazurek/metalab.git"
pip install "metalab[rich] @ git+https://github.com/matthewmazurek/metalab.git"
pip install "metalab[postgres] @ git+https://github.com/matthewmazurek/metalab.git"
pip install "metalab[atlas] @ git+https://github.com/matthewmazurek/metalab.git"
pip install "metalab[full] @ git+https://github.com/matthewmazurek/metalab.git"
```

You can combine multiple extras:

```bash
uv add "metalab[numpy,pandas,rich] @ git+https://github.com/matthewmazurek/metalab.git"
```

### When to Use Each Extra

- **numpy**: Required if your experiments capture numpy arrays via `capture.data()` or `capture.artifact()`
- **pandas**: Required if you want to export results as DataFrames with `results.to_dataframe()`
- **rich**: Recommended for better visual feedback during experiment runs
- **postgres**: Required for large-scale experiments where you need fast queries across many runs
- **atlas**: Installs the Atlas web UI for browsing and visualizing experiment results

For most users getting started, we recommend:

```bash
uv add "metalab[pandas,rich] @ git+https://github.com/matthewmazurek/metalab.git"
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
