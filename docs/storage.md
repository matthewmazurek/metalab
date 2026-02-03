# Storage

metalab supports filesystem storage out of the box and optional PostgreSQL for large-scale experiments.

## FileStore (default)

```python
metalab.run(exp, store="./runs/my_exp")
```

## PostgresStore (optional)

```python
metalab.run(
    exp,
    store="postgresql://user@localhost/metalab?experiments_root=/shared/experiments",
)
```

Install support:

```bash
uv add metalab[postgres]
```

## Postgres Architecture

- Run records and structured data in PostgreSQL
- Logs and artifacts on the filesystem (shared path)
- Fast access to `capture.data(...)` for derived metrics

## Fallback Store

If Postgres becomes unavailable, you can fall back to a FileStore automatically.

```python
metalab.run(exp, store="postgresql://localhost/db", fallback=True)
```

## Store Transfer

```bash
metalab store export --from postgresql://localhost/db --to ./runs/export
metalab store import --from ./runs/my_exp --to postgresql://localhost/db
```
