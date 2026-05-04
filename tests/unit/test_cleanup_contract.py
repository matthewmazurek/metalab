from __future__ import annotations

from pathlib import Path


def test_package_metadata_has_no_retired_backends():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    retired = [
        "metalab.stores",
        "service_plugins",
        "psycopg",
        "postgres_index",
        "metalab-atlas",
    ]

    for needle in retired:
        assert needle not in pyproject


def test_package_code_has_no_retired_backend_references():
    retired = [
        "metalab.services",
        "metalab.environment",
        "store.postgres",
        "postgres_index",
        "ssh_tunnel",
        "metalab-atlas",
    ]

    for path in Path("metalab").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in retired:
            assert needle not in text, f"{needle!r} still referenced in {path}"
