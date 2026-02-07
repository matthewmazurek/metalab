"""
ServiceBundle: Tracks all provisioned services for an environment session.

The bundle.json file is the single source of truth for:
- What services are running (postgres, atlas)
- How to connect to them (host, port, credentials)
- How to tear them down (SLURM job IDs, PIDs)
- What store locator to use for discovery

Bundles are runtime artifacts (PIDs, ephemeral passwords) and always live
at ``~/.metalab/services/<project>/<env>/bundle.json``.  This fixed location
makes discovery trivial from any working directory.

Bundle lifecycle:
1. Environment provisions services → handles added to bundle
2. Bundle saved to ``~/.metalab/services/<project>/<env>/bundle.json``
   (permissions 0o600)
3. On reconnect, bundle is loaded and services are health-checked
4. On teardown, services are stopped and bundle file removed
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from metalab.environment.base import ServiceHandle


@dataclass
class ServiceBundle:
    """
    Collection of running services for an environment session.

    Attributes:
        environment: Environment type name (e.g., "slurm", "local").
        profile: Profile name used to create this bundle.
        services: Map of service name → ServiceHandle.
        store_locator: Store locator URI for the session (if applicable).
        tunnel_targets: Serialized tunnel targets for reconnection.
        created_at: ISO timestamp of bundle creation.
    """

    environment: str
    profile: str
    services: dict[str, ServiceHandle] = field(default_factory=dict)
    store_locator: str | None = None
    tunnel_targets: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add(self, name: str, handle: ServiceHandle) -> None:
        """Add a service to the bundle."""
        self.services[name] = handle

    def get(self, name: str) -> ServiceHandle | None:
        """Get a service handle by name."""
        return self.services.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "environment": self.environment,
            "profile": self.profile,
            "services": {
                name: handle.to_dict() for name, handle in self.services.items()
            },
            "store_locator": self.store_locator,
            "tunnel_targets": self.tunnel_targets,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServiceBundle:
        """Create from dictionary."""
        services = {
            name: ServiceHandle.from_dict(handle_data)
            for name, handle_data in data.get("services", {}).items()
        }
        return cls(
            environment=data["environment"],
            profile=data["profile"],
            services=services,
            store_locator=data.get("store_locator"),
            tunnel_targets=data.get("tunnel_targets", []),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )

    def save(self, path: Path) -> None:
        """
        Save to bundle.json file.

        Sets file permissions to 0o600 since the bundle may contain
        passwords or other credentials in service handles.

        Args:
            path: Path to write the bundle JSON file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        os.chmod(path, 0o600)

    @classmethod
    def load(cls, path: Path) -> ServiceBundle:
        """
        Load from bundle.json file.

        Args:
            path: Path to the bundle JSON file.

        Returns:
            A ServiceBundle instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    def remove(self, path: Path) -> None:
        """
        Remove bundle.json file.

        Args:
            path: Path to the bundle JSON file to remove.
        """
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Bundle directory (class-level constant)
    # ------------------------------------------------------------------

    BUNDLE_HOME = Path.home() / ".metalab" / "services"

    @classmethod
    def bundle_path_for(cls, project: str, env: str) -> Path:
        """
        Canonical bundle path for a project + environment.

        All bundles live under ``~/.metalab/services/<project>/<env>/bundle.json``.
        """
        return cls.BUNDLE_HOME / project / env / "bundle.json"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def find_nearest(cls, start_dir: Path | None = None) -> ServiceBundle | None:
        """
        Find and load the nearest bundle.json.

        Resolution order:

        1. Locate the nearest ``.metalab.toml`` (walking up from *start_dir*).
           Read the project name and environment names, then check
           ``~/.metalab/services/<project>/<env>/bundle.json`` for each.
        2. Scan ``~/.metalab/services/`` for any existing bundle (covers
           cases where no ``.metalab.toml`` is reachable).

        Args:
            start_dir: Directory to start searching from. Defaults to cwd.

        Returns:
            A ServiceBundle if found, None otherwise.
        """
        if start_dir is None:
            start_dir = Path.cwd()

        start_dir = start_dir.resolve()

        # Strategy 1: derive bundle path from .metalab.toml
        bundle = cls._find_from_config(start_dir)
        if bundle is not None:
            return bundle

        # Strategy 2: scan ~/.metalab/services/ for any bundle
        if cls.BUNDLE_HOME.is_dir():
            for project_dir in sorted(cls.BUNDLE_HOME.iterdir()):
                if not project_dir.is_dir():
                    continue
                for env_dir in sorted(project_dir.iterdir()):
                    candidate = env_dir / "bundle.json"
                    if candidate.is_file():
                        return cls.load(candidate)

        return None

    @classmethod
    def _find_from_config(cls, start_dir: Path) -> ServiceBundle | None:
        """
        Locate a bundle using project name and environments from ``.metalab.toml``.
        """
        from metalab.config import find_config_file

        config_path = find_config_file(start_dir)
        if config_path is None:
            return None

        try:
            import tomllib

            with open(config_path, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            return None

        project_name = data.get("project", {}).get("name") or "default"

        # Check each environment profile for a bundle
        for env_name in data.get("environments", {}):
            candidate = cls.bundle_path_for(project_name, env_name)
            if candidate.is_file():
                return cls.load(candidate)

        return None
