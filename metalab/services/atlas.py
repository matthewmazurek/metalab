"""
Atlas dashboard service providers.

Provides :class:`SlurmFragment` and :class:`LocalFragment` definitions
so that the Atlas dashboard can be composed into any supported
environment without that environment knowing Atlas-specific details.

The Atlas provider sets ``metadata["tunnel_target"]`` when a gateway is
configured in the environment, allowing the orchestrator to discover
tunnel targets generically.
"""

from __future__ import annotations

import os
from typing import Any


def slurm_provider(spec: Any, env_config: dict[str, Any]) -> Any:
    """Produce a :class:`SlurmFragment` for the Atlas dashboard.

    Generates a bash fragment that starts ``uvicorn atlas.main:app``
    in the foreground (keeping the SLURM job alive).  When the
    environment has a ``gateway`` configured, the resulting handle's
    ``metadata["tunnel_target"]`` is set so the orchestrator can open
    a tunnel.
    """
    from metalab.environment.base import ReadinessCheck, ServiceHandle
    from metalab.environment.slurm import SlurmFragment

    port = spec.config.get("port", 8000)
    store = spec.config.get("store", "")
    file_root = spec.config.get("file_root") or env_config.get("file_root", "")

    setup_bash = f"""
echo "Starting Atlas on $HOSTNAME:{port}"
export ATLAS_STORE_PATH="{store}"
export ATLAS_FILE_ROOT="{file_root}"

python -m uvicorn atlas.main:app --host 0.0.0.0 --port {port}
"""

    gateway = env_config.get("gateway")
    _port = port
    _gateway = gateway

    def _build_handle(job_id: str, hostname: str) -> ServiceHandle:
        metadata: dict[str, Any] = {}
        if _gateway:
            metadata["tunnel_target"] = {
                "host": hostname,
                "remote_port": _port,
                "local_port": _port,
            }
        return ServiceHandle(
            name="atlas",
            host=hostname,
            port=_port,
            process_id=job_id,
            metadata=metadata,
        )

    return SlurmFragment(
        name="atlas",
        setup_bash=setup_bash,
        cleanup_bash="",  # Atlas is foreground; job exit handles it
        readiness=ReadinessCheck(port=port),
        cpus=1,
        build_handle=_build_handle,
    )


def local_provider(spec: Any, env_config: dict[str, Any]) -> Any:
    """Produce a :class:`LocalFragment` for the Atlas dashboard.

    Builds the uvicorn command and environment variables.  No tunnel
    target metadata is set for local environments (no gateway).
    """
    from metalab.environment.base import ReadinessCheck, ServiceHandle
    from metalab.environment.local import LocalFragment

    port = spec.config.get("port", 8000)
    store = spec.config.get("store", "")
    file_root = spec.config.get("file_root") or env_config.get("file_root", "")
    host = spec.config.get("host", "127.0.0.1")

    cmd = [
        "python", "-m", "uvicorn", "atlas.main:app",
        "--host", host, "--port", str(port),
    ]

    env: dict[str, str] = {
        "ATLAS_STORE_PATH": store,
    }
    if file_root:
        env["ATLAS_FILE_ROOT"] = str(file_root)

    _port = port
    _host = host

    def _build_handle(pid: str, hostname: str) -> ServiceHandle:
        return ServiceHandle(
            name="atlas",
            host=_host,
            port=_port,
            process_id=pid,
            metadata={},
        )

    return LocalFragment(
        name="atlas",
        command=cmd,
        env=env,
        readiness=ReadinessCheck(port=port),
        log_name="atlas",
        build_handle=_build_handle,
    )


# Auto-register providers
try:
    from metalab.services.registry import register_provider as _register

    _register("atlas", "slurm", slurm_provider)
    _register("atlas", "local", local_provider)
except ImportError:
    pass
