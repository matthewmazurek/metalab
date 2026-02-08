"""
AtlasPlugin: Class-based service plugin for the Atlas dashboard.

Provides SLURM and local providers for the Atlas uvicorn-based dashboard.
"""

from __future__ import annotations

from typing import Any

from metalab.services.base import ServicePlugin


class AtlasPlugin(ServicePlugin):
    """Service plugin for the Atlas dashboard."""

    name = "atlas"

    # ------------------------------------------------------------------
    # plan_slurm
    # ------------------------------------------------------------------

    def plan_slurm(
        self,
        spec: Any,
        env_config: dict[str, Any],
    ) -> Any:
        """Return a :class:`SlurmFragment` for the Atlas dashboard.

        Generates a bash fragment that starts ``uvicorn atlas.main:app``
        in the background.  The composed sbatch script's keep-alive loop
        keeps the SLURM job running.  When the environment has a
        ``gateway`` configured, the resulting handle's
        ``metadata["tunnel_target"]`` is set so the orchestrator can open
        a tunnel.
        """
        from metalab.environment.base import ReadinessCheck, ServiceHandle
        from metalab.environment.slurm import SlurmFragment

        port = spec.config.get("port", 8000)
        store = spec.config.get("store", "")
        file_root = spec.config.get("file_root") or env_config.get("file_root", "")
        svc_dir = f"{file_root}/services/atlas" if file_root else "/tmp"

        setup_bash = f"""
echo "Starting Atlas on $HOSTNAME:{port}"
export ATLAS_STORE_PATH="{store}"
export ATLAS_FILE_ROOT="{file_root}"
mkdir -p "{svc_dir}"

python -m uvicorn atlas.main:app --host 0.0.0.0 --port {port} \
    > "{svc_dir}/atlas.log" 2>&1 &
ATLAS_PID=$!
echo "Atlas PID: $ATLAS_PID"
"""

        gateway = env_config.get("gateway")
        _port = port
        _gateway = gateway

        _svc_dir = svc_dir

        def _build_handle(job_id: str, hostname: str) -> ServiceHandle:
            metadata: dict[str, Any] = {
                "log_file": f"{_svc_dir}/atlas.log",
            }
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
            cleanup_bash='if [ -n "${ATLAS_PID:-}" ]; then kill "$ATLAS_PID" 2>/dev/null || true; fi',
            readiness=ReadinessCheck(port=port),
            cpus=1,
            build_handle=_build_handle,
        )

    # ------------------------------------------------------------------
    # plan_local
    # ------------------------------------------------------------------

    def plan_local(
        self,
        spec: Any,
        env_config: dict[str, Any],
    ) -> Any:
        """Return a :class:`LocalFragment` for the Atlas dashboard.

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
