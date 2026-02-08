"""
SSHTunnelConnector: Managed SSH tunnel via subprocess.

Uses ``ssh -L … -J … -N`` as a managed subprocess to:
- Leverage the user's existing SSH config, agent, and keys
- Support jump hosts (ProxyJump / -J flag)
- Optional explicit key path override
- Health checking via TCP probe on the local port

The tunnel runs as a child process and is torn down when disconnect()
is called or the parent process exits. No third-party dependencies
(paramiko, sshtunnel) are required.
"""

from __future__ import annotations

import logging
import signal
import socket
import subprocess
import time

from metalab.environment.connector import ConnectionTarget, TunnelHandle

logger = logging.getLogger(__name__)

# Tunnel establishment settings
_PROBE_TIMEOUT: float = 1.0  # TCP connect timeout per attempt
_PROBE_MAX_ATTEMPTS: int = 30  # Maximum number of probe attempts
_PROBE_INTERVAL: float = 1.0  # Seconds between probe attempts


def build_ssh_command(target: ConnectionTarget) -> list[str]:
    """
    Build an ``ssh`` tunnel command for the given connection target.

    Returns a list of arguments suitable for :func:`subprocess.Popen` or
    :func:`shlex.join` (for display to the user).

    Command form::

        ssh -N -L local_port:127.0.0.1:remote_port
            [-J user@gateway] [-i ssh_key] [user@]remote_host
    """
    cmd: list[str] = [
        "ssh",
        "-N",
        "-L",
        f"{target.local_port}:127.0.0.1:{target.remote_port}",
    ]

    # Jump host
    if target.gateway:
        gateway_spec = target.gateway
        if target.user and "@" not in gateway_spec:
            gateway_spec = f"{target.user}@{gateway_spec}"
        cmd.extend(["-J", gateway_spec])

    # Explicit key
    if target.ssh_key:
        cmd.extend(["-i", target.ssh_key])

    # Destination
    destination = target.remote_host
    if target.user:
        destination = f"{target.user}@{target.remote_host}"
    cmd.append(destination)

    return cmd


def _tcp_probe(host: str, port: int, timeout: float = _PROBE_TIMEOUT) -> bool:
    """
    Probe whether a TCP port is accepting connections.

    Args:
        host: Host to connect to.
        port: Port to connect to.
        timeout: Connection timeout in seconds.

    Returns:
        True if the port accepted the connection.
    """
    try:
        conn = socket.create_connection((host, port), timeout=timeout)
        conn.close()
        return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


class SSHTunnelConnector:
    """
    Connector that establishes SSH tunnels via subprocess.

    Builds and runs an ``ssh -L`` command to forward a local port to
    a remote host:port, optionally through a jump host. The tunnel
    process is managed as a subprocess and can be health-checked.

    Implements the Connector protocol.
    """

    def connect(self, target: ConnectionTarget) -> TunnelHandle:
        """
        Establish an SSH tunnel to the remote service.

        Spawns an ``ssh -N -L local_port:127.0.0.1:remote_port`` process
        and waits for the local port to become reachable via TCP probe.

        Args:
            target: Connection target with remote host/port and SSH options.

        Returns:
            A TunnelHandle for the active tunnel.

        Raises:
            RuntimeError: If the tunnel cannot be established within the
                probe timeout window.
        """
        cmd = build_ssh_command(target)
        logger.info("Starting SSH tunnel: %s", " ".join(cmd))

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait for the local port to become reachable
        local_host = "127.0.0.1"
        if not self._wait_for_tunnel(local_host, target.local_port, process):
            # Tunnel failed to come up — clean up and report
            stderr_output = ""
            if process.stderr:
                stderr_output = process.stderr.read().decode(errors="replace")
            process.kill()
            process.wait()
            raise RuntimeError(
                f"SSH tunnel failed to establish within "
                f"{_PROBE_MAX_ATTEMPTS * _PROBE_INTERVAL}s. "
                f"Command: {' '.join(cmd)}\n"
                f"SSH stderr: {stderr_output}"
            )

        logger.info(
            "SSH tunnel established: %s:%d -> %s:%d (pid=%d)",
            local_host,
            target.local_port,
            target.remote_host,
            target.remote_port,
            process.pid,
        )

        return TunnelHandle(
            local_host=local_host,
            local_port=target.local_port,
            remote_host=target.remote_host,
            remote_port=target.remote_port,
            pid=process.pid,
        )

    def disconnect(self, handle: TunnelHandle) -> None:
        """
        Tear down an SSH tunnel.

        Sends SIGTERM to the tunnel process, waits briefly, then
        SIGKILL if it hasn't exited.

        Args:
            handle: The tunnel handle returned by connect().
        """
        if handle.pid is None:
            return

        try:
            # Graceful shutdown
            os_signal = signal.SIGTERM
            _send_signal(handle.pid, os_signal)

            # Wait up to 5 seconds for clean exit
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _process_alive(handle.pid):
                    logger.info("SSH tunnel (pid=%d) terminated.", handle.pid)
                    return
                time.sleep(0.2)

            # Force kill
            logger.warning(
                "SSH tunnel (pid=%d) did not exit gracefully, sending SIGKILL.",
                handle.pid,
            )
            _send_signal(handle.pid, signal.SIGKILL)

        except ProcessLookupError:
            # Process already gone
            pass

    def is_alive(self, handle: TunnelHandle) -> bool:
        """
        Check whether the tunnel is still alive.

        Verifies both that the tunnel process is running and that
        the local port is accepting connections.

        Args:
            handle: The tunnel handle to check.

        Returns:
            True if the process is running and the local port is reachable.
        """
        if handle.pid is not None and not _process_alive(handle.pid):
            return False
        return _tcp_probe(handle.local_host, handle.local_port)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wait_for_tunnel(
        host: str,
        port: int,
        process: subprocess.Popen,  # type: ignore[type-arg]
    ) -> bool:
        """
        Wait for the tunnel's local port to become reachable.

        Returns False if the process exits or max attempts are exhausted.
        """
        for _ in range(_PROBE_MAX_ATTEMPTS):
            # Check if ssh process died
            if process.poll() is not None:
                return False

            if _tcp_probe(host, port):
                return True

            time.sleep(_PROBE_INTERVAL)

        return False


# ------------------------------------------------------------------
# Process management helpers
# ------------------------------------------------------------------


def _send_signal(pid: int, sig: signal.Signals) -> None:
    """Send a signal to a process by PID."""
    import os

    os.kill(pid, sig)


def _process_alive(pid: int) -> bool:
    """Check whether a process is still running."""
    import os

    try:
        os.kill(pid, 0)  # Signal 0 = existence check
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it
        return True
