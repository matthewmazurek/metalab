"""
Connector protocols: Making remote services accessible locally.

When services run on remote hosts (e.g., SLURM compute nodes), a Connector
establishes local access — typically via SSH tunnels. The protocol is simple:

- connect(): establish a route from local_port to remote_host:remote_port
- disconnect(): tear down the route
- is_alive(): check if the route is still active

Implementations:
- DirectConnector: No-op for services already on localhost
- SSHTunnelConnector: SSH -L tunnel via subprocess
- Future: kubectl port-forward, cloud IAP tunnels, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ConnectionTarget:
    """
    Target for a tunnel connection.

    Describes where the remote service is and how to reach it.

    Attributes:
        remote_host: Hostname or IP of the remote service.
        remote_port: Port the remote service is listening on.
        local_port: Local port to bind for the tunnel.
        gateway: Optional jump host (SSH ProxyJump / -J flag).
        user: SSH username for the connection.
        ssh_key: Path to an SSH private key (overrides agent/config).
    """

    remote_host: str
    remote_port: int
    local_port: int
    gateway: str | None = None
    user: str | None = None
    ssh_key: str | None = None


@dataclass
class TunnelHandle:
    """
    Handle for an active tunnel.

    Returned by Connector.connect() and used to manage the tunnel lifecycle.

    Attributes:
        local_host: Local address the tunnel is bound to (usually "127.0.0.1").
        local_port: Local port the tunnel is listening on.
        remote_host: Remote host being tunneled to.
        remote_port: Remote port being tunneled to.
        pid: Process ID of the tunnel process (if applicable).
    """

    local_host: str
    local_port: int
    remote_host: str
    remote_port: int
    pid: int | None = None

    @property
    def local_url(self) -> str:
        """URL for the local end of the tunnel."""
        return f"http://{self.local_host}:{self.local_port}"


@runtime_checkable
class Connector(Protocol):
    """
    Protocol for making remote services accessible locally.

    Connector implementations handle the details of establishing
    network routes between the local machine and remote services.
    """

    def connect(self, target: ConnectionTarget) -> TunnelHandle:
        """
        Establish a connection to the remote service.

        Args:
            target: Connection target describing the remote endpoint.

        Returns:
            A TunnelHandle for the active connection.

        Raises:
            RuntimeError: If the connection cannot be established.
        """
        ...

    def disconnect(self, handle: TunnelHandle) -> None:
        """
        Tear down an active connection.

        Args:
            handle: The tunnel handle returned by connect().
        """
        ...

    def is_alive(self, handle: TunnelHandle) -> bool:
        """
        Check whether a connection is still active.

        Args:
            handle: The tunnel handle to check.

        Returns:
            True if the connection is still alive and usable.
        """
        ...
