"""
DirectConnector: No-op connector for local environments.

When services are already reachable on the local network (e.g., running
on localhost or on a directly-routable host), no tunneling is needed.
DirectConnector satisfies the Connector protocol by passing through
the service's host and port without establishing any tunnel process.
"""

from __future__ import annotations

from metalab.environment.connector import ConnectionTarget, TunnelHandle


class DirectConnector:
    """
    No-op connector that returns a handle pointing directly at the service.

    Used for local environments where services are already accessible
    without tunneling. Implements the Connector protocol.
    """

    def connect(self, target: ConnectionTarget) -> TunnelHandle:
        """
        Return a handle pointing directly at the remote service.

        No tunnel is established — the "local" endpoint is the
        remote endpoint itself.

        Args:
            target: Connection target (remote_host/port used directly).

        Returns:
            A TunnelHandle with local_host=remote_host, local_port=remote_port.
        """
        return TunnelHandle(
            local_host=target.remote_host,
            local_port=target.remote_port,
            remote_host=target.remote_host,
            remote_port=target.remote_port,
            pid=None,
        )

    def disconnect(self, handle: TunnelHandle) -> None:
        """No-op: no tunnel to tear down."""

    def is_alive(self, handle: TunnelHandle) -> bool:
        """Always returns True: no tunnel process to fail."""
        return True
