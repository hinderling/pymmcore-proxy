"""Shared fixtures: a DemoCamera core, a proxy server, and a remote client."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn

from pymmcore_proxy import ProxyServer, RemoteMMCore


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def demo_core():
    """A CMMCorePlus instance with the DemoCamera configuration loaded."""
    from pymmcore_plus import CMMCorePlus

    core = CMMCorePlus()
    core.loadSystemConfiguration("MMConfig_Demo.cfg")
    return core


@pytest.fixture(scope="session")
def server_url(demo_core):
    """Start a ProxyServer in a background thread, yield its base URL."""
    port = _free_port()
    proxy = ProxyServer(demo_core, port=port)

    config = uvicorn.Config(
        proxy.app, host="127.0.0.1", port=port, log_level="warning", ws="wsproto"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to accept connections
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("Proxy server failed to start")

    yield url

    server.should_exit = True
    thread.join(timeout=5.0)


@pytest.fixture()
def core(server_url):
    """A RemoteMMCore client connected to the test server."""
    client = RemoteMMCore(server_url, connect_signals=True)
    # Give the signal listener a moment to connect
    time.sleep(0.3)
    yield client
    client.close()


@pytest.fixture()
def core_no_signals(server_url):
    """A RemoteMMCore client without the signal listener (faster for RPC-only tests)."""
    client = RemoteMMCore(server_url, connect_signals=False)
    yield client
    client.close()
