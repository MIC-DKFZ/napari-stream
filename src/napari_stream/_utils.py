from __future__ import annotations
import os
import socket

DEFAULT_TCP_PORT = 5556


def default_endpoint(public: bool = False) -> str:
    if public:
        return f"tcp://{_preferred_ip()}:{DEFAULT_TCP_PORT}"
    if os.name == "nt":  # Windows: prefer TCP
        return f"tcp://127.0.0.1:{DEFAULT_TCP_PORT}"
    # Unix: fast local IPC
    return "ipc:///tmp/napari_stream.sock"


def _preferred_ip() -> str:
    """Best-effort guess of a non-loopback IPv4 address for sharing endpoints."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET)
        for _, _, _, _, (addr, *_rest) in infos:
            if not addr.startswith("127."):
                return addr
    except Exception:
        pass
    return "127.0.0.1"