from __future__ import annotations
import socket

DEFAULT_TCP_PORT = 5556


def default_endpoint(public: bool = False) -> str:
    if public:
        return f"tcp://{_preferred_ip()}:{DEFAULT_TCP_PORT}"
    # TCP everywhere for simplicity
    return f"tcp://127.0.0.1:{DEFAULT_TCP_PORT}"


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
