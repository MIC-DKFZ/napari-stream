from __future__ import annotations
import json
import os
import sys
import socket
from typing import Optional

import numpy as np
import zmq
from qtpy.QtCore import QObject, QThread, Signal


DEFAULT_TCP_PORT = 5556


class ZMQImageListener(QObject):
    """Background worker that pulls image frames over ZeroMQ.

    Emits `received(array, meta)` whenever an image arrives.
    """

    received = Signal(object, dict)
    status = Signal(str)
    error = Signal(str)

    def __init__(self, endpoint: Optional[str] = None, high_water_mark: int = 10):
        super().__init__()
        self._endpoint = endpoint or default_endpoint()
        self._hwm = high_water_mark
        self._ctx: Optional[zmq.Context] = None
        self._sock: Optional[zmq.Socket] = None
        self._running = False

    def start(self):
        try:
            self._ctx = zmq.Context.instance()
            self._sock = self._ctx.socket(zmq.PULL)
            self._sock.setsockopt(zmq.RCVHWM, self._hwm)
            self._sock.setsockopt(zmq.LINGER, 0)

            # Bind as a local server that senders connect to
            self._sock.bind(self._endpoint)
            self._running = True
            self.status.emit(f"Listening on {self._endpoint}")

            poller = zmq.Poller()
            poller.register(self._sock, zmq.POLLIN)

            while self._running:
                events = dict(poller.poll(timeout=100))  # 100 ms tick for responsiveness
                if self._sock in events:
                    try:
                        header_b, payload = self._sock.recv_multipart(flags=zmq.NOBLOCK)
                        meta = json.loads(header_b.decode("utf-8"))
                        arr = _from_bytes(payload, meta)
                        self.received.emit(arr, meta)
                    except zmq.Again:
                        pass
                    except Exception as e:  # noqa: BLE001
                        self.error.emit(repr(e))
        except Exception as e:  # noqa: BLE001
            self.error.emit(repr(e))
        finally:
            self._teardown()

    def stop(self):
        self._running = False
        self.status.emit("Stopping listener…")

    def _teardown(self):
        try:
            if self._sock is not None:
                self._sock.close(0)
        finally:
            self._sock = None
            self._ctx = None
            self.status.emit("Listener stopped.")


def default_endpoint(public: bool = False) -> str:
    if public:
        return f"tcp://{_preferred_ip()}:{DEFAULT_TCP_PORT}"
    if os.name == "nt":  # Windows: prefer TCP
        return f"tcp://127.0.0.1:{DEFAULT_TCP_PORT}"
    # Unix: fast local IPC
    return "ipc:///tmp/napari_stream.sock"


def bind_endpoint_for_public(endpoint: str) -> str:
    """Convert a TCP endpoint to bind on all interfaces."""
    if not endpoint.startswith("tcp://"):
        return endpoint
    try:
        host_port = endpoint[len("tcp://"):]
        host, port = host_port.rsplit(":", 1)
        port_int = int(port)
    except Exception:
        return endpoint
    if host in ("*", "0.0.0.0"):
        return endpoint
    return f"tcp://*:{port_int}"


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


def _from_bytes(buf: bytes, meta: dict) -> np.ndarray:
    """Reconstruct ndarray from raw bytes and metadata.

    Expected `meta` keys: shape, dtype, order ("C"/"F").
    Optional: For clarity, we also pass through viewer kwargs like name/colormap/etc.
    """
    shape = tuple(meta["shape"])  # e.g. [Z, Y, X] or [Y, X]
    dtype = np.dtype(meta["dtype"])  # e.g. "float32"
    order = meta.get("order", "C")

    arr = np.frombuffer(buf, dtype=dtype)
    arr = arr.reshape(shape, order=order)
    return arr
