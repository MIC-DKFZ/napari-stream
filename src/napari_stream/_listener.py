from __future__ import annotations
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np
import zmq
from qtpy.QtCore import QObject, QThread, Signal


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


def default_endpoint() -> str:
    if os.name == "nt":  # Windows: prefer TCP
        return "tcp://127.0.0.1:5556"
    # Unix: fast local IPC
    return "ipc:///tmp/napari_stream.sock"


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