from __future__ import annotations
import traceback
from typing import Optional

import numpy as np
from qtpy.QtCore import QThread
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QHBoxLayout,
    QCheckBox,
)

from ._listener import ZMQImageListener, bind_endpoint_for_public, default_endpoint

try:
    from napari.types import ImageData
    from napari import Viewer
except Exception:  # pragma: no cover
    Viewer = object  # type: ignore[misc,assignment]
    ImageData = np.ndarray  # type: ignore[assignment]


class ReceiverWidget(QWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer

        self._thread: Optional[QThread] = None
        self._worker: Optional[ZMQImageListener] = None

        self._last_auto_endpoint = default_endpoint()
        self.endpoint_edit = QLineEdit(self._last_auto_endpoint)
        self.status_label = QLabel("Idle")
        self.public_access = QCheckBox("Enable public Access")
        self.autocontrast = QCheckBox("Auto-contrast on new images")
        self.autocontrast.setChecked(True)

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_copy = QPushButton("Copy Endpoint")

        top = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Endpoint:"))
        row.addWidget(self.endpoint_edit)
        row.addWidget(self.btn_copy)
        top.addLayout(row)
        top.addWidget(self.public_access)
        top.addWidget(self.autocontrast)
        top.addWidget(self.status_label)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_start)
        row2.addWidget(self.btn_stop)
        top.addLayout(row2)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_copy.clicked.connect(self._copy_endpoint)
        self.public_access.stateChanged.connect(self._on_public_toggled)

    def _on_start(self):
        endpoint = self.endpoint_edit.text().strip()
        bind_endpoint = self._resolve_endpoint_for_worker(endpoint)
        self._thread = QThread()
        self._worker = ZMQImageListener(bind_endpoint)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start)
        self._worker.received.connect(self._on_received)
        self._worker.status.connect(self.status_label.setText)
        self._worker.error.connect(self._on_error)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._thread.start()

    def _on_stop(self):
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_public_toggled(self, checked: int):
        is_public = bool(checked)
        suggested = default_endpoint(public=is_public)
        current = self.endpoint_edit.text().strip()
        if current == self._last_auto_endpoint:
            self.endpoint_edit.setText(suggested)
        self._last_auto_endpoint = suggested

    def _copy_endpoint(self):
        endpoint = self.endpoint_edit.text().strip()
        QGuiApplication.clipboard().setText(endpoint)

    def _resolve_endpoint_for_worker(self, endpoint: str) -> str:
        if not self.public_access.isChecked():
            return endpoint
        if endpoint.startswith("tcp://"):
            return bind_endpoint_for_public(endpoint)
        # If the field holds a non-TCP endpoint, fall back to a sensible TCP default.
        fallback = default_endpoint(public=True)
        self.endpoint_edit.setText(fallback)
        self._last_auto_endpoint = fallback
        return bind_endpoint_for_public(fallback)

    def _on_received(self, arr: np.ndarray, meta: dict):
        name = meta.get("name", "array")
        is_labels = bool(meta.get("is_labels", False))

        # Build kwargs common + per-layer-type
        viewer_kwargs = {}

        # Affine: accept any square >= 2x2 (2x2, 3x3, 4x4, ...)
        if "affine" in meta:
            try:
                A = np.asarray(meta["affine"], dtype=float)
                if A.ndim == 2 and A.shape[0] == A.shape[1] and A.shape[0] >= 2:
                    viewer_kwargs["affine"] = A
            except Exception:
                pass

        # Shared kwargs (supported by both images and labels)
        for key in ("scale", "translate", "opacity", "blending"):
            if key in meta:
                viewer_kwargs[key] = meta[key]

        if is_labels:
            # Labels-specific: do NOT pass image-only args like colormap/contrast_limits/rgb
            layer = self.viewer.add_labels(arr, name=name, **viewer_kwargs)
        else:
            # Image-specific kwargs
            for key in ("colormap", "contrast_limits", "rgb"):
                if key in meta:
                    viewer_kwargs[key] = meta[key]
            layer = self.viewer.add_image(arr, name=name, **viewer_kwargs)

            # Optional autocontrast for grayscale images without provided limits
            if self.autocontrast.isChecked() and "contrast_limits" not in meta and not meta.get("rgb", False):
                try:
                    lo, hi = np.percentile(arr[~np.isnan(arr)], (1, 99))
                    layer.contrast_limits = (float(lo), float(hi))
                except Exception:
                    pass

    def _on_error(self, msg: str):
        self.status_label.setText("Error — see console")
        print("[napari-ipc-bridge] Listener error:\n" + msg)
        traceback.print_exc()


def receiver_widget(viewer=None) -> ReceiverWidget:
    """npe2 command entrypoint: returns a QWidget dock widget.

    Works both when launched from VS Code/example and when starting napari
    from the console, even if type injection doesn't occur. If `viewer`
    is not injected, we try to fetch the current active viewer; as a
    last resort we create a new one.
    """
    try:
        if viewer is None:
            import napari
            viewer = napari.current_viewer() or napari.Viewer()
    except Exception:
        # Extremely defensive: create a viewer if current_viewer failed
        import napari
        viewer = napari.Viewer()
    return ReceiverWidget(viewer)
