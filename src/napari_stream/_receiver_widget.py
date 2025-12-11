from __future__ import annotations
import traceback
import shutil
import subprocess
from typing import Optional, Tuple
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
    QRadioButton,
    QButtonGroup,
)

from ._listener import ZMQImageListener, bind_endpoint_for_public
from ._utils import DEFAULT_TCP_PORT, default_endpoint

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

        # Mode-specific endpoints
        self._endpoint_local = f"tcp://127.0.0.1:{DEFAULT_TCP_PORT}"
        self._endpoint_private = default_endpoint(public=True)
        self._endpoint_tunnel = ""

        self._tunnel_proc: Optional[subprocess.Popen] = None
        self._last_mode = "local"

        self.endpoint_edit = QLineEdit(self._endpoint_local)
        self.status_label = QLabel("Idle")
        self.mode_local = QRadioButton("Local only")
        self.mode_private = QRadioButton("Private network")
        self.mode_tunnel = QRadioButton("Private network (Reverse SSH Tunnel)")
        self.mode_local.setChecked(True)
        self.mode_group = QButtonGroup(self)
        for btn in (self.mode_local, self.mode_private, self.mode_tunnel):
            self.mode_group.addButton(btn)

        self.autocontrast = QCheckBox("Auto-contrast on new images")
        self.autocontrast.setChecked(True)
        self.ignore_affine = QCheckBox("Ignore affine")

        self.btn_run = QPushButton("Start")
        self.btn_copy = QPushButton("Copy Endpoint")

        top = QVBoxLayout(self)
        top.addWidget(QLabel("Endpoint:"))
        top.addWidget(self.endpoint_edit)
        top.addWidget(self.btn_copy)
        top.addWidget(self.mode_local)
        top.addWidget(self.mode_private)
        top.addWidget(self.mode_tunnel)
        top.addWidget(self.autocontrast)
        top.addWidget(self.ignore_affine)
        top.addWidget(self.status_label)
        row2 = QHBoxLayout()
        row2.addWidget(self.btn_run)
        top.addLayout(row2)

        self.btn_run.clicked.connect(self._on_toggle_clicked)
        self.btn_copy.clicked.connect(self._copy_endpoint)
        self.mode_group.buttonClicked.connect(self._on_mode_changed)
        self.destroyed.connect(lambda *_: self._stop_tunnel())
        app = QGuiApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(lambda *_: self._stop_tunnel())

    def _on_toggle_clicked(self):
        if self._is_running():
            self._on_stop()
        else:
            self._on_start()

    def _on_start(self):
        if self._is_running():
            return
        mode = self._current_mode()
        user_entry = self.endpoint_edit.text().strip()
        endpoint = user_entry or self._endpoint_for_mode(mode)
        if mode == "local":
            endpoint = self._ensure_local_endpoint(endpoint, mode="local")
        elif mode == "private":
            endpoint = self._ensure_private_endpoint(endpoint)
        elif mode == "tunnel":
            ssh_target, ssh_port = self._parse_ssh_target(user_entry)
            if not ssh_target:
                self.status_label.setText("Enter SSH target for reverse tunnel.")
                return
            endpoint = self._ensure_local_endpoint(
                self._endpoint_tunnel or f"tcp://127.0.0.1:{DEFAULT_TCP_PORT}",
                mode="tunnel",
                update_field=False,
            )
            self._store_endpoint_for_mode("tunnel", user_entry)
        else:
            self.status_label.setText("Unknown mode.")
            return

        self._thread = QThread()
        self._worker = ZMQImageListener(endpoint)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.start)
        self._worker.received.connect(self._on_received)
        self._worker.status.connect(self.status_label.setText)
        self._worker.error.connect(self._on_error)

        self.btn_run.setText("Stop")
        self.btn_run.setEnabled(True)
        self._thread.start()

        if mode == "tunnel":
            if not self._start_reverse_tunnel(ssh_target, ssh_port, endpoint):
                self._on_stop()

    def _on_stop(self):
        self._stop_tunnel()
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
        self._thread = None
        self._worker = None
        self.btn_run.setText("Start")
        self.btn_run.setEnabled(True)

    def _copy_endpoint(self):
        endpoint = self.endpoint_edit.text().strip()
        QGuiApplication.clipboard().setText(endpoint)

    def _on_mode_changed(self, *_):
        # Stop if running to allow reconfiguration
        if self._is_running():
            self._on_stop()
        # Persist current text to the previous mode slot
        prev_mode = getattr(self, "_last_mode", None)
        if prev_mode:
            self._store_endpoint_for_mode(prev_mode, self.endpoint_edit.text().strip())
        mode = self._current_mode()
        self.endpoint_edit.setPlaceholderText("user@remote or ssh-alias[#port]" if mode == "tunnel" else "")
        self.endpoint_edit.setText(self._endpoint_for_mode(mode))
        self._last_mode = mode
        self.status_label.setText("Idle")

    def _current_mode(self) -> str:
        if self.mode_private.isChecked():
            return "private"
        if self.mode_tunnel.isChecked():
            return "tunnel"
        return "local"

    def _endpoint_for_mode(self, mode: str) -> str:
        if mode == "private":
            return self._endpoint_private
        if mode == "tunnel":
            # Show blank to prompt SSH target entry; fall back to stored value if present
            return "" if not self._endpoint_tunnel else self._endpoint_tunnel
        return self._endpoint_local

    def _store_endpoint_for_mode(self, mode: str, endpoint: str) -> None:
        if mode == "private":
            self._endpoint_private = endpoint or self._endpoint_private
        elif mode == "tunnel":
            self._endpoint_tunnel = endpoint or self._endpoint_tunnel
        else:
            self._endpoint_local = endpoint or self._endpoint_local

    def _ensure_local_endpoint(self, endpoint: str, *, mode: str = "local", update_field: bool = True) -> str:
        port = self._extract_port(endpoint) or DEFAULT_TCP_PORT
        value = f"tcp://127.0.0.1:{port}"
        if update_field:
            self.endpoint_edit.setText(value)
            self._store_endpoint_for_mode(mode, value)
        return value

    def _ensure_private_endpoint(self, endpoint: str) -> str:
        # Prefer a shareable IP, then bind on all interfaces.
        if not endpoint.startswith("tcp://"):
            endpoint = self._endpoint_private or default_endpoint(public=True)
        if endpoint.count(":") < 2:
            # Missing port: append default
            endpoint = endpoint.rstrip("/") + f":{DEFAULT_TCP_PORT}"
        if not endpoint.startswith("tcp://"):
            endpoint = "tcp://" + endpoint
        self._endpoint_private = endpoint
        self.endpoint_edit.setText(endpoint)
        return bind_endpoint_for_public(endpoint)

    def _extract_port(self, endpoint: str) -> Optional[int]:
        if not endpoint:
            return None
        try:
            _, port_str = endpoint.rsplit(":", 1)
            return int(port_str)
        except Exception:
            return None

    def _parse_ssh_target(self, raw: str) -> Tuple[str, int]:
        raw = (raw or "").strip()
        if not raw:
            return "", 22
        if "#" in raw:
            target, port_part = raw.rsplit("#", 1)
            try:
                port = int(port_part)
            except Exception:
                port = 22
            return target or "", port
        return raw, 22

    def _start_reverse_tunnel(self, target: str, ssh_port: int, endpoint: str) -> bool:
        self._stop_tunnel()
        cmd = shutil.which("autossh") or shutil.which("ssh")
        if cmd is None:
            self.status_label.setText("autossh/ssh not found in PATH.")
            return False
        local_port = self._extract_port(endpoint) or DEFAULT_TCP_PORT
        remote_port = local_port
        args = [cmd]
        if cmd.endswith("autossh"):
            args += ["-M", "0"]
        args += ["-N", "-R", f"{remote_port}:localhost:{local_port}"]
        if ssh_port != 22:
            args += ["-p", str(ssh_port)]
        args.append(target)
        try:
            self._tunnel_proc = subprocess.Popen(
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=False,
            )
            self.status_label.setText(f"Reverse tunnel via {target} (port {ssh_port})")
            return True
        except Exception as exc:  # noqa: BLE001
            self._tunnel_proc = None
            self.status_label.setText(f"Failed to start tunnel: {exc}")
            return False

    def _stop_tunnel(self, *_):
        if self._tunnel_proc is None:
            return
        try:
            self._tunnel_proc.terminate()
            self._tunnel_proc.wait(timeout=2)
        except Exception:
            try:
                self._tunnel_proc.kill()
            except Exception:
                pass
        finally:
            self._tunnel_proc = None

    def _is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _restart_listener(self):
        self._on_stop()
        self._on_start()

    def _on_received(self, arr: np.ndarray, meta: dict):
        name = meta.get("name", "array")
        is_labels = bool(meta.get("is_labels", False))

        # Build kwargs common + per-layer-type
        viewer_kwargs = {}

        # Affine: accept any square >= 2x2 (2x2, 3x3, 4x4, ...)
        if "affine" in meta and not self.ignore_affine.isChecked():
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
