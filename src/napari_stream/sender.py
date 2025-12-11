from __future__ import annotations
import argparse
import json
import os
import time
from typing import Optional, Sequence, Any
from collections.abc import Mapping
import numpy as np
import zmq
from ._utils import default_endpoint


class StreamSender:
    """Send data to a napari receiver.

    Accepts:
      - numpy.ndarray
      - torch.Tensor  (detach().cpu().numpy())
      - blosc2.NDArray
      - zarr.Array
      - Python lists/tuples and dicts (recursively searched for arraylikes;
        nested Python lists of numbers are also converted to NumPy)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        high_water_mark: int = 10,
        linger_ms: int = 2000,
        ensure_delivery: bool = True,
    ):
        if endpoint is None:
            endpoint = os.environ.get("NAPARI_STREAM_ENDPOINT") or default_endpoint()
        self.endpoint = endpoint
        self.ensure_delivery = ensure_delivery
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUSH)
        self._sock.setsockopt(zmq.SNDHWM, high_water_mark)
        self._sock.setsockopt(zmq.LINGER, linger_ms)
        # Fail fast if there is no receiver instead of silently queueing.
        self._sock.setsockopt(zmq.IMMEDIATE, 1)
        self._sock.connect(self.endpoint)

    # ------------------------------ public API ------------------------------

    def send(
        self,
        array: Any,
        *,
        name: Optional[str] = None,
        colormap: Optional[str] = None,
        contrast_limits: Optional[Sequence[float]] = None,
        rgb: Optional[bool] = None,
        affine: Optional[np.ndarray] = None,
        scale: Optional[Sequence[float]] = None,
        translate: Optional[Sequence[float]] = None,
        opacity: Optional[float] = None,
        blending: Optional[str] = None,
        is_labels: bool = False,
    ) -> None:
        """Send one or many arrays.

        If `array` is a list/tuple/dict, recursively find every arraylike leaf and
        send each with a path-qualified name (e.g., `name[0][foo][2]`)."""
        base = name or "array"

        # If the input is a structure, traverse & send each leaf.
        if isinstance(array, Mapping) or (isinstance(array, (list, tuple)) and not isinstance(array, np.ndarray)):
            array = self.retrieve_array_like(base, array).items()
            array = dict(array)
            for path, arr_np in array.items():
                arr_np = self._to_numpy(arr_np)
                self._send_numpy(
                    arr_np,
                    name=path,
                    colormap=colormap,
                    contrast_limits=contrast_limits,
                    rgb=rgb,
                    affine=affine,
                    scale=scale,
                    translate=translate,
                    opacity=opacity,
                    blending=blending,
                    is_labels=is_labels,
                )
            return

        # Single object path
        arr_np = self._to_numpy(array)
        self._send_numpy(
            arr_np,
            name=base,
            colormap=colormap,
            contrast_limits=contrast_limits,
            rgb=rgb,
            affine=affine,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            is_labels=is_labels,
        )

    def close(self):
        try:
            self._sock.close()
        finally:
            self._sock = None

    # ------------------------------ internals ------------------------------

    def _send_numpy(
        self,
        arr: np.ndarray,
        *,
        name: Optional[str],
        colormap: Optional[str],
        contrast_limits: Optional[Sequence[float]],
        rgb: Optional[bool],
        affine: Optional[np.ndarray] = None,
        scale: Optional[Sequence[float]],
        translate: Optional[Sequence[float]],
        opacity: Optional[float],
        blending: Optional[str],
        is_labels: bool,
    ) -> None:
        # Ensure contiguous so memoryview is a single buffer
        if not (arr.flags["C_CONTIGUOUS"] or arr.flags["F_CONTIGUOUS"]):
            arr = np.ascontiguousarray(arr)
        order = "F" if arr.flags["F_CONTIGUOUS"] else "C"

        meta = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "order": order,
            "is_labels": bool(is_labels),
        }
        if name is not None:
            meta["name"] = name
        if colormap is not None:
            meta["colormap"] = colormap
        if contrast_limits is not None:
            meta["contrast_limits"] = list(map(float, contrast_limits))
        if rgb is not None:
            meta["rgb"] = bool(rgb)
        if affine is not None:
            A = np.asarray(affine, dtype=float).tolist()  # (N x N) arbitrary square
            meta["affine"] = A
        if scale is not None:
            meta["scale"] = list(map(float, scale))
        if translate is not None:
            meta["translate"] = list(map(float, translate))
        if opacity is not None:
            meta["opacity"] = float(opacity)
        if blending is not None:
            meta["blending"] = str(blending)

        header = json.dumps(meta).encode("utf-8")
        buf = memoryview(arr)  # zero-copy
        tracker = None
        last_err = None
        for _ in range(3):
            try:
                tracker = self._sock.send_multipart(
                    [header, buf],
                    flags=zmq.NOBLOCK,
                    copy=False,
                    track=self.ensure_delivery,
                )
                last_err = None
                break
            except zmq.Again as e:
                last_err = e
                # Allow a brief window for sockets to finish connecting
                time.sleep(0.05)
        if last_err is not None:
            raise RuntimeError(f"No receiver available at endpoint {self.endpoint}") from last_err
        if self.ensure_delivery and tracker is not None:
            tracker.wait()

    # ---- conversion helpers ----

    def _to_numpy(self, x: Any) -> np.ndarray:
        """Convert a supported object to np.ndarray without importing optional
        deps unless present. Supports numpy, torch (incl. torchvision.tv_tensors),
        blosc2 NDArray, zarr Array, and numeric Python sequences.
        """
        # Already NumPy
        if isinstance(x, np.ndarray):
            return x

        # --- PyTorch / torchvision.tv_tensors ---
        try:
            import torch  # type: ignore

            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()

            # torchvision.tv_tensors (Image, Mask, Video, BoundingBoxes, etc.)
            mod = getattr(type(x), "__module__", "")
            if mod.startswith("torchvision.tv_tensors"):
                try:
                    t = torch.as_tensor(x)
                except Exception:
                    t = getattr(x, "data", None)
                    if not isinstance(t, torch.Tensor):
                        raise
                return t.detach().cpu().numpy()
        except Exception:
            pass  # torch/torchvision not installed or not applicable

        # --- blosc2.NDArray ---
        try:
            import blosc2  # type: ignore
            if hasattr(blosc2, "NDArray") and isinstance(x, blosc2.NDArray):  # type: ignore[attr-defined]
                return np.asarray(x[:])  # materialize
            to_numpy = getattr(x, "to_numpy", None)
            if callable(to_numpy):
                try:
                    return np.asarray(to_numpy())
                except Exception:
                    pass
        except Exception:
            pass  # blosc2 not installed or not applicable

        # --- zarr arrays (v2/v3) ---
        try:
            import zarr  # type: ignore
            if isinstance(x, getattr(zarr, "Array", ())):
                return np.asarray(x[...])
            core = getattr(zarr, "core", None)
            if core is not None and isinstance(x, getattr(core, "Array", ())):
                return np.asarray(x[...])
        except Exception:
            pass  # zarr not installed or not applicable

        # --- Generic NumPy protocol objects ---
        if hasattr(x, "__array__") or (hasattr(x, "shape") and hasattr(x, "dtype")):
            arr = np.asarray(x)
            if isinstance(arr, np.ndarray) and arr.dtype != object:
                return arr

        # --- Numeric Python sequences (lists/tuples) ---
        if isinstance(x, (list, tuple)):
            try:
                arr = np.asarray(x)
                if isinstance(arr, np.ndarray) and arr.dtype != object:
                    return arr
            except Exception:
                pass

        raise TypeError(f"Unsupported array type: {type(x)!r}")

    def retrieve_array_like(self, name, obj):
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                result = self.retrieve_array_like(f"{name}[{key}]", value)
                new_obj.update(result)
            return new_obj
        elif isinstance(obj, list):
            new_obj = {}
            for i, value in enumerate(obj):
                result = self.retrieve_array_like(f"{name}[{i}]", value)
                new_obj.update(result)
            return new_obj
        elif self.is_arraylike(obj) and len(obj) >= 1:
            return {name: obj}
        else:
            return {}

    def is_arraylike(self, x):
        """Return True if `x` behaves like a NumPy array (NumPy, Torch, Zarr, Blosc2, etc.)."""
        import numpy as np

        if isinstance(x, (str, bytes, bytearray, dict, set)):
            return False
        if isinstance(x, (np.ndarray, np.generic)):
            return True

        try:
            import torch
            if isinstance(x, torch.Tensor):
                return True
            import torchvision.tv_tensors as tvt
            tv_tensor_types = tuple(
                getattr(tvt, name) for name in dir(tvt)
                if name and name[0].isupper() and hasattr(getattr(tvt, name), "__mro__")
            )
            if isinstance(x, tv_tensor_types):
                return True
        except Exception:
            pass

        try:
            import blosc2
            if hasattr(blosc2, "NDArray") and isinstance(x, blosc2.NDArray):
                return True
        except Exception:
            pass

        try:
            import zarr
            if isinstance(x, zarr.Array):
                return True
        except Exception:
            pass

        if hasattr(x, "__array__") or (hasattr(x, "shape") and hasattr(x, "dtype")):
            return True

        if isinstance(x, (list, tuple)):
            try:
                arr = np.asarray(x)
                return isinstance(arr, np.ndarray) and arr.dtype != object
            except Exception:
                return False

        return False


def send(*args, **kwargs) -> None:
    """Send data to a napari receiver.

    Accepts:
      - numpy.ndarray
      - torch.Tensor  (detach().cpu().numpy())
      - blosc2.NDArray
      - zarr.Array
      - Python lists/tuples and dicts (recursively searched for arraylikes;
        nested Python lists of numbers are also converted to NumPy)
    """
    sender_keys = ("endpoint", "high_water_mark", "linger_ms", "ensure_delivery")
    sender_kwargs = {k: kwargs.pop(k) for k in sender_keys if k in kwargs}
    sender = StreamSender(**sender_kwargs)
    sender.send(*args, **kwargs)


def _load_array(path: str, key: Optional[str] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
    lower = path.lower()
    is_medvol_candidate = lower.endswith((".nii", ".nii.gz", ".nrrd"))
    if is_medvol_candidate:
        from medvol import MedVol  # type: ignore
        mv = MedVol(path)
        return np.asarray(mv.array), np.asarray(mv.affine) if mv.affine is not None else None

    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = list(arr.keys())
        if key is None:
            if len(keys) != 1:
                raise ValueError(f"NPZ contains multiple arrays {keys}; specify --key.")
            key = keys[0]
        if key not in keys:
            raise ValueError(f"Key {key!r} not found in NPZ; available: {keys}")
        return np.asarray(arr[key]), None
    return np.asarray(arr), None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send an array to a napari-stream receiver.")
    parser.add_argument("path", help="Path to .npy/.npz or medical volume (.nii/.nii.gz/.nrrd).")
    parser.add_argument("--key", help="Array key inside .npz (if omitted and multiple arrays exist, will error).")
    parser.add_argument("--endpoint", help="Receiver endpoint (tcp://host:port). Falls back to NAPARI_STREAM_ENDPOINT or default.")
    parser.add_argument("--name", help="Layer name.")
    parser.add_argument("--colormap", help="Colormap name.")
    parser.add_argument("--contrast-limits", nargs=2, type=float, metavar=("LOW", "HIGH"), help="Contrast limits.")
    parser.add_argument("--rgb", action="store_true", help="Treat array as RGB.")
    parser.add_argument("--affine", help="Path to .npy/.npz containing affine matrix (overrides MedVol affine).")
    parser.add_argument("--scale", nargs="+", type=float, help="Scale factors.")
    parser.add_argument("--translate", nargs="+", type=float, help="Translation.")
    parser.add_argument("--opacity", type=float, help="Opacity.")
    parser.add_argument("--blending", help="Blending mode.")
    parser.add_argument("--labels", action="store_true", help="Send as labels layer.")
    parser.add_argument("--high-water-mark", type=int, default=10, help="Sender high water mark (default: 10).")
    parser.add_argument("--linger-ms", type=int, default=2000, help="Socket linger in ms (default: 2000).")
    parser.add_argument("--no-ensure-delivery", dest="ensure_delivery", action="store_false", help="Disable delivery tracking.")
    parser.set_defaults(ensure_delivery=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    arr, inferred_affine = _load_array(args.path, args.key)
    affine = None
    if args.affine:
        affine, _ = _load_array(args.affine)
    elif inferred_affine is not None:
        affine = inferred_affine

    sender_kwargs = {
        "endpoint": args.endpoint,
        "high_water_mark": args.high_water_mark,
        "linger_ms": args.linger_ms,
        "ensure_delivery": args.ensure_delivery,
    }

    send_kwargs = {
        "name": args.name,
        "colormap": args.colormap,
        "contrast_limits": args.contrast_limits,
        "rgb": True if args.rgb else None,
        "affine": affine,
        "scale": args.scale,
        "translate": args.translate,
        "opacity": args.opacity,
        "blending": args.blending,
        "is_labels": args.labels,
    }
    send_kwargs = {k: v for k, v in send_kwargs.items() if v is not None}
    send(arr, **send_kwargs, **sender_kwargs)


if __name__ == "__main__":
    main()
