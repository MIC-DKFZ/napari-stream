import numpy as np

from napari_stream._receiver_widget import (
    ReceiverWidget,
    bind_endpoint_for_public,
    receiver_widget,
)


def test_bind_endpoint_for_public():
    assert bind_endpoint_for_public("tcp://127.0.0.1:5556") == "tcp://*:5556"
    assert bind_endpoint_for_public("tcp://*:5556") == "tcp://*:5556"
    assert bind_endpoint_for_public("ipc:///tmp/napari_stream.sock") == "ipc:///tmp/napari_stream.sock"


def test_receiver_widget_uses_existing_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = receiver_widget(viewer)
    assert isinstance(widget, ReceiverWidget)
    assert widget.viewer is viewer


def test_autocontrast_updates_layer_limits(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = receiver_widget(viewer)

    # simulate a received grayscale image without contrast_limits
    arr = np.random.rand(32, 32).astype(np.float32)
    meta = {"name": "foo", "rgb": False}
    widget._on_received(arr, meta)

    layer = viewer.layers["foo"]
    assert layer.contrast_limits is not None
