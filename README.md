# napari-stream

[![License MIT](https://img.shields.io/pypi/l/napari-stream.svg?color=green)](https://github.com/Karol-G/napari-stream/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-stream.svg?color=green)](https://pypi.org/project/napari-stream)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-stream.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/Karol-G/napari-stream/branch/main/graph/badge.svg)](https://codecov.io/gh/Karol-G/napari-stream)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-stream)](https://napari-hub.org/plugins/napari-stream)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A plugin to send and receive image data for visualization in Napari from other applications.

`napari-stream` lets you push images from any application, process, or codebase into a running napari instance—even from another machine. It can automatically pull array-like data (NumPy, PyTorch tensors, zarr arrays, etc.) from nested Python lists and dicts, so you can stream complex structures without manual extraction. You can keep things private (local IPC/loopback) or make the receiver reachable publicly over TCP. The receiver endpoint can also be set via the `NAPARI_STREAM_ENDPOINT` environment variable.

## Quick usage

```python
from napari_stream.sender import StreamSender, send
import numpy as np

# Option 1: explicit sender (recommended when reusing across many sends)
sender = StreamSender(endpoint="tcp://192.0.2.10:5556")  # or leave None to use NAPARI_STREAM_ENDPOINT/default
sender.send(np.random.rand(256, 256), name="image")

# Option 2: convenience function; pass connection kwargs through
send(np.random.rand(64, 64), name="quick", endpoint="tcp://127.0.0.1:5556")
```

On the receiving side, open the napari dock widget, choose your endpoint, and toggle public access if you want to accept connections from other machines.


## Installation

You can install `napari-stream` via [pip]:

```
pip install napari-stream
```

If napari is not already installed, you can install `napari-stream` with napari and Qt via:

```
pip install "napari-stream[all]"
```



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-stream" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

# Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/vidata/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.

This repository was generated with [copier] using the [napari-plugin-template].

[copier]: https://copier.readthedocs.io/en/stable/
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
