import numpy as np
from napari_stream import send


def main():
    img = (np.random.rand(512, 512) * 4).astype("uint8")
    send([img, img])


if __name__ == "__main__":
    main()