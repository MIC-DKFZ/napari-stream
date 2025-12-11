import numpy as np
from napari_stream import send


def main():
    img = (np.random.rand(128, 128))
    send(img)


if __name__ == "__main__":
    main()