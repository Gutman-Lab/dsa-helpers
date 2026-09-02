# Function to read image from disk.
from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

cv = lazy.load("cv2")
np = lazy.load("numpy")

if TYPE_CHECKING:
    import numpy as np


def imread(fp: str, grayscale: bool = False) -> np.ndarray:
    """Read image from disk. Currently does not support reading RGBA images.

    Args:
        fp (str): The file path to save the image.
        grayscale (bool, optional): If True, the image will be read as grayscale.
            Defaults to False.

    Returns:
        np.ndarray: The image read from disk.

    """
    if grayscale:
        img = cv.imread(fp, cv.IMREAD_GRAYSCALE)
    else:
        img = cv.imread(fp, cv.IMREAD_UNCHANGED)

        # Convert the image from OpenCV's BGR to RGB.
        shape = img.shape

        if len(shape) == 3:
            if shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            elif shape[2] == 4:
                img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)

    return img
