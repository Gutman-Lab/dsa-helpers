# Function to save image to disk.
import cv2 as cv
import numpy as np


def imwrite(fp: str, img: np.ndarray, grayscale: bool = False) -> None:
    """Save an image to disk.

    Args:
        img (np.ndarray): The image to save.
        fp (str): The file path to save the image.
        grayscale (bool, optional): Whether to save the image as grayscale.
            Defaults to False.

    Raises:
        ValueError: The image must be grayscale, RGB, or RGBA.

    """
    if grayscale:
        if len(img.shape) == 2:
            _ = cv.imwrite(fp, img)
        else:
            _ = cv.imwrite(fp, cv.cvtColor(img, cv.COLOR_RGB2GRAY))
    else:
        shape = img.shape

        if len(shape) == 2:
            _ = cv.imwrite(fp)
        elif len(shape) == 3:
            if shape[2] == 3:
                _ = cv.imwrite(fp, cv.cvtColor(img, cv.COLOR_RGB2BGR))
            elif shape[2] == 4:
                _ = cv.imwrite(fp, cv.cvtColor(img, cv.COLOR_RGBA2BGRA))
            else:
                raise ValueError("The image must be grayscale, RGB, or RGBA.")
        else:
            raise ValueError("The image must be grayscale, RGB, or RGBA.")
