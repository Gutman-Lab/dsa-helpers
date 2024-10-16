# Functions for working images.
import numpy as np
import pandas as pd
from pathlib import Path
import cv2 as cv
from . import imwrite


def tile_image(
    img: np.ndarray,
    save_loc: str,
    tile_size: int,
    stride: int | None = None,
    fill: int | tuple = (255, 255, 255),
    prepend_name: str = "",
    overwrite: bool = False,
    grayscale: bool = False,
) -> pd.DataFrame:
    """Tile an image into smaller images.

    Args:
        img (np.ndarray): The image to tile.
        save_loc (str): The location to save the tiles.
        tile_size (int): The size of the tiles.
        stride (int | None, optional): The stride for the tiles. Defaults to None,
            in which case it is set  equal to tile_size.
        fill (int | tuple, optional): The fill value for tiles over the edges.
            Defaults to (255, 255, 255).
        prepend_name (str, optional): A string to prepend to the tile names.
        overwrite (bool, optional): Whether to overwrite existing images.
            Defaults to False.
        grayscale (bool, optional): Whether to save images as grayscale instead
            of the default RGB. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame with the tile locations.

    """
    h, w = img.shape[:2]

    img = cv.copyMakeBorder(
        img, 0, tile_size, 0, tile_size, cv.BORDER_CONSTANT, value=fill
    )

    # Pad the image to avoid tiles not of the right size.
    save_loc = Path(save_loc)
    save_loc.mkdir(parents=True, exist_ok=True)

    if stride is None:
        stride = tile_size

    df_data = []

    xys = [(x, y) for x in range(0, w, stride) for y in range(0, h, stride)]

    if prepend_name:
        prepend_name += "_"

    for xy in xys:
        # Get the top left and bottom right coordinates of the tile.
        x, y = xy

        # Filepath for saving tile.
        save_fp = save_loc / f"{prepend_name}x{x}y{y}tilesize{tile_size}.png"

        if not save_fp.is_file() or overwrite:
            # Get the tile from the image.
            tile_img = img[y : y + tile_size, x : x + tile_size]

            imwrite(save_fp, tile_img, grayscale=grayscale)

        df_data.append([save_fp, x, y, tile_size])

    return pd.DataFrame(df_data, columns=["fp", "x", "y", "tile_size"])
