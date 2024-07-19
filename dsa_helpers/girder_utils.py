# Functions using the girder client.
from girder_client import GirderClient
import pickle
import numpy as np


def get_item_large_image_metadata(gc: GirderClient, item_id: str) -> dict:
    """Get large image metadata for an item.

    Args:
        gc (girder_client.GirderClient): The authenticated girder client.
        item_id (str): The item id.

    Returns:
        dict: The metadata of the large image item.

    """
    return gc.get(f"item/{item_id}/tiles")


def get_thumbnail(
    gc: GirderClient,
    item_id: str,
    mag: float | None = None,
    width: int | None = None,
    height: int | None = None,
    fill: int | tuple = (255, 255, 255),
) -> np.ndarray:
    """Get the thumbnail image by a specific magnification or shape. If mag is
    not None, then width and height are ignored. Fill is only used when both
    width and height are provided, to return the thumbnail at the exact shape.
    DSA convention will fill the height of the image, centering the image and
    filling the top and bottom of the image equally.

    Args:
        gc (girder_client.GirderClient): The authenticated girder client.
        item_id (str): The item id.
        mag (float, optional): The magnification. Defaults to None.
        width (int, optional): The width of the thumbnail. Defaults to None.
        height (int, optional): The height of the thumbnail. Defaults to None.
        fill (int | tuple, optional): The fill color. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: The thumbnail image.

    Raises:
        ValueError: If neither mag, width, nor height is provided.

    """
    if all([mag is None, width is None, height is None]):
        raise ValueError("Either mag, width, or height must be provided.")

    if mag is not None:
        get_url = f"item/{item_id}/tiles/region?magnification{mag}"
    else:
        # Instead use width and height.
        get_url = f"item/{item_id}/tiles/thumbnail?"

        if width is not None and height is not None:
            if isinstance(fill, (tuple, list)):
                if len(fill) == 3:
                    fill = f"rgb({fill[0]},{fill[1]},{fill[2]})"
                elif len(fill) == 4:
                    fill = f"rgba({fill[0]},{fill[1]},{fill[2]},{fill[3]})"

            get_url += f"width={width}&height={height}&fill={fill}"
        elif width is not None:
            get_url += f"width={width}"
        else:
            get_url += f"height={height}"

    get_url += "&encoding=pickle"

    response = gc.get(get_url, jsonResp=False)

    return pickle.loads(response.content)


def get_region(
    gc: GirderClient,
    item_id: str,
    left: int,
    top: int,
    width: int,
    height: int,
    mag: float | None = None,
) -> np.ndarray:
    """Get a region of the image for an item. Note that the output image might
    not be in the shape (width, height) if left + width or top + height exceeds
    the image size. The output image will be the maximum size possible.

    Args:
        gc (girder_client.GirderClient): The authenticated girder client.
        item_id (str): The item id.
        left (int): The left coordinate.
        top (int): The top coordinate.
        width (int): The width of the region.
        height (int): The height of the region.
        mag (float, optional): The magnification. Defaults to None which returns
            the image at scan magnification. Using a mag lower than the scan
            magnification will result in an ouptut image smaller than the
            width and height. Similarly, using a mag higher than the scan
            magnification will result in an output image larger than the width
            and height.

    Returns:
        np.ndarray: The region of the image.

    """
    get_url = (
        f"item/{item_id}/tiles/region?left={left}&top={top}&regionWidth="
        f"{width}&regionHeight={height}&encoding=pickle"
    )

    if mag is not None:
        get_url += f"&magnification={mag}"

    response = gc.get(get_url, jsonResp=False)

    return pickle.loads(response.content)
