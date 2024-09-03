# Functions using the girder client.
from girder_client import GirderClient
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import cv2 as cv
from copy import deepcopy


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

    """
    get_url = f"item/{item_id}/tiles/"

    if mag is not None:
        get_url += f"region?magnification={mag}&encoding=pickle"
    else:
        # Instead use width and height.
        params = ["encoding=pickle"]

        if width is not None and height is not None:
            if isinstance(fill, (tuple, list)):
                if len(fill) == 3:
                    fill = f"rgb({fill[0]},{fill[1]},{fill[2]})"
                elif len(fill) == 4:
                    fill = f"rgba({fill[0]},{fill[1]},{fill[2]},{fill[3]})"

            params.extend([f"width={width}", f"height={height}", f"fill={fill}"])
        elif width is not None:
            params.append(f"width={width}")
        elif height is not None:
            params.append(f"height={height}")

        get_url += "thumbnail?" + "&".join(params)

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
    the image size. The outpuzerositem id.
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


def get_element_contours(element: dict) -> np.ndarray:
    """Get the contours of an element, regardless of the type.

    Args:
        element (dict): The element dictionary.

    Returns:
        np.ndarray: The contours of the element.

    """
    if element["type"] == "rectangle":
        return get_rectangle_element_coords(element)
    else:
        return None


def get_roi_images(
    gc: GirderClient,
    item_id: str,
    save_dir: str,
    roi_groups: str | list,
    doc_names: str | list | None = None,
    mag: float | None = None,
    rgb_pad: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    """Gets regions of interest (ROIs) as images from DSA annotations.

    Args:
        gc (girder_client.GirderClient): The authenticated girder client.
        item_id (str): The item id.
        save_dir (str): The directory to save the roi images.
        roi_groups (str | list): The roi group name or list of roi group names.
        doc_names (str | list, optional): List of documents to look for, if None
            then it looks at all documents. Defaults to None.
        mag (float, optional): The magnification to get the roi images. Defaults
            to None which returns the images at scan magnification.
        rgb_pad (tuple[int, int, int], optional): The RGB values to pad the image with.
            This only is used when the annotation is a rotated rectangle or a polygon.

    Returns:
        pd.DataFrame: The roi images metadata.

    """
    if isinstance(roi_groups, str):
        roi_groups = [roi_groups]

    if isinstance(doc_names, str):
        doc_names = [doc_names]

    # Convert to a comma separated string.
    roi_groups_str = ",".join(roi_groups)

    if doc_names is None:
        docs = gc.get(
            f"annotation?itemId={item_id}&text="
            f"{roi_groups_str}&limit=0&offset=0&sort=lowerName&sortdir=1"
        )
    else:
        docs = []

        for doc_name in doc_names:
            docs.extend(
                gc.get(
                    f"annotation?itemId={item_id}&text={roi_groups_str}&name={doc_name}&limit=0&offset=0&sort=lowerName&sortdir=1"
                )
            )

    # Create the location to save the images.
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Loop through each document.
    for doc in docs:
        # Get the full annotations.
        doc = gc.get(f"annotation/{doc['_id']}")

        # Get the elements.
        for element in doc.get("annotation", {}).get("elements", []):
            if element["group"] in roi_groups:
                # The annotations need to be handled based on type.
                contour = get_element_contours(element)

                # Get the minimum and maximum coordinates.
                xmin, ymin = contour.min(axis=0)
                xmax, ymax = contour.max(axis=0)
                w, h = xmax - xmin, ymax - ymin

                # Get the region of interest from the image.
                roi_image = get_region(gc, item_id, xmin, ymin, w, h, mag=mag)[
                    :, :, :3
                ]  # for now assuming images are RGB

                # Use pad if needed.
                if rgb_pad is not None:
                    roi_mask = np.ones((h, w), dtype=np.uint8)

                    # Draw the contours.
                    roi_mask = cv.drawContours(
                        roi_mask, [contour - (xmin, ymin)], -1, 0, cv.FILLED
                    )

                    roi_image[roi_mask == 1] = rgb_pad

                return roi_image, contour - (xmin, ymin)


def _rotate_point_list(point_list, rotation, center=(0, 0)):
    """Rotate a list of x, y points around a center location.
    Adapted from: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/annotations_and_masks/annotation_and_mask_utils.py
    INPUTS
    ------
    point_list : list
        list of x, y coordinates
    rotation : int or float
        rotation in radians
    center : list
        x, y location of center of rotation
    RETURN
    ------
    point_list_rotated : list
        list of x, y coordinates after rotation around center
    """
    point_list_rotated = []

    for point in point_list:
        cos, sin = np.cos(rotation), np.sin(rotation)
        x = point[0] - center[0]
        y = point[1] - center[1]

        point_list_rotated.append(
            (int(x * cos - y * sin + center[0]), int(x * sin + y * cos + center[1]))
        )

    return point_list_rotated


def get_rectangle_element_coords(element):
    """Get the corner coordinate from a rectangle HistomicsUI element, can handle rotated elements.
    Adapted from: https://github.com/DigitalSlideArchive/HistomicsTK/blob/master/histomicstk/annotations_and_masks/annotation_and_mask_utils.py
    INPUTS
    ------
    element : dict
        rectangle element, in HistomicsUI format
    RETURN
    ------
    corner_coords : array
        array of shape [4, 2] for the four corners of the rectangle in (x, y) format
    """
    # element is a dict so prevent referencing
    element = deepcopy(element)

    # calculate the corner coordinates, assuming no rotation
    center_x, center_y = element["center"][:2]
    h, w = element["height"], element["width"]
    x_min = center_x - w // 2
    x_max = center_x + w // 2
    y_min = center_y - h // 2
    y_max = center_y + h // 2
    corner_coords = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]

    # if there is rotation rotate
    if element["rotation"]:
        corner_coords = _rotate_point_list(
            corner_coords, rotation=element["rotation"], center=(center_x, center_y)
        )

    corner_coords = np.array(corner_coords, dtype=np.int32)

    return corner_coords


def login(
    api_url: str,
    login_or_email: str | None = None,
    password: str | None = None,
    api_key: str | None = None,
) -> GirderClient:
    """Authenticate a girder client with the given credentials or interactively
    if none is given.

    Args:
        api_url (str): The DSA girder API url.
        login_or_email (str | None): The login or email. Defaults to None.
        password (str | None): Password for login / email. Defaults to None.
        api_key (str | None): The api key to authenticate with. Defaults to None.

    Returns:
        girder_client.GirderClient: The authenticated girder client.

    """
    gc = GirderClient(apiUrl=api_url)

    if api_key is None:
        if login_or_email is None:
            _ = gc.authenticate(interactive=True)
        elif password is None:
            _ = gc.authenticate(username=login_or_email, interactive=True)
        else:
            _ = gc.authenticate(username=login_or_email, password=password)
    else:
        _ = gc.authenticate(apiKey=api_key)

    return gc
