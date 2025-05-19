"""Image utility functions for DSA Helpers.

This module provides various utility functions for image processing and
manipulation.

It includes functions for:
- Exctracting contours from binary masks, and formatting them into a
GeoDataFrame.

Note:
    This module requires external dependencies such as OpenCV and
    shapely.
"""

import numpy as np
import cv2 as cv
from .imread import imread

from shapely.affinity import translate
from shapely.geometry import Polygon


def label_mask_to_polygons(
    mask: str | np.ndarray,
    x_offset: int = 0,
    y_offset: int = 0,
    exclude_labels: int | list[int] = 0,
    min_area: int = 0,
) -> list[Polygon, int]:
    """
    Extract contours from a label mask and convert them into shapely
    polygons.

    Args:
        mask (str | np.ndarray): Path to the mask image or the mask.
        x_offset (int): Offset to add to x coordinates of polygons.
        y_offset (int): Offset to add to y coordinates of polygons.
        exclude_labels (int | list[int]): Label(s) to exclude from the
            output.
        min_area (int): Minimum area of polygons to include.

    Returns:
        list[Polygon, int]: List of polygons and their corresponding
            labels.

    """
    # Convert exclude_labels to a list if it's an integer.
    if isinstance(exclude_labels, int):
        exclude_labels = [exclude_labels]

    # Read the mask if it's a path.
    if isinstance(mask, str):
        mask = imread(mask, grayscale=True)

    # Unique labels, excluding those specified in exclude_labels.
    labels = [
        label for label in np.unique(mask) if label not in exclude_labels
    ]

    polygons = []  # track all polygons

    # Loop through unique label index.
    for label in labels:
        # Filter to mask for this label.
        label_mask = (mask == label).astype(np.uint8)

        # Find contours.
        contours, hierarchy = cv.findContours(
            label_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )

        # Process the contours.
        polygons_dict = {}

        for idx, (contour, h) in enumerate(zip(contours, hierarchy[0])):
            # The contour should contain at least three points, otherwise it's
            # not a polygon.
            if len(contour) > 3:
                # To include holes, group the polygons by their parent index.
                if idx not in polygons_dict:
                    polygons_dict[idx] = {"holes": []}

                if h[3] == -1:
                    # This does not have a parent so its a polygon.
                    polygons_dict[idx]["polygon"] = contour.reshape(-1, 2)
                else:
                    # This has a parent so its a hole.
                    polygons_dict[h[3]]["holes"].append(contour.reshape(-1, 2))

        # Convert to shapely polygon objects.
        for data in polygons_dict.values():
            if "polygon" in data:
                polygon = Polygon(data["polygon"], holes=data["holes"])

                # Shift the polygon by the offset.
                polygon = translate(polygon, xoff=x_offset, yoff=y_offset)

                # Include the polygon if it's greater than the minimum area.
                if polygon.area >= min_area:
                    polygons.append([polygon, int(label)])

    return polygons


def convert_label_mask_to_rgb(
    mask: np.ndarray,
    int2rgb: dict,
    alpha: bool = False,
    fill_color: tuple[int, int, int] | tuple[int, int, int, int] = (
        255,
        255,
        255,
    ),
) -> np.ndarray:
    """
    Convert a label mask to an RGB mask.

    Args:
        mask: Label mask to convert.
        int2rgb: Dictionary mapping integer labels to RGB colors. If
            passing alpha as True, then pass RGBA colors.
        alpha: Whether to include the alpha channel in the RGB mask.
        fill_color: Color to fill the mask with.

    Returns:
        (np.ndarray): RGB(A) mask.

    """
    # Create a new RGB mask of the same size as the label mask.
    rgb_mask = np.zeros(
        (mask.shape[0], mask.shape[1], 4 if alpha else 3), dtype=np.uint8
    )

    rgb_mask[:] = fill_color

    # Loop through the label mask.
    for label, rgb in int2rgb.items():
        rgb_mask[mask == label] = rgb

    return rgb_mask
