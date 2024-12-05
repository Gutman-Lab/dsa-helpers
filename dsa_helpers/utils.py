import numpy as np
from . import imread
import cv2 as cv

from shapely.geometry import Polygon
from shapely.affinity import translate


def mask_to_shapely(
    mask: str | np.ndarray,
    x_offset: int = 0,
    y_offset: int = 0,
    background_label: int = 0,
    min_area: int = 0,
) -> list[Polygon, int]:
    """
    Extract contours from a label mask and convert them into shapely
    polygons.

    Args:
        mask (str | np.ndarray): Path to the mask image or the mask.
        x_offset (int): Offset to add to x coordinates of polygons.
        y_offset (int): Offset to add to y coordinates of polygons.
        background_label (int): Label value of the background class,
            which is ignored.

    Returns:
        list[Polygon, int]: List of polygons and their corresponding
            labels.

    """
    if isinstance(mask, str):
        mask = imread(mask, grayscale=True)

    # Find unique labels (excluding background 0)
    labels = [label for label in np.unique(mask) if label != background_label]

    polygons = []  # Track all polygons.

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
            if len(contour) > 3:
                if idx not in polygons_dict:
                    polygons_dict[idx] = {"holes": []}

                if h[3] == -1:
                    polygons_dict[idx]["polygon"] = contour.reshape(-1, 2)
                else:
                    polygons_dict[h[3]]["holes"].append(contour.reshape(-1, 2))

        # Now that we know the polygon and the holes, create a Polygon object for each.
        for data in polygons_dict.values():
            if "polygon" in data:
                polygon = Polygon(data["polygon"], holes=data["holes"])

                # Shift the polygon by the offset.
                polygon = translate(polygon, xoff=x_offset, yoff=y_offset)

                # Skip small polygons.
                if polygon.area >= min_area:
                    polygons.append([polygon, label])

    return polygons
