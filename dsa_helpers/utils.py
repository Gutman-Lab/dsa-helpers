"""Utility functions for DSA Helpers.

This module provides various miscellaneous utility functions that are
not grouped into their own modules.
"""

from shapely.geometry import Polygon
import numpy as np
import cv2 as cv


def return_mag_and_resolution(
    mag: int | None = None,
    mm_px: float | None = None,
    standard_mag: int = 40,
    standard_mm_px: float = 0.0002519,
) -> tuple[float, float]:
    """Given an input magnification or resolution, return the other
    based on the standard values defined by standard_mag and
    standard_mm_per_px.

    Args:
        mag (int | None, optional): Magnification to use. Defaults to
            None.
        mm_px (float | None, optional): Resolution to use. Defaults to
            None.
        standard_mag (int, optional): Standard magnification. Matches
            the standard resolution provided. Defaults to 40.
        standard_mm_px (float, optional): Standard resolution. Matches
            the standard magnification provided. Defaults to 0.0002519.

    Returns:
        tuple[float, float, float]: Returns the magnification and
            resolution of the image. These are standardized to the
            standard values provided in the inputs.

    """
    assert (
        mag is not None or mm_px is not None
    ), "Either mag or mm_px must be provided."
    assert (
        mag is None or mm_px is None
    ), "Only one of mag or mm_px can be provided."

    if mm_px is None:
        # Use the magnification to calculate the resolutions.
        mm_px = standard_mm_px * standard_mag / mag
    else:
        # Use the resolution to calculate the magnification.
        mag = standard_mag * standard_mm_px / mm_px

    return mag, mm_px


def remove_small_holes(
    polygon: Polygon, hole_area_threshold: float
) -> Polygon:
    """Remove small holes from a shapely polygon.

    Args:
        polygon (shapely.geometry.Polygon): Polygon to remove holes
            from.
        hole_area_threshold (float): Minimum area of a hole to keep it.

    Returns:
        shapely.geometry.Polygon: Polygon with small holes removed.

    """
    if not polygon.interiors:  # if there are no holes, return as is
        return polygon

    # Filter out small holes
    new_holes = [
        hole
        for hole in polygon.interiors
        if Polygon(hole).area > hole_area_threshold
    ]

    # Create a new polygon with only large holes
    return Polygon(polygon.exterior, new_holes)


def convert_to_json_serializable(
    data: int | float | str | list | dict,
) -> int | float | str | list | dict:
    """Convert a list, integer, float, or dictionary into a JSON
    serializable version of the object. Uses recursion to make sure the
    entire input data structure is converted to a JSON serializable
    version.

    Args:
        data (int | float | str | list | dict): Data to convert to
            JSON serializable.

    Returns:
        int | float | str | list | dict: JSON serializable version of
            the input data.

    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    return data


def contours_to_polygons(binary_image: np.ndarray) -> list[Polygon]:
    """Convert a binary image to a list of polygons.

    Args:
        binary_image (numpy.ndarray): Binary image.

    Returns:
        list[shapely.geometry.Polygon]: List of polygons.

    """
    contours, hierarchy = cv.findContours(
        binary_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )
    hierarchy = hierarchy[0]  # shape: (n_contours, 4)

    def contour_to_coords(contour):
        return [(int(p[0][0]), int(p[0][1])) for p in contour]

    polygons = []
    used = set()

    for i, (next_idx, prev_idx, child_idx, parent_idx) in enumerate(hierarchy):
        if parent_idx != -1:
            # Skip child contours for now (they'll be added as holes)
            continue

        # Outer contour
        exterior = contour_to_coords(contours[i])

        if len(exterior) < 4:
            # Skip contours that are too small
            used.add(i)
            continue

        holes = []

        # Look for children (holes)
        child = hierarchy[i][2]
        while child != -1:
            hole_coords = contour_to_coords(contours[child])
            holes.append(hole_coords)
            used.add(child)
            # Check for nested children (e.g. box in a hole in a box)
            grandchild = hierarchy[child][2]
            if grandchild != -1:
                # Treat grandchild as a new polygon later
                pass
            child = hierarchy[child][0]  # next sibling hole

        used.add(i)
        polygon = Polygon(exterior, holes)

        if not polygon.is_valid:
            # Attempt to fix the polygon
            polygon = polygon.buffer(0)

        if polygon.is_valid:
            polygons.append(polygon)

    # Add any unused outer contours (e.g. nested objects)
    for i in range(len(contours)):
        if i not in used:
            coords = contour_to_coords(contours[i])

            if len(coords) < 4:
                # Skip contours that are too small
                continue

            poly = Polygon(coords)

            if not poly.is_valid:
                # Attempt to fix the polygon
                poly = poly.buffer(0)

            if poly.is_valid:
                polygons.append(poly)

    return polygons
