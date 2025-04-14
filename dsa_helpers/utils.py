"""Utility functions for DSA Helpers.

This module provides various miscellaneous utility functions that are
not grouped into their own modules.
"""

from shapely.geometry import Polygon
import numpy as np
import cv2 as cv


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
