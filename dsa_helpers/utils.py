"""Utility functions for DSA Helpers.

This module provides various miscellaneous utility functions that are
not grouped into their own modules.
"""

from shapely.geometry import Polygon
import numpy as np


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
