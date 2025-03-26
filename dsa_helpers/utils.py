"""Utility functions for DSA Helpers.

This module provides various miscellaneous utility functions that are
not grouped into their own modules.
"""

from shapely.geometry import Polygon


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
