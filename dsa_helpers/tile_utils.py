# -*- coding: utf-8 -*-
"""Tile functions.

This module contains functions used when working tiles.

"""
from geopandas import GeoDataFrame
from .image_utils import label_mask_to_polygons


def merge_tile_masks(
    tile_list: list, buffer: int = 1, background_label: int | None = 0
) -> GeoDataFrame:
    """
    Merge the tile masks into a single mask for a large image.

    Args:
        tile_list (list): List of three-lenght tuples containing:
            (1) fp or array for mask, (2) x-coordinate for tile, (3)
            y-coordinate for tile.
        buffer (int): Buffer to apply to the polygons before dissolving.
        background_label (int | None): Label value of the background class,
            which is ignored. Default is 0. If None then all labels are
            considered.

    Returns:
        GeoDataFrame: GeoDataFrame with the merged mask.

    """
    polygons_and_labels = []

    # Process each tile.
    for tile_info in tile_list:
        tile, x, y = tile_info

        # Process the mask by converting it to polygons.
        polygons_and_labels.extend(
            label_mask_to_polygons(
                tile, x_offset=x, y_offset=y, exclude_labels=background_label
            )
        )

    # Convert polygons and labels into a GeoDataFrame.
    gdf = GeoDataFrame(polygons_and_labels, columns=["geometry", "label"])

    # Apply a buffer to make edges touch.
    gdf["geometry"] = gdf["geometry"].buffer(buffer)

    # Dissolve the dataframe by the label.
    gdf = gdf.dissolve(by="label", as_index=False)

    # Remove the buffer.
    gdf["geometry"] = gdf["geometry"].buffer(-buffer)

    # Check that all polygons are valid.
    for i, row in gdf.iterrows():
        if not row["geometry"].is_valid:
            gdf.loc[i, "geometry"] = row["geometry"].buffer(0)

    return gdf
