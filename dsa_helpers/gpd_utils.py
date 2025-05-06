"""
Utility functions for working with GeoPandas.

Functions:
- count_polygon_points: Count the total number of points in a Shapely polygon.
- total_points_in_gdf: Calculate the total number of points in all polygons in a GeoDataFrame.
- plot_gdf: Create a figure from a geopandas dataframe.
- remove_gdf_overlaps: Remove overlaps from a GeoDataFrame.
- draw_gdf_on_array: Draw a GeoDataFrame on an array.

"""

import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.features import rasterize
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def count_polygon_points(polygon: Polygon) -> int:
    """
    Count the total number of points in a Shapely polygon, including
    exterior and interior rings.

    Args:
        polygon (shapely.geometry.Polygon): A Shapely polygon object.

    Returns:
        int: The total number of points in the polygon.

    """
    # Count exterior points.
    exterior_points = len(polygon.exterior.coords)

    # Count interior points (holes).
    interior_points = sum(
        len(interior.coords) for interior in polygon.interiors
    )

    # Total points.
    return exterior_points + interior_points


def total_points_in_gdf(gdf: gpd.GeoDataFrame) -> int:
    """
    Calculate the total number of points in all polygons in a GeoDataFrame.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing polygons.

    Returns:
        int: The total number of points in all polygons.

    """
    return int(gdf.geometry.apply(count_polygon_points).sum())


def plot_gdf(
    gdf: gpd.GeoDataFrame,
    figsize: int | tuple[int, int] | None = None,
    label_column: str = "group",
    label2color: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create figure from a geopandas dataframe.

    Args:
        gdf (geopandas.GeoDataFrame): A GeoDataFrame containing
            polygons. Must have a 'label' column.
        figsize (int | tuple[int, int] | None): The size of the figure.
            If None, the figure size will be (5, 5).
        label_column (str): The column name of the label.
        label2color (dict | None): A dictionary mapping labels to
            colors. If None then a legend will not be included and the
            colors will be random. Colors should be an accepted
            matplotlib color.

    Returns:
        tuple[plt.Figure, plt.Axes]: The figure and axes objects.

    """
    if figsize is None:
        figsize = (5, 5)
    elif isinstance(figsize, int):
        figsize = (figsize, figsize)

    gdf = gdf.copy()

    # Blow up any multi-polygons.
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    if label2color is not None:
        gdf["color"] = gdf[label_column].map(label2color)

        # Remove rows where color is nan.
        gdf = gdf.dropna(subset=["color"])

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    gdf.plot(color=gdf["color"], ax=ax)

    if label2color is not None:
        # Add legend.
        handles = [
            mpatches.Patch(color=color, label=key)
            for key, color in label2color.items()
        ]

        ax.legend(
            handles=handles,
            title="Legend",
            loc="upper left",
            bbox_to_anchor=(
                1.02,
                1,
            ),  # x > 1 pushes it to the right, y = 1 keeps it top-aligned
            borderaxespad=0,
        )

    return fig, ax


def remove_gdf_overlaps(
    gdf: gpd.GeoDataFrame,
    columns: str | list[str],
    ascending: bool | list[bool] = True,
    is_notebook: bool = False,
) -> gpd.GeoDataFrame:
    """Remove overlaps from a GeoDataFrame.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame to remove
            overlaps from.
        columns (str | list[str]): The columns to sort dataframe by. The
            order determines the order of removal overlapping polygon
            regions.
        ascending (bool): Whether to remove overlaps in ascending order.
        is_notebook (bool): Switch which tqdm bar to use. Default is
            False, which uses the terminal progress bar.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with overlaps removed.

    """
    if isinstance(columns, str):
        columns = [columns]

    if isinstance(ascending, bool):
        ascending = [ascending] * len(columns)

    # Remove multi-polygons.
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Sort the GeoDataFrame by the columns and ascending order.
    gdf = gdf.sort_values(by=columns, ascending=ascending).reset_index(
        drop=True
    )

    if is_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Loop through each row in the GeoDataFrame.
    for i, row in tqdm(gdf.iterrows(), total=len(gdf)):
        # Check overlap with all following rows.
        other_gdf = gdf[i + 1 :]

        if len(other_gdf):
            # Get the rows that overlap.
            overlap = row.geometry.overlaps(other_gdf.geometry)

            # Get the rows that overlap.
            other_rows = other_gdf[overlap]

            other_gdf.loc[:, "geometry"] = other_gdf.geometry.difference(
                row.geometry
            )

    # This could have created multiple polygons, explode them.
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    return gdf


def draw_gdf_on_array(gdf, shape, id_column="idx", default_value: int = 0):
    """Use the polygons in a geopandas dataframe to draw on a passed
    numpy array.

    Args:
        gdf (geopandas.GeoDataFrame): The GeoDataFrame to draw on the
            array.
        shape (tuple): Height and width of the array to draw on.
        id_column (str): The column on the dataframe to use for the
            value of each polygon drawn. The values in this column
            must be integers.
        default_value (int): The default value of the array, this will
            the values returned for pixels that are not drawn on.

    Returns:
        numpy.ndarray: The array with the polygons drawn on it.

    """
    # Create the array.
    array = np.ones(shape, dtype=np.uint8) * default_value

    # Draw each polygon with its corresponding label
    for _, row in gdf.iterrows():
        # Create a list of (geometry, value) pairs
        shapes = [(row.geometry, row[id_column])]

        # Rasterize the shapes
        burned = rasterize(
            shapes=shapes,
            out_shape=shape,
            fill=default_value,
            default_value=default_value,
        )

        # Update the array where the polygon was burned
        array = np.where(burned > 0, burned, array)

    return array
