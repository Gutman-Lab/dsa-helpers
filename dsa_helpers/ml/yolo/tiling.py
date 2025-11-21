"""Tiling function for use in YOLO object detection from Ultrayltics"""

import large_image_source_openslide
import large_image
import geopandas as gpd
import cv2 as cv
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Polygon

from ...girder_utils import get_mag_and_mm_px
from ... import imwrite


def tile_wsi_with_dsa_annotations(
    wsi_fp: str,
    save_dir: str,
    geojson_ann_doc: dict,
    roi_classes: str | list[str],
    object_label2id: dict[str, int],
    mag: float | None = None,
    mm_px: float | None = None,
    tile_size: int = 1280,
    stride: int | None = 960,
    pad_rgb: tuple[int, int, int] = (114, 114, 114),
    obj_area_thr: float = 0.5,
) -> tuple[pd.DataFrame, tuple[float, float], tuple[float, float]]:
    """Tile a WSI with DSA annotations used to generate the label text
    files needed for YOLO object detection from Ultrayltics.

    Args:
        wsi_fp (str): Filepath to the WSI.
        save_dir (str): Directory to save the tiles images and labels.
        geojson_ann_doc (dict): DSA annotation document in geojson
            format.
        roi_classes (str | list[str]): Classes of ROIs to tile.
        object_label2id (dict[str, int]): Dictionary mapping object
            labels to IDs.
        mag (float | None): Magnification to tile at. Will be converted
            to mm_px appropriate for Emory scanners.
        mm_px (float | None): Micrometers per pixel to use. If neither
            this or mag is specified, the scan resolution will be used.
        tile_size (int | None): Size of the tiles.
        stride (int | None): Stride to tile. If None it will be set to
            the tile size (no overlap between tiles).

    Returns:
        tuple[pd.DataFrame, tuple[float, float], tuple[float, float]]: A tuple
            containing a DataFrame with the tile metadata. All coordinates
            are relative to the scan resolution.
            - image_fp (str): Filepath to the tile image.
            - label_fp (str): Filepath to the tile label.
            - roi_x1 (int): X coordinate of the top left corner of the ROI.
            - roi_y1 (int): Y coordinate of the top left corner of the ROI.
            - roi_x2 (int): X coordinate of the bottom right corner of the ROI.
            - roi_y2 (int): Y coordinate of the bottom right corner of the ROI.
            - tile_x1 (int): X coordinate of the top left corner of the tile.
            - tile_y1 (int): Y coordinate of the top left corner of the tile.
            - tile_x2 (int): X coordinate of the bottom right corner of the tile.
            - tile_y2 (int): Y coordinate of the bottom right corner of the tile.
            and the magnification and micrometers per pixel used in each
            axis (x, y).

    """
    if isinstance(roi_classes, str):
        roi_classes = [roi_classes]

    if stride is None:
        stride = tile_size

    assert stride > 0, "Stride must be greater than 0."

    gdf = gpd.GeoDataFrame.from_features(geojson_ann_doc["features"])

    # Remove any geometries that are not rectangles or have a rotation.
    gdf = gdf[(gdf["type"] == "rectangle") & (gdf["rotation"] == 0)]

    # Convert the label column to use the value of its dictionary.
    gdf["label"] = gdf["label"].apply(lambda x: x["value"])

    # Filter to ROI gdf and object ones.
    roi_gdf = gdf[gdf["label"].isin(roi_classes)].reset_index(drop=True)
    object_gdf = gdf[gdf["label"].isin(object_label2id.keys())].reset_index(
        drop=True
    )
    object_gdf["area"] = object_gdf.geometry.area

    ts = large_image_source_openslide.open(wsi_fp)
    ts_metadata = ts.getMetadata()

    mag, mm_px, sf = get_mag_and_mm_px(ts_metadata, mag, mm_px)

    # Calculate the tile size and stride at scan magnification.
    scan_tile_size = (int(tile_size / sf[0]), int(tile_size / sf[1]))
    scan_stride = (int(stride / sf[0]), int(stride / sf[1]))

    wsi_name = Path(wsi_fp).stem
    save_dir = Path(save_dir)
    img_dir = save_dir / "images"
    label_dir = save_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(exist_ok=True)

    # Loop through each ROI.
    tile_metadata = []

    for i, roi in tqdm(
        roi_gdf.iterrows(), total=len(roi_gdf), desc="Processing ROIs"
    ):
        roi_geom = roi.geometry

        # Grab the bounds.
        minx, miny, maxx, maxy = roi_geom.bounds
        minx, miny = int(minx), int(miny)
        maxx, maxy = int(maxx), int(maxy)

        # Calculate the tile coordinates for this ROI.
        xys = []

        for x in range(minx, maxx, scan_stride[0]):
            for y in range(miny, maxy, scan_stride[1]):
                xys.append((x, y))

        # Loop through each tile coordinate.
        for xy in tqdm(xys, desc=f"Processing tiles for ROI {i+1}"):
            x1, y1 = xy

            # Calculate the corners of the tile.
            x2 = min(x1 + scan_tile_size[0], maxx)
            y2 = min(y1 + scan_tile_size[1], maxy)

            # Get the tile from the WSI.
            img = ts.getRegion(
                region={
                    "left": x1,
                    "top": y1,
                    "right": x2,
                    "bottom": y2,
                },
                format=large_image.constants.TILE_FORMAT_NUMPY,
                scale={"mm_x": mm_px[0], "mm_y": mm_px[1]},
            )[0][:, :, :3].copy()

            # Create a box on the tile.
            tile_geom = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

            # Get polygons for this tile.
            tile_objects = object_gdf[object_gdf.intersects(tile_geom)].copy()
            tile_objects = tile_objects.clip(tile_geom)

            # Remove objects that most area is missing from the tile.
            tile_objects = tile_objects[
                tile_objects.geometry.area / tile_objects["area"]
                > obj_area_thr
            ]

            # Translate and scale the polygons.
            tile_objects.geometry = tile_objects.geometry.translate(
                xoff=-x1, yoff=-y1
            ).scale(xfact=sf[0], yfact=sf[1], origin=(0, 0))

            # Pad the tile if not the right tile size.
            tile_shape = img.shape[:2]

            if tile_shape != (tile_size, tile_size):
                # Pad the tile.
                img = cv.copyMakeBorder(
                    img,
                    0,
                    tile_size - tile_shape[0],
                    0,
                    tile_size - tile_shape[1],
                    cv.BORDER_CONSTANT,
                    value=pad_rgb,
                )

            # Save the tile.
            key = f"{wsi_name}_roi-x{minx}y{miny}_tile-x{x1}y{y1}."
            img_fp = img_dir / f"{key}png"
            imwrite(img_fp, img)

            # Save the labels in the Ultralytics format.
            label_fp = label_dir / f"{key}txt"

            labels = ""

            for _, r in tile_objects.iterrows():
                # Get the width and height of the object.
                width, height = r.width, r.height

                # Get the center of the object.
                cx, cy = r.geometry.centroid.x, r.geometry.centroid.y

                # Normalize the coordinates by tile size.
                cx /= tile_size
                cy /= tile_size
                width /= tile_size
                height /= tile_size

                # Setup the YOLO line.
                label = r["label"]
                label_id = int(object_label2id[label])
                labels += (
                    f"{label_id} {cx:.4f} {cy:.4f} {width:.4f} {height:.4f}\n"
                )

            if len(labels):
                with open(label_fp, "w") as f:
                    f.write(labels.strip())

            # Append the tile metadata.
            tile_metadata.append(
                {
                    "image_fp": str(img_fp),
                    "label_fp": str(label_fp),
                    "roi_x1": minx,
                    "roi_y1": miny,
                    "roi_x2": maxx,
                    "roi_y2": maxy,
                    "tile_x1": x1,
                    "tile_y1": y1,
                    "tile_x2": x2,
                    "tile_y2": y2,
                }
            )

    df = pd.DataFrame(
        tile_metadata,
        columns=[
            "image_fp",
            "label_fp",
            "roi_x1",
            "roi_y1",
            "roi_x2",
            "roi_y2",
            "tile_x1",
            "tile_y1",
            "tile_x2",
            "tile_y2",
        ],
    )
    return df, mag, mm_px
