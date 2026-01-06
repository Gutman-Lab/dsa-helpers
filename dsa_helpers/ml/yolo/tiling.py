"""Tiling function for use in YOLO object detection from Ultrayltics"""

import large_image, large_image_source_openslide
import geopandas as gpd
import cv2 as cv
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from shapely import force_2d
from shapely.geometry import Polygon
from shapely.affinity import translate, scale

from ...utils import return_mag_and_resolution
from ... import imwrite


def tile_wsi_for_yolo(
    wsi_fp: str,
    geojson_doc: dict,
    roi_labels: str | list[str],
    box_labels: list[str],
    label2id: dict[str, int],
    save_dir: str,
    magnification: int = 20,
    tile_size: int = 1280,
    stride: int | None = 960,
    tile_area_threshold: float = 0.25,
    pad_rgb: tuple[int, int, int] = (114, 114, 114),
    box_area_threshold: float = 0.5,
    allow_small_rotations: bool = False,
) -> pd.DataFrame:
    """Tile WSI for YOLO model training in the Ultralytics format.

    Args:
        wsi_fp (str): File path to the WSI.
        geojson_doc (dict): DSA annotation document in GeoJSON format.
        roi_labels (str | list[str]): Label(s) of the ROI annotations.
            For now only rectangular ROIs with no rotation are supported,
            others are ignored.
        box_labels (list[str]): Labels of the object bounding box
            annotations. The integer labels are determined by the order
            of the labels on the list. Note that again only rectangular
            boxes with no rotation are supported, others are ignored.
        label2id (dict[str, int]): Mapping of label names to integer
            indices. All box labels must be in the label2id dictionary.
        save_dir (str): Directory to save the tiles and labels to.
        magnification (int, optional): Magnification to get tiles at.
            Defaults to 20.
        tile_size (int, optional): Size of the tiles to use. Defaults to
            1280.
        stride (int | None, optional): Stride to use for the tiles. If
            None, the stride is set to the tile size. Defaults to 960.
        tile_area_threshold (float, optional): Fraction of tile that
            must be in the ROI to be included. If the fraction is less
            than this value than it is ignored. Defaults to 0.25.
        pad_rgb (tuple[int, int, int], optional): RGB values to pad the
            tile with if it is smaller than the tile size. Defaults to
            (114, 114, 114).
        box_area_threshold (float, optional): Fraction of box that must
            be in the tile to be included. If the fraction is less than
            this value than it is ignored. Defaults to 0.5.
        allow_small_rotations (bool, optional): Whether to allow small
            rotations of the boxes. Rotations less than 5 degrees are
            ignored, and the smallest bounding box is used instead on
            the tile. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame with the tile metadata.

    """
    assert len(box_labels) == len(
        set(box_labels)
    ), "Box labels must be unique."

    for label in box_labels:
        assert label in label2id, f"Label {label} not in label2id."

    if isinstance(roi_labels, str):
        roi_labels = [roi_labels]

    # Make the labels strings.
    roi_labels = [str(label) for label in roi_labels]
    box_labels = [str(label) for label in box_labels]

    save_dir_path = Path(save_dir)
    img_dir = save_dir_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir = save_dir_path / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.GeoDataFrame.from_features(geojson_doc["features"])

    # Convert the label column, which is a dict to use the value as cell value.
    gdf["label"] = gdf["label"].apply(lambda x: x["value"])

    # Keep only rectangles.
    gdf = gdf[(gdf["type"] == "rectangle")]

    if allow_small_rotations:
        # Remove rows with rotations less than 5 degrees.
        gdf = gdf[gdf["rotation"] <= 5]

        # Take only the smallest bounding box for each box.
        gdf["geometry"] = gdf["geometry"].apply(
            lambda geom: Polygon(
                [
                    (geom.bounds[0], geom.bounds[1]),  # (minx, miny)
                    (geom.bounds[2], geom.bounds[1]),  # (maxx, miny)
                    (geom.bounds[2], geom.bounds[3]),  # (maxx, maxy)
                    (geom.bounds[0], geom.bounds[3]),  # (minx, maxy)
                    (geom.bounds[0], geom.bounds[1]),  # close the polygon
                ]
            )
        )
    else:
        gdf = gdf[(gdf["rotation"] == 0)]

    # Make the geometries 2D.
    gdf["geometry"] = gdf["geometry"].apply(force_2d)

    roi_gdf = gdf[gdf["label"].isin(roi_labels)].reset_index(drop=True)
    box_gdf = gdf[gdf["label"].isin(box_labels)].reset_index(drop=True)

    # Get the large image tile source.
    ts = large_image_source_openslide.open(wsi_fp)
    wsi_name = Path(wsi_fp).stem

    if stride is None:
        stride = tile_size

    magnification, mm_px = return_mag_and_resolution(mag=magnification)

    # Calculate the scale factor on each axis.
    # Calculating a multirplicative factor for going from scan mag to desired mag.
    # With mm_px, higher values are more zoomed in, so we reverse the division.
    ts_metadata = ts.getMetadata()
    sf_x = ts_metadata["mm_x"] / mm_px
    sf_y = ts_metadata["mm_y"] / mm_px

    metadata = []

    n_rois = len(roi_gdf)

    # Loop through each ROI.
    for i, roi_row in roi_gdf.iterrows():
        print(f"Processing ROI {i+1} of {n_rois}...")
        # Get the bounds of the ROI.
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_row["geometry"].bounds
        roi_x1, roi_y1 = int(roi_x1), int(roi_y1)
        roi_x2, roi_y2 = int(roi_x2), int(roi_y2)

        roi_key = f"x{roi_x1}y{roi_y1}x{roi_x2}y{roi_y2}"

        # Make a copy of the label boxes.
        box_gdf_copy = box_gdf.copy()

        # Shift the boxes so zero zero is the top left of the ROI.
        box_gdf_copy["geometry"] = box_gdf_copy["geometry"].apply(
            lambda x: translate(x, xoff=-roi_x1, yoff=-roi_y1)
        )

        # Scale the boxes so they are at the desired magnification.
        box_gdf_copy["geometry"] = box_gdf_copy["geometry"].apply(
            lambda x: scale(x, xfact=sf_x, yfact=sf_y, origin=(0, 0))
        )

        # Get the ROI at the specified resolution.
        roi_img = ts.getRegion(
            region={
                "left": roi_x1,
                "top": roi_y1,
                "right": roi_x2,
                "bottom": roi_y2,
            },
            format=large_image.constants.TILE_FORMAT_NUMPY,
            scale={"mm_x": mm_px, "mm_y": mm_px},
        )[0][:, :, :3].copy()

        # Calculate the top left coordinates of each tile in this ROI.
        xys = []
        roi_h, roi_w = roi_img.shape[:2]

        for x in range(0, roi_w, stride):
            for y in range(0, roi_h, stride):
                xys.append((x, y))

        # Loop through each tile.
        for xy in tqdm(xys, desc="Processing tiles"):
            x, y = xy

            # Get the tile image.
            tile_img = roi_img[y : y + tile_size, x : x + tile_size]

            # Shift all the boxes by the top left corner of this tile.
            box_gdf_shifted = box_gdf_copy.copy()
            box_gdf_shifted["geometry"] = box_gdf_shifted["geometry"].apply(
                lambda geom: translate(geom, xoff=-x, yoff=-y)
            )

            # Add the area of the box to tile.
            box_gdf_shifted["area"] = box_gdf_shifted["geometry"].area

            # Create a polygon for the tile
            tile_h, tile_w = tile_img.shape[:2]

            # Determine if the tile is too small.
            tile_area = tile_h * tile_w
            if tile_area < tile_area_threshold:
                # Skip the tile.
                continue

            tile_box = Polygon(
                [(0, 0), (tile_w, 0), (tile_w, tile_h), (0, tile_h), (0, 0)]
            )

            # Calculate the intersection.
            box_gdf_shifted["geometry"] = box_gdf_shifted[
                "geometry"
            ].intersection(tile_box)

            # Remove the empty geometries.
            box_gdf_shifted = box_gdf_shifted[
                box_gdf_shifted["geometry"].area > 0
            ].reset_index(drop=True)

            # Calculate the new area.
            box_gdf_shifted["clipped_area"] = box_gdf_shifted["geometry"].area

            # Remove boxes that are less than the threshold
            box_gdf_shifted["frac_in_tile"] = (
                box_gdf_shifted["clipped_area"] / box_gdf_shifted["area"]
            )
            box_gdf_shifted = box_gdf_shifted[
                box_gdf_shifted["frac_in_tile"] >= box_area_threshold
            ].reset_index(drop=True)

            if tile_h != tile_size or tile_w != tile_size:
                # Pad the edge of the tile.
                tile_img = cv.copyMakeBorder(
                    tile_img,
                    0,
                    tile_size - tile_h,
                    0,
                    tile_size - tile_w,
                    cv.BORDER_CONSTANT,
                    value=pad_rgb,
                )

            tile_key = (
                f"{wsi_name}_{roi_key}_x{x}y{y}x{x+tile_size}y{y+tile_size}"
            )
            img_fp = img_dir / f"{tile_key}.png"
            imwrite(img_fp, tile_img)

            # Write the labels to file.
            labels = ""

            for _, r in box_gdf_shifted.iterrows():
                # Get the width and height of the object.
                bx1, by1, bx2, by2 = r.geometry.bounds

                xc = (bx1 + bx2) / 2
                yc = (by1 + by2) / 2
                bh = by2 - by1
                bw = bx2 - bx1

                # normalize
                xc /= tile_size
                yc /= tile_size
                bw /= tile_size
                bh /= tile_size

                labels += f"{box_labels.index(r['label'])} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}\n"

            label_fp = label_dir / f"{tile_key}.txt"

            if labels:
                with open(label_fp, "w") as f:
                    f.write(labels.strip())

            metadata.append(
                [
                    str(img_fp),
                    str(label_fp),
                    roi_key,
                    x,
                    y,
                    x + tile_size,
                    y + tile_size,
                    magnification,
                    mm_px,
                ]
            )

    return pd.DataFrame(
        metadata,
        columns=[
            "img_fp",
            "label_fp",
            "roi_key",
            "x1",
            "y1",
            "x2",
            "y2",
            "mag",
            "mm_px",
        ],
    )
