"""Functions for tiling images.

Functions:
- tile_wsi_with_masks_from_dsa_annotations: Tile a WSI with semantic
    segmentation label masks created from DSA annotations.
- tile_image: Tile an image into smaller images.
- tile_image_with_mask: Tile an image and its corresponding mask.
- tile_wsi_for_segformer_semantic_segmentation: Tile a WSI with semantic
    segmentation label masks created from DSA annotations. Serially.

"""

import large_image
import cv2 as cv
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import pandas as pd
from shapely.affinity import scale, translate
import geopandas as gpd
import histomicstk as htk

from . import imwrite, imread
from .gpd_utils import draw_gdf_on_array

stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map

# specify stains of input image
stains = [
    "hematoxylin",  # nuclei stain
    "eosin",  # cytoplasm stain
    "null",
]  # set to null if input contains only two stains

# create stain matrix
W = np.array([stain_color_map[st] for st in stains]).T


def _proccess_tile_with_masks_from_dsa_annotations(
    gdf,
    ts,
    prepend_name,
    scan_x,
    scan_y,
    scan_tile_size,
    tile_size,
    mag,
    scan_to_mag_sf,
    img_dir,
    mask_dir,
    edge_value,
    background_id,
    edge_thr,
    ignore_existing,
    ignore_id,
    ignore_value,
    label_col,
    ignore_labels,
):
    """Processing tiles with masks, used in multiprocessing only."""
    # Calcualte the x and y at magnification desired.
    x_mag = int(scan_x * scan_to_mag_sf)
    y_mag = int(scan_y * scan_to_mag_sf)

    # Format the filepath to save the tile.
    fn = f"{prepend_name}x{x_mag}y{y_mag}.png"
    img_fp = img_dir / fn
    mask_fp = mask_dir / fn

    # Ignore existing tiles.
    if ignore_existing and img_fp.is_file() and mask_fp.is_file():
        return str(img_fp), str(mask_fp), x_mag, y_mag

    # Get the tile.
    img = ts.getRegion(
        region={
            "left": scan_x,
            "top": scan_y,
            "right": scan_x + scan_tile_size,
            "bottom": scan_y + scan_tile_size,
        },
        format=large_image.constants.TILE_FORMAT_NUMPY,
        scale={"magnification": mag},
    )[0][:, :, :3].copy()

    # Check if the tile area is below threshold.
    h, w = img.shape[:2]

    if h * w / (tile_size * tile_size) < edge_thr:
        return None

    if (h, w) != (tile_size, tile_size):
        # Pad the image with zeroes to make it the desired size.
        img = cv.copyMakeBorder(
            img,
            0,
            tile_size - h,
            0,
            tile_size - w,
            cv.BORDER_CONSTANT,
            value=edge_value,
        )

        h, w = img.shape[:2]

    # Create a copy of the geodataframe for this tile.
    tile_gdf = gdf.copy()

    # Translate the coordinates so 0, 0 is the top left corner of the tile.
    tile_gdf["geometry"] = tile_gdf["geometry"].apply(
        lambda geom: translate(geom, xoff=-x_mag, yoff=-y_mag)
    )

    # Draw the dataframe on the tile mask.
    mask = draw_gdf_on_array(
        tile_gdf, (h, w), default_value=background_id
    ).copy()

    if ignore_labels is not None and len(ignore_labels):
        ignore_gdf = tile_gdf[tile_gdf[label_col].isin(ignore_labels)].copy()
        ignore_gdf["idx"] = [1] * len(ignore_gdf)

        # Create a mask of the ignore labels.
        ignore_mask = draw_gdf_on_array(
            ignore_gdf,
            (h, w),
        ).copy()

        # Apply the ignore ids to the image and mask.
        img[np.where(ignore_mask == 1)] = ignore_value
        mask[np.where(ignore_mask == 1)] = ignore_id

    # Save the tile image and mask.
    imwrite(img_fp, img)
    imwrite(mask_fp, mask.astype(np.uint8), grayscale=True)

    return str(img_fp), str(mask_fp), x_mag, y_mag


def tile_wsi_with_masks_from_dsa_annotations(
    wsi_fp: str,
    geojson_ann_doc: dict,
    label2id: dict,
    save_dir: str,
    tile_size: int,
    label_col: str = "group",
    stride: int | None = None,
    mag: float | None = None,
    prepend_name: str = "",
    nproc: int = 1,
    edge_value: tuple[int, int, int] | int = (255, 255, 255),
    background_id: int = 0,
    edge_thr: float = 0.25,
    ignore_existing: bool = False,
    ignore_labels: str | list[str] | None = None,
    ignore_id: int = 255,
    ignore_value: tuple[int, int, int] = (255, 255, 255),
    notebook_tqdm: bool = False,
) -> list[str]:
    """Tile a WSI with semantic segmentation label masks created from
    DSA annotations. DSA annotation class labels for elements are
    specified in the "label" dictionary in the "value" key of each
    element.

    Args:
        wsi_fp (str): file path to WSI.
        geojson_ann_doc dict: DSA annotation document in geojson format.
        label2id (dict): mapping of label names to integer indices.
        save_dir (str): directory to save the tiled images.
        tile_size (int): size of the tile images at desired
            magnificaiton.
        label_col (str, optional): Column name for polygon labels.
            Options are "group" or "label". Defaults to "group".
        stride (int | None, optional): stride of the tiling. If None,
            the stride will be the same as the tile size. Defaults to
            None.
        mag (float | None, optional): magnification level of the WSI.
            If None, the function will use the default magnification
            level. Defaults to None.
        prepend_name (str, optional): prepend name to the created tile
            images and masks. Defaults to "".
        nproc (int, optional): number of processes to use for tiling.
            Defaults to 1.
        background_value (int, optional): Value to use for the
            background in the images, used when padding tiles at the
            edge of WSI. Defaults to (255, 255, 255). If set to a single
            value it will be used for all channels.
        background_index (int, optional): Index of the background class.
            Defaults to 0.
        edge_thr (float, optional): For tiles at the edge, if most of
            the tile is padded background then it is ignored. If the
            amount of tile in WSI / amount of tile padded is less than
            this threshold, the tile is ignored. Defaults to 0.25.
        ignore_existing (bool, optional): Whether to ignore existing
            tiles. If False, tiles will not be created if they already
            exist. Defaults to False.
        ignore_labels (str | list[str] | None, optional): Labels to
            ignore. Defaults to None.
        ignore_index (int, optional): Index of the ignore class. Defaults
            to 255.
        ignore_value (tuple[int, int, int], optional): Value to use for
            the ignore class in the images. Defaults to (255, 255, 255).
            If set to a single value it will be used for all channels.
        notebook_tqdm (bool, optional): Whether to use tqdm in a
            notebook. Defaults to False.

    Returns:
        list[str]: A list of tuples: (tile file path, x, y coordinates
            of tile at magnification desired).

    """
    if notebook_tqdm:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Should be a list of string labels.
    if isinstance(ignore_labels, str):
        ignore_labels = [ignore_labels]

    # Read the tile source.
    ts = large_image.getTileSource(wsi_fp)

    # Get the magnification at scan.
    ts_metadata = ts.getMetadata()
    scan_mag = ts_metadata["magnification"]

    # Set the stride.
    if stride is None:
        stride = tile_size

    # Get tile size & stride at scan magnification.
    # Also, determine scaling factor to go from scan mag to desired mag.
    if mag is None:
        scan_tile_size = tile_size
        scan_stride = stride
        mag = scan_mag
        scan_to_mag_sf = 1
    else:
        scan_to_mag_sf = mag / scan_mag  # scan mag -> desired mag
        scan_tile_size = int(tile_size / scan_to_mag_sf)
        scan_stride = int(stride / scan_to_mag_sf)

    gdf = gpd.GeoDataFrame.from_features(geojson_ann_doc.get("features", []))

    # Convert the label column, which is a dictionary, set it to its 'value' key.
    gdf["label"] = gdf["label"].apply(lambda x: x.get("value", ""))

    # Filter to polygons in label2id.
    gdf = gdf[gdf[label_col].isin(label2id.keys())]

    # Remove rows with geometry is not a Polygon.
    gdf = gdf[gdf["geometry"].type == "Polygon"]

    # Scale the coordinates to mag.
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: scale(
            geom, xfact=scan_to_mag_sf, yfact=scan_to_mag_sf, origin=(0, 0)
        )
    )

    # Add an idx column, which is the map of the group to the label2id.
    gdf["idx"] = gdf[label_col].map(label2id)

    # Get the top left corner of every tile in the WSI, at scan mag.
    scan_xys = [
        (scan_x, scan_y)
        for scan_x in range(0, ts_metadata["sizeX"], scan_stride)
        for scan_y in range(0, ts_metadata["sizeY"], scan_stride)
    ]

    # Create directory to save tile images and masks.
    save_dir = Path(save_dir)
    img_dir = save_dir / "images"
    mask_dir = save_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    with Pool(nproc) as pool:
        jobs = [
            pool.apply_async(
                _proccess_tile_with_masks_from_dsa_annotations,
                (
                    gdf,
                    ts,
                    prepend_name,
                    xy[0],
                    xy[1],
                    scan_tile_size,
                    tile_size,
                    mag,
                    scan_to_mag_sf,
                    img_dir,
                    mask_dir,
                    edge_value,
                    background_id,
                    edge_thr,
                    ignore_existing,
                    ignore_id,
                    ignore_value,
                    label_col,
                    ignore_labels,
                ),
            )
            for xy in scan_xys
        ]

        tile_info = []

        for job in tqdm(jobs, desc="Tiling..."):
            out = job.get()

            if out is not None:
                tile_info.append(job.get())

    return tile_info


def tile_image(
    img: np.ndarray,
    save_loc: str,
    tile_size: int,
    stride: int | None = None,
    fill: int | tuple = (255, 255, 255),
    prepend_name: str = "",
    overwrite: bool = False,
    grayscale: bool = False,
) -> pd.DataFrame:
    """Tile an image into smaller images.

    Args:
        img (np.ndarray): The image to tile.
        save_loc (str): The location to save the tiles.
        tile_size (int): The size of the tiles.
        stride (int | None, optional): The stride for the tiles. Defaults to None,
            in which case it is set  equal to tile_size.
        fill (int | tuple, optional): The fill value for tiles over the edges.
            Defaults to (255, 255, 255).
        prepend_name (str, optional): A string to prepend to the tile names.
        overwrite (bool, optional): Whether to overwrite existing images.
            Defaults to False.
        grayscale (bool, optional): Whether to save images as grayscale instead
            of the default RGB. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame with the tile locations.

    """
    h, w = img.shape[:2]

    img = cv.copyMakeBorder(
        img, 0, tile_size, 0, tile_size, cv.BORDER_CONSTANT, value=fill
    )

    # Pad the image to avoid tiles not of the right size.
    save_loc = Path(save_loc)
    save_loc.mkdir(parents=True, exist_ok=True)

    if stride is None:
        stride = tile_size

    df_data = []

    xys = [(x, y) for x in range(0, w, stride) for y in range(0, h, stride)]

    if prepend_name:
        prepend_name += "_"

    for xy in xys:
        # Get the top left and bottom right coordinates of the tile.
        x, y = xy

        # Filepath for saving tile.
        save_fp = save_loc / f"{prepend_name}x{x}y{y}tilesize{tile_size}.png"

        if not save_fp.is_file() or overwrite:
            # Get the tile from the image.
            tile_img = img[y : y + tile_size, x : x + tile_size]

            imwrite(save_fp, tile_img, grayscale=grayscale)

        df_data.append([save_fp, x, y, tile_size])

    return pd.DataFrame(df_data, columns=["fp", "x", "y", "tile_size"])


def tile_image_with_mask(
    img: np.ndarray | str,
    mask: np.ndarray | str,
    save_dir: str,
    tile_size: int,
    stride: int | None = None,
    fill: tuple[int, int, int] = (255, 255, 255),
    mask_pad_value: int = 0,
    prepend_name: str = "",
) -> pd.DataFrame:
    """Tile an image and its corresponding mask.

    Args:
        img (np.ndarray): The image or filepath to tile.
        mask (np.ndarray): The mask or filepath to mask.
        save_dir (str): The directory to save the tiles. Images are
            saved in a "images" subdirectory and masks in a "masks"
            subdirectory.
        tile_size (int): The size of the tiles.
        stride (int, optional): The stride of the tiles. Defaults to
            None which sets the stride to the tile size
            (no overlap between tiles).
        fill (tuple[int, int, int], optional): The fill value for the
            tiles that go over the edge of image. Defaults to (255,
            255, 255).
        mask_pad_value (int, optional): The value to pad the mask with.
            Defaults to 0.
        prepend_name (str, optional): A string to prepend to the tile
            filenames. Default to "". Naming of tiles is:
            {prepend_name}x{x}y{y}.png.

    Returns:
        pandas.DataFrame: A pandas dataframe containing the filepath of
            the image and mask and x, y coordinate.

    """
    if isinstance(img, str):
        img = imread(img)

    if isinstance(mask, str):
        mask = imread(mask, grayscale=True)

    h, w = img.shape[:2]

    # Pad the mask to avoid tiles going over the image boundary.
    mask = cv.copyMakeBorder(
        mask.copy(),
        0,
        h + tile_size,
        0,
        w + tile_size,
        cv.BORDER_CONSTANT,
        value=mask_pad_value,
    )

    # Same for the image.
    img = cv.copyMakeBorder(
        img.copy(),
        0,
        h + tile_size,
        0,
        w + tile_size,
        cv.BORDER_CONSTANT,
        value=fill,
    )

    # Make subdirectories to save images to.
    img_dir = Path(save_dir).joinpath("images")
    mask_dir = Path(save_dir).joinpath("masks")
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    if stride is None:
        stride = tile_size

    xys = [(x, y) for x in range(0, w, stride) for y in range(0, h, stride)]

    tile_data = []

    for xy in xys:
        x, y = xy

        tile_img = img[y : y + tile_size, x : x + tile_size]
        tile_img_fp = img_dir / f"{prepend_name}x{x}y{y}.png"

        tile_mask = mask[y : y + tile_size, x : x + tile_size]
        tile_mask_fp = mask_dir / f"{prepend_name}x{x}y{y}.png"

        imwrite(tile_img_fp, tile_img)
        imwrite(tile_mask_fp, tile_mask, grayscale=True)

        tile_data.append([str(tile_img_fp), str(tile_mask_fp), x, y])

    return pd.DataFrame(tile_data, columns=["fp", "mask_fp", "x", "y"])


def tile_wsi_for_segformer_semantic_segmentation(
    wsi_fp: str,
    geojson_ann_doc: dict,
    group2id: dict,
    save_dir: str,
    tile_size: int = 512,
    label_col: str = "label",
    stride: int | None = None,
    mag: float | None = 10.0,
    prepend_name: str = "",
    edge_value: tuple[int, int, int] | int = (255, 255, 255),
    background_id: int = 0,
    edge_thr: float = 0.25,
    ignore_labels: str | list[str] | None = "Exclude",
    ignore_id: int = 0,
    ignore_value: tuple[int, int, int] = (255, 255, 255),
    image_type: str = "rgb",
    notebook_tqdm: bool = False,
) -> pd.DataFrame:
    """Tile a WSI with semantic segmentation label masks created from
    DSA annotations. DSA annotation class labels for elements are
    specified in the "label" dictionary in the "value" key of each
    element.

    Args:
        wsi_fp (str): file path to WSI.
        geojson_ann_doc dict: DSA annotation document in geojson format.
        label2id (dict): mapping of label names to integer indices.
        save_dir (str): directory to save the tiled images.
        tile_size (int): size of the tile images at desired
            magnificaiton.
        label_col (str, optional): Column name for polygon labels.
            Options are "group" or "label". Defaults to "group".
        stride (int | None, optional): stride of the tiling. If None,
            the stride will be the same as the tile size. Defaults to
            None.
        mag (float | None, optional): magnification level of the WSI.
            If None, the function will use the default magnification
            level. Defaults to None.
        prepend_name (str, optional): prepend name to the created tile
            images and masks. Defaults to "".
        background_value (int, optional): Value to use for the
            background in the images, used when padding tiles at the
            edge of WSI. Defaults to (255, 255, 255). If set to a single
            value it will be used for all channels.
        background_index (int, optional): Index of the background class.
            Defaults to 0.
        edge_thr (float, optional): For tiles at the edge, if most of
            the tile is padded background then it is ignored. If the
            amount of tile in WSI / amount of tile padded is less than
            this threshold, the tile is ignored. Defaults to 0.25.
        ignore_labels (str | list[str] | None, optional): Labels to
            ignore. Defaults to None.
        ignore_index (int, optional): Index of the ignore class.
            Defaults to 255.
        ignore_value (tuple[int, int, int], optional): Value to use for
            the ignore class in the images. Defaults to (255, 255, 255).
            If set to a single value it will be used for all channels.
        image_type (str, optional): The type of image to save. Can be
            "rgb", "hematoxylin", or "both", defaults to "rgb".
        notebook_tqdm (bool, optional): Whether to use tqdm in a
            notebook. Defaults to False.

    Returns:
        pandas.DataFrame: A pandas dataframe containing the filepath of
            the image and mask and x, y coordinate.

    """
    if notebook_tqdm:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Should be a list of string labels.
    if ignore_labels is None:
        ignore_labels = []
    elif isinstance(ignore_labels, str):
        ignore_labels = [ignore_labels]

    # If there is an ignore label that is not in group2id, remove it.
    ignore_labels = [label for label in ignore_labels if label in group2id]

    # Read the tile source.
    ts = large_image.getTileSource(wsi_fp)

    # Get the magnification at scan.
    ts_metadata = ts.getMetadata()
    scan_mag = ts_metadata["magnification"]

    # Set the stride.
    if stride is None:
        stride = tile_size

    # Get tile size & stride at scan magnification.
    # Also, determine scaling factor to go from scan mag to desired mag.
    if mag is None:
        scan_tile_size = tile_size
        scan_stride = stride
        mag = scan_mag
        scan_to_mag_sf = 1
    else:
        scan_to_mag_sf = mag / scan_mag  # scan mag -> desired mag
        scan_tile_size = int(tile_size / scan_to_mag_sf)
        scan_stride = int(stride / scan_to_mag_sf)

    gdf = gpd.GeoDataFrame.from_features(geojson_ann_doc.get("features", []))

    # Convert the label column, which is a dictionary, set it to its 'value' key.
    gdf["label"] = gdf["label"].apply(lambda x: x.get("value", ""))

    # Filter to polygons in group2id.
    gdf = gdf[gdf[label_col].isin(group2id.keys())]

    # Remove rows with geometry is not a Polygon.
    gdf = gdf[gdf["geometry"].type == "Polygon"]

    # Scale the coordinates to mag.
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: scale(
            geom, xfact=scan_to_mag_sf, yfact=scan_to_mag_sf, origin=(0, 0)
        )
    )

    # Add an idx column, which is the map of the group to the group2id.
    gdf["idx"] = gdf[label_col].map(group2id)

    # Get the top left corner of every tile in the WSI, at scan mag.
    xys = [
        (x, y)
        for x in range(0, ts_metadata["sizeX"], scan_stride)
        for y in range(0, ts_metadata["sizeY"], scan_stride)
    ]

    # Create directory to save tile images and masks.
    save_dir = Path(save_dir)
    img_dir = save_dir / "images"
    mask_dir = save_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    tile_info = []

    for xy in tqdm(xys, desc="Tiling..."):
        x, y = xy

        # Calcualte the x and y at magnification desired.
        x_mag = int(x * scan_to_mag_sf)
        y_mag = int(y * scan_to_mag_sf)

        # Format the filepath to save the tile.
        fn = f"{prepend_name}x{x_mag}y{y_mag}.png"
        deconv_fn = f"{prepend_name}x{x_mag}y{y_mag}_deconv.png"
        img_fp = img_dir / fn
        mask_fp = mask_dir / fn
        deconv_img_fp = img_dir / deconv_fn

        # Get the tile.
        img = ts.getRegion(
            region={
                "left": x,
                "top": y,
                "right": x + scan_tile_size,
                "bottom": y + scan_tile_size,
            },
            format=large_image.constants.TILE_FORMAT_NUMPY,
            scale={"magnification": mag},
        )[0][:, :, :3].copy()

        deconv_img = None

        if image_type in ("hematoxylin", "both"):
            deconv_img = (
                htk.preprocessing.color_deconvolution.color_deconvolution(
                    img, W
                ).Stains[:, :, 0]
            )

            deconv_img = np.stack(
                [deconv_img, deconv_img, deconv_img], axis=-1
            )

        # Check if the tile area is below threshold.
        h, w = img.shape[:2]

        if h * w / (tile_size * tile_size) < edge_thr:
            continue

        if (h, w) != (tile_size, tile_size):
            # Pad the image with zeroes to make it the desired size.
            if image_type in ("rgb", "both"):
                img = cv.copyMakeBorder(
                    img,
                    0,
                    tile_size - h,
                    0,
                    tile_size - w,
                    cv.BORDER_CONSTANT,
                    value=edge_value,
                )

            if deconv_img is not None:
                deconv_img = cv.copyMakeBorder(
                    deconv_img,
                    0,
                    tile_size - h,
                    0,
                    tile_size - w,
                    cv.BORDER_CONSTANT,
                    value=edge_value,
                )

        # Create a copy of the geodataframe for this tile.
        tile_gdf = gdf.copy()

        # Translate the coordinates so 0, 0 is the top left corner of the tile.
        tile_gdf["geometry"] = tile_gdf["geometry"].apply(
            lambda geom: translate(geom, xoff=-x_mag, yoff=-y_mag)
        )

        # Draw the dataframe on the tile mask.
        mask = draw_gdf_on_array(
            tile_gdf, (tile_size, tile_size), default_value=background_id
        ).copy()

        if len(ignore_labels):
            # Apply the ignore labels to regiosn of the image(s) and mask.
            ignore_gdf = tile_gdf[
                tile_gdf[label_col].isin(ignore_labels)
            ].copy()
            ignore_gdf["idx"] = [1] * len(ignore_gdf)

            # Create a mask of the ignore labels.
            ignore_mask = draw_gdf_on_array(
                ignore_gdf,
                (h, w),
            ).copy()

            mask[np.where(ignore_mask == 1)] = ignore_id

            if image_type in ("rgb", "both"):
                img[np.where(ignore_mask == 1)] = ignore_value

            if deconv_img is not None:
                deconv_img[np.where(ignore_mask == 1)] = ignore_value

        # Save the tile image and mask.
        if image_type in ("rgb", "both"):
            imwrite(img_fp, img)
            tile_info.append([str(img_fp), str(mask_fp), x_mag, y_mag])

        if deconv_img is not None:
            imwrite(deconv_img_fp, deconv_img)
            tile_info.append(
                [
                    str(deconv_img_fp),
                    str(mask_fp),
                    x_mag,
                    y_mag,
                ]
            )

        imwrite(mask_fp, mask.astype(np.uint8), grayscale=True)

    # Convert to pandas dataframe.
    df = pd.DataFrame(tile_info, columns=["fp", "mask_fp", "x", "y"])
    return df
