import large_image_source_openslide, ultralytics, large_image
import geopandas as gpd
import cv2 as cv
import numpy as np
from shapely import Polygon
from time import perf_counter

from ...utils import non_max_suppression, return_mag_and_resolution
from ...gpd_utils import remove_contained_boxes


def yolo_inference(
    model: str | ultralytics.YOLO,
    wsi_fp: str,
    mag: float | None = 20,
    mm_px: float | None = None,
    tile_size: int = 640,
    overlap: int = 160,
    batch_size: int = 64,
    prefetch: int = 2,
    workers: int = 8,
    chunk_mult: int = 2,
    agnostic_nms: bool = True,
    conf_thr: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
) -> dict:
    """Perform YOLO inference on a whole slide image.

    Args:
        model (str | ultralytics.YOLO): YOLO model or path to the model
            weights.
        wsi_fp (str): File path to the whole slide image.
        mag (float | None, optional): Desired magnification for tiling.
            Note magnification meaning can vary among scanners. We
            convert this to a corresponding resolution defined by a
            standard magnification to resolution ratio (Emory scanner).
            Default is None. Defined by mm_px, or if that is None than
            the scan resolution along the x axis is used.
        mm_px (float | None, optional): Micrometers per pixel for the
            desired resolution. If not provided, will be inferred from
            the mag parameter by standardizing to the Emory scanner. If
            both are not provided, the scan resolution in the x
            direction is used. Default is None.
        tile_size (int, optional): Image is tiled into smaller images,
            tiles, of this size at the desired resolution.
        overlap (int, optional): The number of pixels to overlap between
            tiles. Default is 160.
        batch_size (int, optional): The number of tiles to process in a
            single batch. Used both for the eager iterator and the YOLO
            training. Default is 64.
        prefetch (int, optional): The number of batches to prefetch.
            Used for the eager iterator. Default is 2.
        workers (int, optional): The number of workers to use for
            parallel processing. Used for the eager iterator. Default is
            8.
        chunk_mult (int, optional): The multiplier for the number of tiles
            to process in a single batch. Used for the eager iterator.
            Default is 2.
        agnostic_nms (bool, optional): Whether to use agnostic NMS.
            Default is True.
        conf_thr (float, optional): The confidence threshold for the
            NMS. Default is 0.25.
        iou (float, optional): The IoU threshold for the NMS. Default is
            0.7.
        device (str | None, optional): The device to use for inference.
            Default is None, will use "cuda" if available, otherwise
            "cpu". You can specify values like "cuda:0", "0", or
            multiple devices separated by commas.

    Returns:
        dict: A dictionary containing the inference results.
            - gdf (geopandas.GeoDataFrame): The inference results as a
              GeoDataFrame.
            - mag (float): The magnification used for inference.
            - mm_px (float): The micrometers per pixel used for inference.
            - time (dict): A dictionary containing the time taken for
              inference.
                - total (float): The total time taken for inference.

    Raises:
        ValueError: If both mag and mm_px are provided.
    """
    start_time = perf_counter()
    if isinstance(model, str):
        model = ultralytics.YOLO(model)

    ts = large_image_source_openslide.open(wsi_fp)
    ts_metadata = ts.getMetadata()

    mm_x = ts_metadata["mm_x"]
    mm_y = ts_metadata["mm_y"]

    # Get the desired resolution.
    if mag is not None and mm_px is not None:
        raise ValueError("Only one of mag or mm_px can be provided.")
    if mag is None and mm_px is None:
        # Use the scan resolution, we use the x resolution.
        mag, mm_px = return_mag_and_resolution(mm_px=mm_x)
    else:
        mag, mm_px = return_mag_and_resolution(mag=mag, mm_px=mm_px)

    # Calculate the x and y size of the tile at scan resolution.
    # desired resolution x sf_* -> scan resolution
    sf_x = mm_px / mm_x
    sf_y = mm_px / mm_y

    scan_tile_x = int(tile_size * sf_x)
    scan_tile_y = int(tile_size * sf_y)

    iterator = ts.eagerIterator(
        scale={"mm_x": mm_px, "mm_y": mm_px},
        tile_size={"width": tile_size, "height": tile_size},
        tile_overlap={"x": overlap, "y": overlap},
        chunk_mult=chunk_mult,
        batch=batch_size,
        prefetch=prefetch,
        workers=workers,
    )

    # Iterate through the WSI.
    boxes = []

    for i, batch in enumerate(iterator, start=1):
        print(f"\rProcessing batch {i}...    ", end="")
        tiles = batch["tile"].view()  # images are in BCHW format, as numpy

        # Convert a list of numpys, and change from RGB to BGR.
        tiles = [cv.cvtColor(tile, cv.COLOR_RGB2BGR) for tile in tiles]
        results = model(
            tiles,
            imgsz=tile_size,
            batch=batch_size,
            agnostic_nms=agnostic_nms,
            verbose=False,
            conf=conf_thr,
            iou=iou,
            device=device,
        )

        # Tile location at scan resolution.
        tile_x_coords = batch["gx"]
        tile_y_coords = batch["gy"]

        # Loop through tile variables.
        for result, x, y in zip(results, tile_x_coords, tile_y_coords):
            xyxys = result.boxes.xyxy
            cls_list = result.boxes.cls
            conf_list = result.boxes.conf

            for xyxy, cls, conf in zip(xyxys, cls_list, conf_list):
                # cls and conf are tensorts in device, convert to to int and float
                cls = int(cls)
                conf = float(conf)

                x1, y1, x2, y2 = xyxy

                # (1) scale to scan resolution, (2) shift to tile location.
                x1 = int(x1 * sf_x) + x
                y1 = int(y1 * sf_y) + y
                x2 = int(x2 * sf_x) + x
                y2 = int(y2 * sf_y) + y

                # Create the polygon.
                geom = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                boxes.append([cls, x1, y1, x2, y2, conf, geom])

    gdf = gpd.GeoDataFrame(
        boxes, columns=["label", "x1", "y1", "x2", "y2", "conf", "geometry"]
    )
    gdf["box_area"] = gdf.geometry.area

    gdf = non_max_suppression(gdf, conf_thr)
    gdf = remove_contained_boxes(gdf, iou).reset_index(drop=True)

    return {
        "gdf": gdf,
        "mag": mag,
        "mm_px": mm_px,
        "time": {"total": perf_counter() - start_time},
    }


def yolo_inference_on_region(
    wsi_fp: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
    weights_fp: str,
    mag: float | None = 20,
    mm_px: float | None = None,
    tile_size: int = 640,
    stride: int | None = 480,
    batch_size: int = 64,
    agnostic_nms: bool = True,
    conf_thr: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
    pad_rgb: tuple[int, int, int] = (114, 114, 114),
    tile_area_threshold: float = 0.25,
    nms_iou_thr: float = 0.25,
    remove_contained_thr: float = 0.7,
) -> tuple[gpd.GeoDataFrame, float, float]:
    """Inference on a region of a WSI using a YOLO model.

    Args:
        wsi_fp (str): The file path to the WSI.
        left (int): The left coordinate of the region.
        top (int): The top coordinate of the region.
        right (int): The right coordinate of the region.
        bottom (int): The bottom coordinate of the region.
        weights_fp (str): The file path to the YOLO model weights.
        mag (float | None, optional): The magnification to use for the
            inference. Default is 20.
        mm_px (float | None, optional): The pixel size to use for the
            inference. Default is None.
        tile_size (int, optional): The size of the tiles to use for the
            inference. This is the size of the tiles at the desired
            resolution. Default is 640.
        stride (int, optional): The stride to use for the inference. If
            None, then the stride will be the tile size (no overlap). This
            is the stride at the desired resolution. Default is 480.
        batch_size (int, optional): The batch size to use for the
            inference. Default is 64.
        agnostic_nms (bool, optional): Whether to use agnostic NMS.
            Default is True.
        conf_thr (float, optional): The confidence threshold to use for
            the inference. Default is 0.25.
        iou (float, optional): The IoU threshold to use for the
            inference. Default is 0.7.
        device (str | None, optional): The device to use for the
            inference. Default is None.
        pad_rgb (tuple[int, int, int], optional): The RGB values to pad
            the tiles with. Default is (114, 114, 114).
        tile_area_threshold (float, optional): The threshold for the tile
            area in region. Default is 0.25.
        nms_iou_thr (float, optional): The IoU threshold to use for the
            NMS after inference. Default is 0.25.
        remove_contained_thr (float, optional): The threshold for the
            contained boxes. Default is 0.7.

    Returns:
        tuple[gpd.GeoDataFrame, float, float]: A tuple containing the
            inference results, the magnification, and the pixel size used.

    Raises:
        ValueError: If the region is out of bounds.
    """
    assert left < right and top < bottom, "Region is out of bounds."

    if stride is None:
        stride = tile_size

    # Load the YOLO model.
    model = ultralytics.YOLO(weights_fp)

    ts = large_image_source_openslide.open(wsi_fp)
    ts_metadata = ts.getMetadata()

    mm_x = ts_metadata["mm_x"]
    mm_y = ts_metadata["mm_y"]

    # Get the desired resolution.
    if mag is not None and mm_px is not None:
        raise ValueError("Only one of mag or mm_px can be provided.")
    if mag is None and mm_px is None:
        # Use the scan resolution, we use the x resolution.
        mag, mm_px = return_mag_and_resolution(mm_px=mm_x)
    else:
        mag, mm_px = return_mag_and_resolution(mag=mag, mm_px=mm_px)

    # Calculate the x and y size of the tile at scan resolution.
    # desired resolution x sf_* -> scan resolution
    sf_x = mm_px / mm_x
    sf_y = mm_px / mm_y

    scan_tile_x = int(tile_size * sf_x)
    scan_tile_y = int(tile_size * sf_y)
    scan_stride_x = int(stride * sf_x)
    scan_stride_y = int(stride * sf_y)

    # Create a low res mask for the region.
    wsi_w = ts_metadata["sizeX"]
    wsi_h = ts_metadata["sizeY"]

    if right > wsi_w or bottom > wsi_h:
        raise ValueError("Region is out of bounds.")

    # Calculate the x, y coordinates
    xys = []

    for x in range(left, right, scan_stride_x):
        for y in range(top, bottom, scan_stride_y):
            xys.append((x, y))

    if len(xys) == 0:
        raise ValueError("No tiles to process with given region.")

    # Process in batches.
    batch_indices = np.arange(0, len(xys), batch_size)

    if len(batch_indices) == 1:
        print(
            f"Found {len(xys)} tiles to process in {len(batch_indices)} batch."
        )
    else:
        print(
            f"Found {len(xys)} tiles to process in {len(batch_indices)} batches."
        )

    # Iterate through the WSI.
    boxes = []

    # Calculate the tile area to be above the threshold.
    tile_area = tile_size * tile_size
    minimum_tile_area = tile_area * tile_area_threshold

    for batch_idx in batch_indices:
        xy_index = batch_idx * batch_size

        # Get the tiles.
        batch_xys = xys[xy_index : xy_index + batch_size]

        tiles = []

        tile_x_coords = []
        tile_y_coords = []

        for xy in batch_xys:
            x1, y1 = xy

            x2 = x1 + scan_tile_x

            if x2 > right:
                x2 = right

            y2 = y1 + scan_tile_y

            if y2 > bottom:
                y2 = bottom

            # Get the tiles.
            tile = ts.getRegion(
                region={
                    "left": x1,
                    "top": y1,
                    "right": x2,
                    "bottom": y2,
                },
                format=large_image.constants.TILE_FORMAT_NUMPY,
                scale={"mm_x": mm_px, "mm_y": mm_px},
            )[0][:, :, :3].copy()

            # Pad the tile if it is not the right shape.
            tile_h, tile_w, _ = tile.shape

            tile_area = tile_h * tile_w

            if tile_area < minimum_tile_area:
                continue

            tile_x_coords.append(x1)
            tile_y_coords.append(y1)

            if tile_h != tile_size or tile_w != tile_size:
                tile = cv.copyMakeBorder(
                    tile,
                    0,
                    tile_size - tile_h,
                    0,
                    tile_size - tile_w,
                    cv.BORDER_CONSTANT,
                    value=pad_rgb,
                )
            # Convert to BGR.
            tile = cv.cvtColor(tile, cv.COLOR_RGB2BGR)

            tiles.append(tile)

        if len(tiles) == 0:
            continue

        results = model(
            tiles,
            imgsz=tile_size,
            batch=batch_size,
            agnostic_nms=agnostic_nms,
            verbose=False,
            conf=conf_thr,
            iou=iou,
            device=device,
        )

        # Loop through tile variables.
        for result, x, y in zip(results, tile_x_coords, tile_y_coords):
            xyxys = result.boxes.xyxy
            cls_list = result.boxes.cls
            conf_list = result.boxes.conf

            for xyxy, cls, conf in zip(xyxys, cls_list, conf_list):
                # cls and conf are tensorts in device, convert to to int and float
                cls = int(cls)
                conf = float(conf)

                x1, y1, x2, y2 = xyxy

                # (1) scale to scan resolution, (2) shift to tile location.
                x1 = int(x1 * sf_x) + x
                y1 = int(y1 * sf_y) + y
                x2 = int(x2 * sf_x) + x
                y2 = int(y2 * sf_y) + y

                # Create the polygon.
                geom = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                boxes.append([cls, x1, y1, x2, y2, conf, geom])

    gdf = gpd.GeoDataFrame(
        boxes, columns=["label", "x1", "y1", "x2", "y2", "conf", "geometry"]
    )
    gdf["box_area"] = gdf.geometry.area

    # Clean up predictions further.
    if nms_iou_thr > 0:
        gdf = non_max_suppression(gdf, nms_iou_thr)
    if remove_contained_thr > 0:
        gdf = remove_contained_boxes(gdf, remove_contained_thr)

    return gdf, mag, mm_px
