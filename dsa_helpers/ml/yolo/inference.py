import large_image_source_openslide, ultralytics
import geopandas as gpd
import cv2 as cv
from shapely import Polygon

from ...utils import non_max_suppression, return_mag_and_resolution
from ...gpd_utils import remove_contained_boxes


def yolo_inference(
    model: str | ultralytics.YOLO,
    wsi_fp: str,
    mag: float | None = 20,
    mm_px: float | None = None,
    tile_size: int = 640,
    overlap: int = 480,
    batch_size: int = 64,
    prefetch: int = 2,
    workers: int = 8,
    chunk_mult: int = 2,
    agnostic_nms: bool = True,
    conf_thr: float = 0.25,
    iou: float = 0.7,
    device: str | None = None,
):
    """
    Perform YOLO inference on a whole slide image.

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
            tiles. Default is 480.
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
        geopandas.GeoDataFrame: A GeoDataFrame containing the inference
            results.

    Raises:
        ValueError: If both mag and mm_px are provided.

    """
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
            xyxys = result.boxes.xyxyn
            cls_list = result.boxes.cls
            conf_list = result.boxes.conf

            for xyxy, cls, conf in zip(xyxys, cls_list, conf_list):
                # cls and conf are tensorts in device, convert to to int and float
                cls = int(cls)
                conf = float(conf)

                x1, y1, x2, y2 = xyxy

                # (1) scale to scan resolution, (2) shift to tile location.
                x1 = int(x1 * scan_tile_x) + x
                y1 = int(y1 * scan_tile_y) + y
                x2 = int(x2 * scan_tile_x) + x
                y2 = int(y2 * scan_tile_y) + y

                # Create the polygon.
                geom = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                boxes.append([cls, x1, y1, x2, y2, conf, geom])

    gdf = gpd.GeoDataFrame(
        boxes, columns=["label", "x1", "y1", "x2", "y2", "conf", "geometry"]
    )
    gdf["box_area"] = gdf.geometry.area

    gdf = non_max_suppression(gdf, conf_thr)
    gdf = remove_contained_boxes(gdf, iou)

    return gdf.reset_index(drop=True)
