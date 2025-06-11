import torch
from large_image_eager_iterator import LargeImagePrefetch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from PIL import Image
from large_image import getTileSource
import geopandas as gpd
from shapely.affinity import scale
import pandas as pd

from ...image_utils import label_mask_to_polygons
from ...gpd_utils import remove_gdf_overlaps


def inference(
    model: str | torch.nn.Module,
    wsi_fp: str,
    batch_size: int = 16,
    tile_size: int = 512,
    mag: float | None = None,
    workers: int = 8,
    chunk_mult: int = 2,
    prefetch: int = 2,
    device: str | None = None,
    tolerance: float | None = 2.0,
    min_area: float = 1000,
) -> gpd.GeoDataFrame:
    """Inference using SegFormer semantic segmentation model on a WSI.

    Args:
        model (str | torch.nn.Module): Path to the model checkpoint or
            a pre-loaded model.
        wsi_fp (str): File path to the WSI.
        batch_size (int, optional): Batch size for inference. Defaults
            to 16.
        tile_size (int, optional): Tile size for inference. Defaults to
            512.
        mag (float, optional): Magnification for inference. Defaults to
            None, which will use the scan magnification of WSI.
        workers (int, optional): Number of workers for inference.
            Defaults to 8.
        chunk_mult (int, optional): Chunk multiplier for inference.
            Defaults to 2.
        prefetch (int, optional): Number of prefetch for inference.
            Defaults to 2.
        device (str, optional): Device for inference. Default is None,
            will use "gpu" if available, otherwise "cpu".
        tolerance (float | None, optional): Tolerance for simplification
            of the predicted polygons. Guidelines: 0.5 for mild
            simplification, 1.0 for moderate, 2.0 for more aggressive.
            If set to None or 0.0, the polygons will not be simplified.
        min_area (float, optional): Minimum area of a polygon to be
            simplified. This is added to avoid simplifying small
            polygons that would probably lose too much information.
            Defaults to 1000.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the predicted
            polygons and labels.

    """
    # Initiate the tile iterator.
    iterator = LargeImagePrefetch(
        wsi_fp,
        batch=batch_size,
        tile_size=(tile_size, tile_size),
        scale_mode="mag",
        target_scale=mag,
        workers=workers,
        chunk_mult=chunk_mult,
        prefetch=prefetch,
        nchw=False,
        icc=True,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

    device = torch.device(device)

    # Load the model.
    if isinstance(model, str):
        model = SegformerForSemanticSegmentation.from_pretrained(
            model, local_files_only=True, device_map=device
        )

    model.eval()

    # Iterate through batches.
    batch_n = 0

    # Image processor for images.
    processor = SegformerImageProcessor()

    # Track all predicted polygons.
    wsi_polygons = []

    # Scaling factor, multiply to go from scan magnification to desired mag.
    ts_metadata = getTileSource(wsi_fp).getMetadata()
    scan_mag = ts_metadata["magnification"]

    if mag is None:
        mag = scan_mag
        sf = 1.0
    else:
        sf = mag / scan_mag

    for batch in iterator:
        # Get the batch of images.
        imgs = batch[0].view()  # returns a numpy array of shape (N, H, W, C)
        coordinates = batch[1]

        # Convert the numpy arrays to PIL images.
        imgs = [Image.fromarray(img) for img in imgs]

        # Pass the images through the processor.
        inputs = processor(imgs, return_tensors="pt")
        inputs = inputs.to(model.device)

        # Predict on the batch.
        with torch.no_grad():
            output = model(inputs["pixel_values"])
            logits = output.logits

            # Get the logits out, resizing them to the original tile size.
            logits = torch.nn.functional.interpolate(
                logits,
                size=tile_size,
                mode="bilinear",
            )

            # Get predicted class labels for each pixel.
            masks = torch.argmax(logits, dim=1).detach().cpu().numpy()

        # Loop through each mask to extract the contours as shapely polygons.
        for i, mask in enumerate(masks):
            img_metadata = coordinates[i]
            x, y = img_metadata[6], img_metadata[4]
            x = int(x * sf)
            y = int(y * sf)

            polygon_and_labels = label_mask_to_polygons(
                mask,
                x_offset=x,
                y_offset=y,
            )

            for polygon_and_label in polygon_and_labels:
                polygon, label = polygon_and_label
                label = int(label)

                # Do something with the polygon and label.
                wsi_polygons.append([polygon, label])

        batch_n += 1
        print(f"\r    Processed batch {batch_n}.    ", end="")
    print()

    # Convert polygons and labels to a GeoDataFrame.
    gdf = gpd.GeoDataFrame(wsi_polygons, columns=["geometry", "label"])
    gdf["geometry"] = gdf["geometry"].buffer(1)
    gdf = gdf.dissolve(by="label", as_index=False)
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    if tolerance:
        # Add the area column.
        gdf["area"] = gdf["geometry"].area

        # Split the dataframe into those polygons that need to be simplified and those that do not.
        gdf_to_simplify = gdf[gdf["area"] > min_area].reset_index(drop=True)
        gdf_not_to_simplify = gdf[gdf["area"] <= min_area].reset_index(
            drop=True
        )

        # Simplify the polygons that need to be simplified.
        gdf_to_simplify["geometry"] = gdf_to_simplify["geometry"].simplify(
            tolerance=tolerance, preserve_topology=True
        )

        # Concatenate the simplified polygons with the polygons that do not need to be simplified.
        gdf = pd.concat(
            [gdf_to_simplify, gdf_not_to_simplify], ignore_index=True
        )

    # Remove overlapping polygons.
    no_overlap_gdf = remove_gdf_overlaps(
        gdf,
        columns="label",
    )

    # Scale the polygon to scan magnification.
    no_overlap_gdf["geometry"] = no_overlap_gdf["geometry"].apply(
        lambda geom: scale(geom, xfact=1 / sf, yfact=1 / sf, origin=(0, 0))
    )

    return no_overlap_gdf
