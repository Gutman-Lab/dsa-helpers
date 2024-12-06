import numpy as np
from pathlib import Path
import cv2 as cv
import shutil

import transformers
from transformers import SegformerImageProcessor

import torch
import torch.nn as nn

from ... import imread
from ...tiling import tile_image


def inference(
    img: np.ndarray,
    model: transformers.models.segformer.modeling_segformer.SegformerForSemanticSegmentation,
    device: str | None = None,
    tile_size: int = 512,
    temp_dir: str = ".temp",
    batch_size: int = 8,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Inference on an image using SegFormer model for semantic segmentation.

    Args:
        img (numpy.ndarray): Image to segment.
        model (transformers.models.segformer.modeling_segformer.SegformerForSemanticSegmentation):
            Pre-trained model.
        device (str | None): Device to use. Defaults to None, in which case
            it is inferred based on devices available.
        tile_size (int): Size of the tiles. Defaults to 512.
        temp_dir (str): Temporary directory to save tiles. Defaults to ".temp".
        batch_size (int): Batch size. Defaults to 8.

    Returns:
        tuple[numpy.ndarray, list[numpy.ndarray]]: The predicted label mask and
            the contours of the mask.

    """
    # Create location to save tile images.
    tile_dir = Path(temp_dir)
    tile_dir.mkdir(exist_ok=True, parents=True)

    # Tile the image.
    tiles_df = tile_image(
        img,
        str(tile_dir),
        tile_size=tile_size,
    )

    # Create instance of the image processor.
    img_processor = SegformerImageProcessor()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    # Set model to eval mode.
    model.to(device)  # to device
    model.eval()

    # Create an empty mask the size of the image.
    h, w = img.shape[:2]
    predicted_mask = np.zeros((h + tile_size, w + tile_size), dtype=np.uint8)

    # Loop through tile images in batch size.
    for i in range(0, len(tiles_df), batch_size):
        # Get batch of tile images.
        batch = tiles_df.iloc[i : i + batch_size]

        # Read all the images in the batch.
        images = [imread(fp) for fp in batch["fp"]]

        # Process all the images.
        inputs = img_processor(
            images=images,
            return_tensors="pt",
        )

        # Send to the same device as model.
        inputs = inputs.to(device)

        with torch.no_grad():
            # Sent inputs through the model.
            outputs = model(**inputs)

            # Get out the logits.
            logits = outputs.logits

            # Reshape the logits to the original image size.
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=(tile_size, tile_size),  # (height, width)
                mode="bilinear",
                align_corners=False,
            )

            # Take the max value of the class dimension.
            pred_seg = upsampled_logits.argmax(dim=1)

            # Place tile predictions into the overall predicted mask.
            for j in range(len(batch)):
                r = batch.iloc[j]
                x, y = r["x"], r["y"]
                predicted_mask[y : y + tile_size, x : x + tile_size] = (
                    pred_seg[j].cpu().numpy()
                ).astype(np.uint8)

    # Remove the temp directory.
    shutil.rmtree(temp_dir)

    # Remove edges.
    predicted_mask = predicted_mask[:h, :w]

    # Extract the contours of the mask.
    contours = cv.findContours(
        predicted_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )[0]

    # Return the mask.
    return predicted_mask, contours
