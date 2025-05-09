from torch import nn
import pandas as pd
import numpy as np
import torch

from .transforms import val_transforms
from .utils import create_segformer_segmentation_dataset


def per_class_dice_on_dataset(
    model: nn.Module,
    data: pd.DataFrame | str,
    label2id: dict[str, int],
    batch_size: int = 16,
    device: torch.device | None = None,
    tile_size: int = 512,
    tqdm_notebook: bool = False,
):
    """Calculate the DICE coefficient for a given model and set of
    images. The model must be a semantic segmentation model that outputs
    an object with the logits attribute. The dataset must be iterable
    and return a dictionary with the keys "pixel_values" and "labels".
    The pixel values should be in the format the model expects for the
    input.

    Args:
        model (nn.Module): The model to evaluate.
        data (pd.DataFrame | str): The dataframe or path to a csv file
            containing the data to evaluate on. The dataframe must have
            the "fp" (filepath to image) and "mask_fp" (filepath to
            mask) columns.
        label2id (dict): A dictionary mapping labels to their integer
            representations.
        batch_size (int, optional): The batch size to use for evaluation.
            Default is 16.
        device (torch.device | None, optional): The device to use for
            evaluation. If None, the device will be "cuda" if available,
            otherwise "cpu". Default is None.
        tile_size (int, optional): The size of the tiles to use for
            evaluation. Default is 512.
        tqdm_notebook (bool, optional): Whether to use a tqdm notebook.
            Default is False.

    Returns:
        dict: A dictionary with the keys "mean_dice" and the DICE
            coefficient for each label.
    """
    if tqdm_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    if isinstance(data, str):
        data = pd.read_csv(data)

    # Create the dataset object.
    dataset = create_segformer_segmentation_dataset(
        data, transforms=val_transforms
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    # Move the model to the device.
    model.to(device)
    model.eval()  # model should be in evaluation mode

    # Number of images in the dataset.
    n = len(dataset)

    # Initialize the intersection and denominator for each label.
    intersection = {label: 0 for label in label2id}
    denominator = {label: 0 for label in label2id}

    batches = list(range(0, n, batch_size))

    for i in tqdm(batches):
        # Get the batch of images and labels.
        batch = dataset[i : i + batch_size]

        inputs = np.array(batch["pixel_values"])
        labels = np.array(batch["labels"])
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = np.array(labels)

        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = model(inputs)
            logits = outputs.logits

            # Get the logits out, resizing them to the original tile size.
            logits = torch.nn.functional.interpolate(
                logits,
                size=tile_size,
                mode="bilinear",
            )

            # Get predicted class labels for each pixel.
            masks = torch.argmax(logits, dim=1).detach().cpu().numpy()

            # Flatten both masks and labels.
            masks = masks.flatten()
            labels = labels.flatten()

            for label, integer in label2id.items():
                # Calculate where both mask and labels are equal to the integer.
                gt_mask = labels == integer
                pred_mask = masks == integer
                intersection[label] += np.sum(gt_mask & pred_mask)
                denominator[label] += np.sum(gt_mask) + np.sum(pred_mask)

    # Calculate the dice for each class.
    metrics = {}
    for label in label2id:
        denominator_value = denominator[label]

        if denominator_value:
            metrics[f"{label}_dice"] = float(
                2 * intersection[label] / denominator[label]
            )
        else:
            metrics[f"{label}_dice"] = 1

    # Calculate the mean dice.
    return metrics
