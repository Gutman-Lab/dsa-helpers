"""Functions for evaluating datasets with machine learning models.

- evaluate_semantic_segmentation_segformer_group: Evaluate a segformer
  trainer model on a specific dataset provided as a dataframe, grouped
  by a specific column.
- per_class_dice_on_dataset: Per class DICE coefficient on a set of
    images.

"""

import torch
from torch import nn
from tqdm import tqdm
from transformers.trainer import Trainer
import pandas as pd
import numpy as np
from .datasets.utils import create_segformer_segmentation_dataset
from .transforms.segformer_transforms import val_transforms
from ..tile_utils import merge_tile_masks


def evaluate_semantic_segmentation_segformer_group(
    trainer: Trainer,
    df: pd.DataFrame,
    group_col: str,
    batch_size: int = 16,
    tile_size: int = 512,
    background_label: int | None = None,
) -> float:
    """Evaluate a segformer trainer model on a specific dataset provided
    as a dataframe, grouped by a specific column. For a unique value of
    the column it will predict on the images for that group and then
    merge the predictions into a single multipolygon. The mean IoU will
    be calculated agains the ground truth masks for that group. The
    reported mean IoU will be the average of all the mean IoUs for each
    group.

    Args:
        trainer (Trainer): The trainer object with the model to evaluate.
        df (pandas.DataFrame): The dataframe with the tile metadata.
            Must have the fp column (file path), x column (x coordinate),
            y column (y coordinate), and group_col.
        group_col (str): The column to group the tiles by.
        batch_size (int): The batch size to use for prediction. Note that
            this is not the batch size use for predictions, rather it is
            how many masks will be stored in memory. Batch size used
            for predictions is specified on the trainer input. Default is
            16.
        tile_size (int): The size of the tiles. Default is 512.
        background_label (int | None): The label to use for the background
            class, which will be ignored. If None then no label will be
            ignored. Default is None.

    Returns:
        float: The mean IoU across all groups.

    """
    ious = []

    # Predict on the images for each group separately.
    groups = list(df[group_col].unique())
    for group in tqdm(groups, desc="Evaluating groups"):
        # Get the tiles in this group.
        group_df = df[df[group_col] == group]

        pred_tile_list = []
        true_tile_list = []

        # Predict on the group in batches.
        for i in range(0, len(group_df), batch_size):
            # Create dataset for this batch.
            batch_df = group_df.iloc[i : i + batch_size]
            x_list = batch_df["x"].tolist()
            y_list = batch_df["y"].tolist()

            # Create dataset for this batch.
            group_dataset = create_segformer_segmentation_dataset(
                batch_df, transforms=val_transforms
            )

            # Predict on the batch.
            out = trainer.predict(group_dataset)
            preds = out[0]
            true = out[1]

            # Reshape the prediction logits to tile size.
            preds = torch.from_numpy(preds).cpu()

            preds = nn.functional.interpolate(
                preds,
                size=(tile_size, tile_size),
                mode="bilinear",
                align_corners=False,
            ).argmax(
                dim=1
            )  # logits -> class predictions

            preds = preds.detach().numpy()

            # Append to the tile list.
            pred_tile_list.extend(
                [(pred, x, y) for x, y, pred in zip(x_list, y_list, preds)]
            )

            true_tile_list.extend(
                [(mask, x, y) for x, y, mask in zip(x_list, y_list, true)]
            )

        # Merge the tile masks.
        pred_gdf = merge_tile_masks(
            pred_tile_list, background_label=background_label
        )
        true_gdf = merge_tile_masks(
            true_tile_list, background_label=background_label
        )

        # Get the unique labels in either the prediction or the ground truth.
        labels = set(pred_gdf["label"].unique()) | set(
            true_gdf["label"].unique()
        )

        # Calculate the IoU for each label.
        ious = []

        for label in labels:
            pred_label_gdf = pred_gdf[pred_gdf["label"] == label]
            true_label_gdf = true_gdf[true_gdf["label"] == label]

            # If either the prediction or the ground truth is empty, the IoU is 0.
            if pred_label_gdf.empty or true_label_gdf.empty:
                ious.append(0)
            else:
                # Calculate the IoU.
                geom1 = pred_label_gdf.iloc[0]["geometry"]
                geom2 = true_label_gdf.iloc[0]["geometry"]

                intersection_area = geom1.intersection(geom2).area
                union = geom1.union(geom2).area

                ious.append(intersection_area / union)

    # Return the average of the ious.
    return float(sum(ious) / len(ious))


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


def evaluate_semantic_segmentation_segformer_by_wsi(
    model: nn.Module,
    data: pd.DataFrame | str,
    group_col: str,
    label2id: dict[str, int],
    batch_size: int = 16,
    tile_size: int = 512,
    background_label: int | None = None,
    device: torch.device | None = None,
) -> float:
    """Evaluate a segformer trainer model on a specific dataset provided
    as a dataframe, grouped by a specific column. For a unique value of
    the column it will predict on the images for that group and then
    merge the predictions into a single multipolygon. The mean IoU will
    be calculated agains the ground truth masks for that group. The
    reported mean IoU will be the average of all the mean IoUs for each
    group.

    Args:
        model (nn.Module): The model to evaluate.
        data (pd.DataFrame | str): The dataframe or path to a csv file
            containing the data to evaluate on. The dataframe must have
            the "fp" (filepath to image) and "mask_fp" (filepath to
            mask) columns. The column matching the "group_col" parameter
            will be used to group tiles into groups. The "x" and "y"
            parameters will be used for stiching the images back to
            their original size.
        group_col (str): The column to group the tiles by.
        batch_size (int): The batch size to use for prediction. Note that
            this is not the batch size use for predictions, rather it is
            how many masks will be stored in memory. Batch size used
            for predictions is specified on the trainer input. Default is
            16.
        label2id (dict): A dictionary mapping labels to their integer
            representations.
        tile_size (int): The size of the tiles. Default is 512.
        background_label (int | None): The label to use for the background
            class, which will be ignored. If None then no label will be
            ignored. Default is None.
        device (torch.device | None, optional): The device to use for
            evaluation. If None, the device will be "cuda" if available,
            otherwise "cpu". Default is None.

    Returns:
        float: The mean IoU across all groups.

    """
    # Read the data.
    if isinstance(data, str):
        data = pd.read_csv(data)

    ious = []

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the device.
    model.to(device)
    model.eval()  # model should be in evaluation mode

    # Predict on the images for each group separately.
    groups = list(data[group_col].unique())
    for group in tqdm(groups, desc="Evaluating groups"):
        # Get the tiles in this group.
        group_df = data[data[group_col] == group]

        pred_tile_list = []
        true_tile_list = []

        # Predict on the group in batches.
        for i in range(0, len(group_df), batch_size):
            # Create dataset for this batch.
            batch_df = group_df.iloc[i : i + batch_size]
            x_list = batch_df["x"].tolist()
            y_list = batch_df["y"].tolist()

            # Create dataset for this batch.
            dataset = create_segformer_segmentation_dataset(
                batch_df, transforms=val_transforms
            )

            # Get the inputs.
            inputs = np.array(dataset[:]["pixel_values"])
            inputs = torch.tensor(inputs, dtype=torch.float32)
            inputs = inputs.to(device)

            true = np.array(dataset[:]["labels"])

            # Predict on the batch.
            with torch.no_grad():
                out = model(inputs)
                preds = out.logits

                # Reshape the prediction logits to tile size.
                preds = torch.from_numpy(preds)

                preds = nn.functional.interpolate(
                    preds,
                    size=(tile_size, tile_size),
                    mode="bilinear",
                    align_corners=False,
                ).argmax(
                    dim=1
                )  # logits -> class predictions

                preds = preds.detach().cpu().numpy()

                # Append to the tile list.
                pred_tile_list.extend(
                    [(pred, x, y) for x, y, pred in zip(x_list, y_list, preds)]
                )

                true_tile_list.extend(
                    [(mask, x, y) for x, y, mask in zip(x_list, y_list, true)]
                )

        # Merge the tile masks.
        pred_gdf = merge_tile_masks(
            pred_tile_list, background_label=background_label
        )
        true_gdf = merge_tile_masks(
            true_tile_list, background_label=background_label
        )

        # Calculate the Dice for each label.
        metrics = {label: [] for label in label2id}

        for label, integer in label2id.items():
            if integer == background_label:
                continue  # don't calculate dice for background

            # For this integer, get the prediction and ground truth.
            pred_label_gdf = pred_gdf[pred_gdf["label"] == integer]
            true_label_gdf = true_gdf[true_gdf["label"] == integer]

            # If both are empty, the dice is 1.
            if pred_label_gdf.empty and true_label_gdf.empty:
                metrics[label].append(1)
            else:
                # Calculate the intersection and denominator.
                intersection = pred_label_gdf.geometry.intersection(
                    true_label_gdf.geometry
                ).area.sum()

                # The denominator is the sum area of the two polygons.
                denominator = (
                    pred_label_gdf.geometry.area.sum()
                    + true_label_gdf.geometry.area.sum()
                )

                # The dice is 2 * intersection / denominator.
                metrics[label].append(float(2 * intersection / denominator))

    # Return the dice as means and standard deviations.
    metrics = {
        label: {
            "mean": float(np.mean(metrics[label])),
            "std": float(np.std(metrics[label])),
        }
        for label in label2id
    }
    return metrics
