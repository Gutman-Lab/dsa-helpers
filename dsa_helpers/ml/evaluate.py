"""This module contains functions for evaluating model performance on 
datasets."""

import torch
from torch import nn
from tqdm import tqdm
from transformers.trainer import Trainer
import pandas as pd

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
