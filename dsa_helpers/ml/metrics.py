# -*- coding: utf-8 -*-
"""Metric functions.

This module contains functions used to evaluate the performance of 
machine learning models.

"""
import torch
from torch import nn
import numpy as np


def binary_dice_coefficient(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict:
    """Calculate the dice coefficient for binary masks. Developed for use
    in compute_metrics parameters of transformers Trainer class, for
    SegFormer senantic segmentation model of two classes.

    Args:
        eval_pred (tuple[np.ndarray, np.ndarray]): A tuple containing the
            predicted and true masks.

    Returns:
        dict: A dictionary containing the dice coefficient.

    """
    with torch.no_grad():
        logits, labels = eval_pred

        # Convert to tensors.
        logits = torch.from_numpy(logits).cpu()

        # Resize the logits to be the same shape as the labels.
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        preds = logits.detach().numpy()

        # Convert the masks to binary.
        preds = preds > 0
        labels = labels > 0

        # Calculate dice.
        intersection = np.sum(preds * labels)
        union = np.sum(preds) + np.sum(labels)

        if union == 0:
            dice = 1.0  # Both masks are empty
        else:
            dice = 2.0 * intersection / union

        return {"dice_coefficient": dice}


def mean_iou(label2idx: dict):
    """Return the compute metric function for mean intersection over
    union for a multi-class semantic segmentation task.

    Args:
        label2idx (dict): A dictionary mapping class labels to indices.

    Returns:
        function: The compute_metrics function.

    """
    idx2label = {v: k for k, v in label2idx.items()}

    def compute_metrics(eval_pred):
        """Compute the mean intersection over union (IoU) for semantic
        segmentation. Also returns the IoU for each class."""
        with torch.no_grad():
            logits, labels = eval_pred

            # Convert logits to tensor.
            logits_tensor = torch.from_numpy(logits).cpu()
            labels_tensor = torch.from_numpy(labels).cpu()

            # From logits get the number of classes in the dataset.
            num_classes = logits.shape[1]

            # Scale the logits back to the shape of the labels.
            logits_tensor = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            # Turn the logits to predictions.
            pred = logits_tensor.argmax(dim=1)

            # Calculate the IoU for each class.
            ious = []

            metrics = {}

            for cls in range(num_classes):
                pred_mask = pred == cls
                labels_mask = labels_tensor == cls

                intersection = (
                    torch.logical_and(pred_mask, labels_mask).sum().item()
                )
                union = torch.logical_or(pred_mask, labels_mask).sum().item()

                if union == 0:
                    iou = float("nan")  # or 1.0
                else:
                    iou = intersection / union

                metrics[f"{idx2label[cls]}_iou"] = iou

                ious.append(iou)

            mean_iou = np.nanmean(ious)  # Use NumPy for NaN handling
            metrics["mean_iou"] = mean_iou

            return metrics

    return compute_metrics
