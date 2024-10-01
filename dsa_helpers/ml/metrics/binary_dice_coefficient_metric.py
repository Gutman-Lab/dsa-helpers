import torch
from torch import nn
from ...utils import binary_dice_coefficient


def binary_dice_coefficient_metric(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred  # assumed both are numpy

        logits_tensor = torch.from_numpy(logits).cpu()
        labels_tensor = torch.from_numpy(labels).cpu()

        # Scale the logits to the size of the label.
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().numpy()
        labels = labels_tensor.numpy()

        return {"dice_coefficient": binary_dice_coefficient(pred_labels, labels)}
