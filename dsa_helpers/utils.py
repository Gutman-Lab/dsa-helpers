# Basic utility functions.
import numpy as np


def binary_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Calculate the dice coefficient between two binary masks.

    Args:
        mask1 (np.ndarray): A binary mask.
        mask2 (np.ndarray): A binary mask.

    Returns:
        float: The dice coefficient between the two masks.

    """
    # Convert the masks to binary.
    mask1 = mask1 > 0
    mask2 = mask2 > 0

    # Calculate dice.
    intersection = np.sum(mask1 * mask2)

    sum_masks = np.sum(mask1) + np.sum(mask2)

    if sum_masks == 0:
        return 1.0  # Both masks are empty

    return 2.0 * intersection / sum_masks
