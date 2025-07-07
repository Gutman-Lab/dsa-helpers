from transformers import SegformerImageProcessor
from torchvision.transforms import ColorJitter
import albumentations as A
from typing import Callable


def val_transforms(example_batch):
    """Default transforms for validation images."""
    processor = SegformerImageProcessor()

    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]

    inputs = processor(images, labels)

    return inputs


def train_transforms(example_batch):
    """Default transforms for training images."""

    processor = SegformerImageProcessor()
    jitter = ColorJitter(
        brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1
    )

    images = [jitter(x) for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]

    inputs = processor(images, labels)

    return inputs


def get_train_transforms(
    square_symmetry_prob: float | None = 1.,
    rotate_limit: int = 45,
    rotate_prob: float | None = 0.5
    rotate_fill: int
) -> Callable:
    """Get a train transform function that can be used to transform a
    batch of images and labels. Used with batches for segformer
    semantic segmentation models.

    Args:

    """
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
        ]
    )
