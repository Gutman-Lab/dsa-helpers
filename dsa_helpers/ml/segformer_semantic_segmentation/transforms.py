from transformers import SegformerImageProcessor
from torchvision.transforms import ColorJitter
import albumentations as A
from typing import Callable
import numpy as np
from PIL import Image


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
    square_symmetry_prob: float | None = 1.0,
    rotate_limit: int = 45,
    rotate_prob: float | None = 0.5,
    fill: tuple[float, ...] | float = 255.0,
    fill_mask: tuple[float, ...] | float = 0.0,
    brightness: float = 0.25,
    contrast: float = 0.25,
    saturation: float = 0.25,
    hue: float = 0.1,
) -> Callable:
    """Get a train transform function that can be used to transform a
    batch of images and labels. Used with batches for segformer
    semantic segmentation models.

    Args:
        square_symmetry_prob (float | None, optional): Probability of
            applying square symmetry. Default is 1.0.
        rotate_limit (int, optional): Maximum rotation limit in degrees.
            Default is 45.
        rotate_prob (float | None, optional): Probability of applying
            rotation. Default is 0.5.
        fill (tuple[float, ...] | float, optional): Fill value for the
            image. Default is 255.0.
        fill_mask (tuple[float, ...] | float, optional): Fill value for
            the mask. Default is 0.0.
        brightness (float, optional): Brightness adjustment factor.
            Default is 0.25.
        contrast (float, optional): Contrast adjustment factor.
            Default is 0.25.
        saturation (float, optional): Saturation adjustment factor.
            Default is 0.25.
        hue (float, optional): Hue adjustment factor. Default is 0.1.

    Returns:
        Callable: A function that can be used to transform a batch of
            images and labels.

    """
    albumentation_pipeline = []

    if square_symmetry_prob is not None:
        albumentation_pipeline.append(A.SquareSymmetry(p=square_symmetry_prob))

    if rotate_prob is not None:
        albumentation_pipeline.append(
            A.Rotate(
                limit=rotate_limit,
                fill=fill,
                p=rotate_prob,
                fill_mask=fill_mask,
            )
        )

    if len(albumentation_pipeline):
        albumentation_pipeline = A.Compose(albumentation_pipeline)
    else:
        albumentation_pipeline = None

    jitter = ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    processor = SegformerImageProcessor()

    def transform(batch):
        if albumentation_pipeline is not None:
            images, labels = [], []

            # Pass through the albumentation pipeline.
            for img, label in zip(batch["pixel_values"], batch["label"]):
                # Apply the albumentation pipeline.
                img = np.array(img)
                label = np.array(label)

                augmented = albumentation_pipeline(image=img, mask=label)
                images.append(Image.fromarray(augmented["image"]))
                labels.append(Image.fromarray(augmented["mask"]))
        else:
            images = batch["pixel_values"]
            labels = batch["label"]

        # Apply color jitter.
        images = [jitter(x) for x in images]

        # Process the images and labels.
        return processor(images, labels)

    return transform
