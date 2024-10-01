from transformers import SegformerImageProcessor
from torchvision.transforms import ColorJitter


def train_transforms(example_batch):
    """Default transforms for training images."""
    processor = SegformerImageProcessor()  # required for using SegFormer model.
    jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    images = [jitter(x) for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = processor(images, labels)
    return inputs


def val_transforms(example_batch):
    """Default transforms for validation images."""
    processor = SegformerImageProcessor()  # required for using SegFormer model.

    images = [x for x in example_batch["pixel_values"]]
    labels = [x for x in example_batch["label"]]
    inputs = processor(images, labels)
    return inputs
