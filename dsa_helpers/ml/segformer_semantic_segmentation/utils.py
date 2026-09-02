from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy

datasets = lazy.load("datasets")
pd = lazy.load("pandas")

if TYPE_CHECKING:
    import pandas as pd

    from .datasets import SegFormerSegmentationDataset


def dataset_generator(dataset):
    """Yield a dataset."""
    for item in dataset:
        yield item


def create_segformer_segmentation_dataset(
    df: pd.DataFrame | str,
    low_memory: bool = True,
    transforms=None,
    extras: bool = False,
    group: str | None = None,
) -> SegFormerSegmentationDataset:
    """Create a SegFormer segmentation dataset from a DataFrame.

    Args:
        df (pandas.DataFrame): Must have columns "fp" and "mask_fp".
            Additionally, it can have columns "x", "y". If these are not
            present, then they will be set to 0.
        low_memory (bool, optional): Whether to read the CSV file in low
            memory mode. Default is True.
        transforms: Function to add transforms to the images.
        group (str | None, optional): Additional column to pass to
            the model, if None it will be an empty string. Default
            is None.

    Returns:
        (SegFormerSegmentationDataset): A Dataset object to be used for
        HuggingFaces SegFormer model training.

    """
    from .datasets import SegFormerSegmentationDataset

    if isinstance(df, str):
        df = pd.read_csv(df, low_memory=low_memory)

    dataset = SegFormerSegmentationDataset(df, group=group, extras=extras)

    if extras:
        features = datasets.Features(
            {
                "pixel_values": datasets.Image(),
                "label": datasets.Image(),
                "x": datasets.Value("int32"),
                "y": datasets.Value("int32"),
                "group": datasets.Value("string"),
            }
        )
    else:
        features = datasets.Features(
            {
                "pixel_values": datasets.Image(),
                "label": datasets.Image(),
            }
        )
    dataset = datasets.Dataset.from_generator(
        generator=lambda: dataset_generator(dataset), features=features
    )

    if transforms is not None:
        dataset.set_transform(transforms)

    return dataset
