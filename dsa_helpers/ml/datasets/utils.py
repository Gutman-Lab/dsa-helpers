from __future__ import annotations

from typing import TYPE_CHECKING

import lazy_loader as lazy
from colorama import Fore, Style

pd = lazy.load("pandas")
datasets = lazy.load("datasets")

if TYPE_CHECKING:
    import pandas as pd


def dataset_generator(dataset):
    """Yield a dataset."""
    for item in dataset:
        yield item


def create_segformer_segmentation_dataset(
    df: pd.DataFrame | str, low_memory: bool = True, transforms=None
):
    """DEPRECATED: see
    dsa_helpers.ml.segformer_semantic_segmentation.datasets.create_segformer_segmentation_dataset

    Create a SegFormer segmentation dataset from a DataFrame.

    Args:
        df (pd.DataFrame | str): A pandas DataFrame with columns "fp" and "mask_fp" or a
            path to a CSV file.
        low_memory (bool): Whether to read the CSV file in low memory mode.
        transforms: A function that takes in a batch of samples (dictionary with
            pixel_values and label keys) and returns a transformed batch.

    Returns:
        A Dataset object to be used for HuggingFaces SegFormer model training.

    """
    from .SegFormerSegmentationDataset import SegFormerSegmentationDataset

    print(Fore.RED)
    print(
        "This is deprecated, please import from dsa_helpers.ml.segformer_semantic_segmentation.datasets"
    )
    print(Style.RESET_ALL)
    if isinstance(df, str):
        df = pd.read_csv(df, low_memory=low_memory)

    dataset = SegFormerSegmentationDataset(df)

    dataset = datasets.Dataset.from_generator(
        generator=lambda: dataset_generator(dataset),
        features=datasets.Features(
            {"pixel_values": datasets.Image(), "label": datasets.Image()}
        ),  # Example shape
    )

    if transforms is not None:
        dataset.set_transform(transforms)

    return dataset
