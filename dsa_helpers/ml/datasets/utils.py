# Utility functions, usualy for creating Datasets in specific formats.
import pandas as pd
from .SegFormerSegmentationDataset import SegFormerSegmentationDataset
from datasets import Dataset, Features, Image
from colorama import Style, Fore


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
    print(Fore.RED)
    print(
        "This is deprecated, please import from dsa_helpers.ml.segformer_semantic_segmentation.datasets"
    )
    print(Style.RESET_ALL)
    if isinstance(df, str):
        df = pd.read_csv(df, low_memory=low_memory)

    dataset = SegFormerSegmentationDataset(df)

    dataset = Dataset.from_generator(
        generator=lambda: dataset_generator(dataset),
        features=Features(
            {"pixel_values": Image(), "label": Image()}
        ),  # Example shape
    )

    if transforms is not None:
        dataset.set_transform(transforms)

    return dataset
