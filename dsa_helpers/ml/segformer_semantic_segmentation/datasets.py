from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from datasets import Dataset, Features, Image, Value


class SegFormerSegmentationDataset(Dataset):
    """PyTorch dataset class for semantic segmentation using
    HuggingFaces SegFormer model.

    """

    def __init__(
        self, df: pd.DataFrame, extras: bool = False, group: str = "wsi_name"
    ):
        """Initiate an instance of the segmentation dataset.

        Args:
            df (pandas.DataFrame): Must have columns "fp" and "mask_fp".
            extras (bool, optional): Whether to include additional
                columns "x", "y" and "group" in the dataset. Default
                is False.
            group (str | None, optional): Additional column to add when
                extras is True. Default is "wsi_name".

        """
        self.image_files = df["fp"].tolist()
        self.mask_files = df["mask_fp"].tolist()
        self.extras = extras

        if extras:
            self.x = df["x"].tolist()
            self.y = df["y"].tolist()
            self.group = df[group].tolist()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the filepath to the image and its mask.
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        if self.extras:
            return {
                "pixel_values": Image.open(image_path),
                "label": Image.open(mask_path),
                "x": self.x[idx],
                "y": self.y[idx],
                "group": self.group[idx],
            }
        else:
            return {
                "pixel_values": Image.open(image_path),
                "label": Image.open(mask_path),
            }


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
    if isinstance(df, str):
        df = pd.read_csv(df, low_memory=low_memory)

    dataset = SegFormerSegmentationDataset(df, group=group, extras=extras)

    if extras:
        features = Features(
            {
                "pixel_values": Image(),
                "label": Image(),
                "x": Value("int32"),
                "y": Value("int32"),
                "group": Value("string"),
            }
        )
    else:
        features = Features(
            {
                "pixel_values": Image(),
                "label": Image(),
            }
        )
    dataset = Dataset.from_generator(
        generator=lambda: dataset_generator(dataset), features=features
    )

    if transforms is not None:
        dataset.set_transform(transforms)

    return dataset
