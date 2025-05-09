from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


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
