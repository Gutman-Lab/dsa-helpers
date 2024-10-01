from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np


class SegFormerSegmentationDataset(Dataset):
    """PyTorch dataset class for semantic segmentation using HuggingFaces SegFormer
    model.

    """

    def __init__(self, df: pd.DataFrame):
        """Initiate an instance of the segmentation dataset.

        Args:
            df (pandas.DataFrame): A pandas DataFrame with columns "fp" and "mask_fp".

        """
        self.image_files = df["fp"].tolist()
        self.mask_files = df["mask_fp"].tolist()
        self.df = df  # store potential metadata

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get the filepath to the image and its mask.
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # Return image & mask as PIL images in a dictionary.
        return {"pixel_values": Image.open(image_path), "label": Image.open(mask_path)}
