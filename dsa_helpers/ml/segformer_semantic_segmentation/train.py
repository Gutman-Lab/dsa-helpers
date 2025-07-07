import pandas as pd
from transformers import (
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
import numpy as np
from pathlib import Path

import torch
from torch import nn

from .utils import create_segformer_segmentation_dataset
from .transforms import get_train_transforms, val_transforms
from .evaluate import per_class_dice_on_dataset
from ..callbacks import MetricsLoggerCallback


def train(
    save_dir: str,
    train_data: pd.DataFrame | str,
    val_data: pd.DataFrame | str,
    label2id: dict[str, int],
    device: torch.device | None = None,
    test_data: pd.DataFrame | str | None = None,
    model_checkpoint: str | None = None,
    learning_rate: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 16,
    eval_accumulation_steps: int = 100,
    max_val_size: int | None = 500,
    random_state: int = 42,
    tile_size: int = 512,
    tqdm_notebook: bool = False,
    square_symmetry_prob: float | None = 1.0,
    rotate_limit: int = 45,
    rotate_prob: float | None = 0.5,
    fill: tuple[float, ...] | float = 255.0,
    fill_mask: tuple[float, ...] | float = 0.0,
    brightness: float = 0.25,
    contrast: float = 0.25,
    saturation: float = 0.25,
    hue: float = 0.1,
):
    """Workflow for training a SegFormer semantic segmentation model.

    Args:
        save_dir (str): Directory where model checkpoint directory will
            be saved to.
        train_csv_fp (str): Train data, should be a pandas
            dataframe or a path to a csv file containing the train data.
            Must have the "fp" and "mask_fp" columns, which are the
            filepaths to the images and masks, respectively.
        val_data (pd.DataFrame | str): See train_data, but for the
            validation data.
        label2id (dict): A dictionary mapping the label names to their
            corresponding integer ids.
        device (torch.device | None, optional): Device to run the model
            on. If None, will use "cuda" if available, otherwise "cpu".
            Defaults to None.
        test_data (pd.DataFrame | str | None, optional): See train_data,
            but for the test data. Defaults to None.
        model_checkpoint (str | None, optional): A local path to a model
            checkpoint to load the model from. If None it will use the
            default weights from huggingface: "nvidia/mit-b0". Defaults
            to None.
        learning_rate (float, optional): Learning rate for the model.
            Defaults to 1e-4.
        epochs (int, optional): Number of epochs to train the model.
            Defaults to 10.
        batch_size (int, optional): Batch size for the model.
            Defaults to 16.
        eval_accumulation_steps (int, optional): Number of steps to
            accumulate the validation metrics over. Defaults to 100.
        max_val_size (int | None, optional): Maximum number of
            validation samples to use. If None, all validation samples
            will be used. Defaults to None. Note that there is a bug
            that this value must be kept short otherwise a cuda memory
            error will occur.
        random_state (int, optional): Random state for the validation
            split. Defaults to 42.
        tile_size (int, optional): Size of the tiles to use for the model.
            Defaults to 512.
        tqdm_notebook (bool, optional): Whether to use a tqdm notebook.
            Defaults to False.
        square_symmetry_prob (float | None, optional): Probability of
            applying square symmetry to the images. Defaults to 1.0.
        rotate_limit (int, optional): Maximum number of degrees to
            rotate the images. Defaults to 45.
        rotate_prob (float | None, optional): Probability of applying
            rotation to the images. Defaults to 0.5.
        fill (tuple[float, ...] | float, optional): Value to fill the
            image with. Defaults to 255.0.
        fill_mask (tuple[float, ...] | float, optional): Value to fill
            the mask with. Defaults to 0.0.
        brightness (float, optional): Brightness factor to apply to the
            images. Defaults to 0.25.
        contrast (float, optional): Contrast factor to apply to the
            images. Defaults to 0.25.
        saturation (float, optional): Saturation factor to apply to the
            images. Defaults to 0.25.
        hue (float, optional): Hue factor to apply to the images.
            Defaults to 0.1.

    Returns:
        tuple: A tuple containing the trainer and the results dictionary.

    """
    save_dir_path = Path(save_dir)

    if save_dir_path.is_dir():
        raise FileExistsError(f"Save directory {save_dir} already exists.")

    # Read the dataframes if they are strings.
    if isinstance(train_data, str):
        train_data = pd.read_csv(train_data)
    if isinstance(val_data, str):
        val_data = pd.read_csv(val_data)
    if test_data is not None and isinstance(test_data, str):
        test_data = pd.read_csv(test_data)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the default pretrained model.
    label2id = {k: int(v) for k, v in label2id.items()}
    id2label = {int(v): k for k, v in label2id.items()}

    if model_checkpoint is None:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", id2label=id2label, label2id=label2id
        ).to(device)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_checkpoint,
            id2label=id2label,
            label2id=label2id,
            local_files_only=True,
            ignore_mismatched_sizes=True,
        ).to(device)

    # Training arguments.
    training_args = TrainingArguments(
        str(save_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=eval_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    # Create datasets for the model.
    train_transforms = get_train_transforms(
        square_symmetry_prob=square_symmetry_prob,
        rotate_limit=rotate_limit,
        rotate_prob=rotate_prob,
        fill=fill,
        fill_mask=fill_mask,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    train_dataset = create_segformer_segmentation_dataset(
        train_data, transforms=train_transforms
    )

    if val_data is not None and max_val_size < len(val_data):
        val_dataset = create_segformer_segmentation_dataset(
            val_data.sample(n=max_val_size, random_state=random_state),
            transforms=val_transforms,
        )
    else:
        val_dataset = create_segformer_segmentation_dataset(
            val_data, transforms=val_transforms
        )

    def compute_metrics(eval_pred):
        """Function for computing DICE coefficient metrics for each
        class, as well as the mean DICE and weighted mean DICE (maybe).
        """
        logits, labels = eval_pred

        # Convert logits to tensor.
        logits = torch.from_numpy(logits)

        # Scale the logits back to the shape of the labels.
        logits = nn.functional.interpolate(
            logits,
            size=tile_size,
            mode="bilinear",
            align_corners=False,
        )

        # Turn the logits to predictions.
        pred = logits.argmax(dim=1)

        # Calculate the Dice for each class.
        intersection = {label: 0 for label in label2id}
        denominator = {label: 0 for label in label2id}

        # For easy calculations, convert pred to a numpy.
        pred = pred.numpy()

        # Flatten both arrays.
        pred = pred.flatten()
        labels = labels.flatten()

        # Loop through each index.
        for label, integer in label2id.items():
            pred_mask = pred == integer
            gt_mask = labels == integer

            # Calculate the interesection.
            intersection[label] += np.sum(gt_mask & pred_mask)
            denominator[label] += np.sum(gt_mask) + np.sum(pred_mask)

        # Calculate the Dice for each label.
        metrics = {}
        for label in label2id:
            denominator_value = denominator[label]

            if denominator_value:
                metrics[f"{label}_dice"] = float(
                    2 * intersection[label] / denominator_value
                )
            else:
                metrics[f"{label}_dice"] = 1

        # Add the mean Dice.
        metrics["mean_dice"] = float(np.mean(list(metrics.values())))
        return metrics

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MetricsLoggerCallback],  # Add the custom callback here
    )

    # Train the model.
    _ = trainer.train()

    # Rename the checkpoint to not include a step number.
    save_dir = Path(save_dir)
    for p in save_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-"):
            p.rename(save_dir / "checkpoint")

    results = {"train": {}, "val": {}}

    for log in trainer.state.log_history[:-1]:
        epoch = log["epoch"]

        if epoch % 1:
            continue  # only track end of each epoch

        # Check if this is tracking train or validation.
        if "eval_loss" in log:
            data = results["val"]
        else:
            data = results["train"]

        for k, v in log.items():
            if k.startswith("eval_"):
                k = k[5:]

            if k not in data:
                data[k] = []

            data[k].append(v)

    # For each of the datasets provided, get the metrics.
    model = trainer.model

    print("Calculating train metrics...")
    results["train"]["metrics"] = per_class_dice_on_dataset(
        model,
        train_data,
        label2id,
        batch_size=batch_size,
        device=device,
        tile_size=tile_size,
        tqdm_notebook=tqdm_notebook,
    )

    print("Calculating val metrics...")
    results["val"]["metrics"] = per_class_dice_on_dataset(
        model,
        val_data,
        label2id,
        batch_size=batch_size,
        device=device,
        tile_size=tile_size,
        tqdm_notebook=tqdm_notebook,
    )

    if test_data is not None:
        print("Calculating test metrics...")
        results["test"] = {}

        results["test"]["metrics"] = per_class_dice_on_dataset(
            model,
            test_data,
            label2id,
            batch_size=batch_size,
            device=device,
            tile_size=tile_size,
            tqdm_notebook=tqdm_notebook,
        )

    return trainer, results
