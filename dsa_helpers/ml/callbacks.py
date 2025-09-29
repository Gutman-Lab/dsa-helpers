"""Provides custom callbacks for the Hugging Face Trainer class."""

from transformers import TrainerCallback
from pathlib import Path
import pandas as pd
from copy import deepcopy
import os, shutil


class MetricsLoggerCallback(TrainerCallback):
    def _load_current_file(self, output_dir):
        output_dir = Path(output_dir)
        metrics_file = output_dir / "metrics.csv"

        return (
            pd.read_csv(metrics_file)
            if metrics_file.exists()
            else pd.DataFrame()
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Triggered during training logging (e.g., for training loss).
        """
        if "loss" in logs:  # Ensure training loss is available
            metrics_df = self._load_current_file(args.output_dir)

            # Append the epoch and the training loss.
            loss = logs["loss"]
            epoch = state.epoch

            metrics_df = pd.concat(
                [
                    metrics_df,
                    pd.DataFrame([{"epoch": epoch, "train_loss": loss}]),
                ],
                ignore_index=True,
            )

            # Save it back to file.
            metrics_df.to_csv(
                Path(args.output_dir) / "metrics.csv", index=False
            )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        This is triggered at the end of evaluation (including at the end of each epoch).
        """
        # Get the current metrics file.
        metrics_df = self._load_current_file(args.output_dir)

        # Add metrics to the last row of the current metrics file.
        for k, v in metrics.items():
            metrics_df.at[metrics_df.index[-1], k] = v

        # Save the updated metrics file.
        metrics_df.to_csv(Path(args.output_dir) / "metrics.csv", index=False)


class BestModelCallback(TrainerCallback):
    """Callback for tracking and saving the best model checkpoint.

    This class was developed with the help of ChatGPT."""

    def __init__(self, save_dir: str):
        """
        Initialize the callback.

        Args:
            save_dir (str): Directory where model is saving checkpoints.
                The best checkpoint will be saved in a subdirectory
                called "best-checkpoint".

        """
        self.best_loss = float("inf")
        self.best_model_state = None
        self.save_dir = save_dir

    def on_evaluate(self, args, state, control, metrics, model, **kwargs):
        """Called after evaluation. Track the best model state."""
        # Get the mean_dice metric
        current_metric = metrics.get("eval_loss")

        if current_metric is not None and current_metric < self.best_loss:
            self.best_loss = current_metric
            print("New best loss.")

            # Store the model state.
            self.best_model_state = deepcopy(model.state_dict())

    def on_train_end(self, args, state, control, model, **kwargs):
        """Logic:
        * Create a best-checkpoint directory
        * Save the best model state into the best-checkpoint directory
        * Copy all the other model files (from the normal checkpoint
          directory) into the best-checkpoint directory

        """
        if self.best_model_state is not None:
            # Create best-checkpoint directory
            best_checkpoint_dir = os.path.join(
                self.save_dir, "best-checkpoint"
            )
            os.makedirs(best_checkpoint_dir, exist_ok=True)

            # Load the best model state
            model.load_state_dict(self.best_model_state)

            # Save the model (this creates model.safetensors or pytorch_model.bin)
            model.save_pretrained(best_checkpoint_dir)

            # Copy other checkpoint files from the latest checkpoint
            # Find the latest checkpoint directory (checkpoint-{step_number})
            save_path = Path(self.save_dir)
            checkpoint_dirs = [
                d
                for d in save_path.iterdir()
                if d.is_dir() and d.name.startswith("checkpoint-")
            ]
            if checkpoint_dirs:
                # Sort by step number to get the latest
                latest_checkpoint = max(
                    checkpoint_dirs, key=lambda x: int(x.name.split("-")[1])
                )

                # Copy all files from the latest checkpoint, except the model files we just created
                for file_path in latest_checkpoint.iterdir():
                    if file_path.is_file():
                        # Skip model files since we already saved the best model state
                        if not file_path.name.startswith(
                            ("model.", "pytorch_model.")
                        ):
                            dst = Path(best_checkpoint_dir) / file_path.name
                            shutil.copy2(file_path, dst)
