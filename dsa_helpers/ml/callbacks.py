"""Provides custom callbacks for the Hugging Face Trainer class."""

from transformers import TrainerCallback
from pathlib import Path
import pandas as pd


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
