"""
Model evaluation script.

Evaluates trained TCN model on test set and computes comprehensive metrics
including MSE, RMSE, MAE, R², and directional accuracy for each prediction horizon.
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.logging_config import get_logger, setup_logging
from config.settings import settings
from src.models.dataset import LOBDataset, create_train_val_test_split, load_data_from_db
from src.models.tcn import LOBPricePredictionTCN

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation.

    Computes multiple metrics across all prediction horizons and generates
    detailed performance reports.

    Args:
        model (nn.Module): Trained model to evaluate.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to run evaluation on.
        horizons (list): List of prediction horizon names.

    Attributes:
        predictions (np.ndarray): All model predictions.
        targets (np.ndarray): All ground truth targets.
        metrics (dict): Computed evaluation metrics.

    Example:
        >>> evaluator = ModelEvaluator(model, test_loader, device)
        >>> metrics = evaluator.evaluate()
        >>> evaluator.print_report()
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        horizons: list[str] = ["1s", "5s", "10s", "30s", "60s"],
    ):
        """Initialize evaluator."""
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.horizons = horizons

        self.predictions = None
        self.targets = None
        self.metrics = None

    def evaluate(self) -> Dict:
        """
        Run evaluation on test set.

        Returns:
            Dict: Evaluation metrics for all horizons.
        """
        logger.info("Starting model evaluation...")

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Evaluating"):
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        # Concatenate all batches
        self.predictions = np.concatenate(all_predictions, axis=0)
        self.targets = np.concatenate(all_targets, axis=0)

        logger.info(f"Evaluated {len(self.predictions)} samples")

        # Compute metrics
        self.metrics = self._compute_metrics()

        return self.metrics

    def _compute_metrics(self) -> Dict:
        """
        Compute all evaluation metrics.

        Returns:
            Dict: Metrics organized by horizon.
        """
        metrics = {}

        num_horizons = self.predictions.shape[1]

        for i, horizon in enumerate(self.horizons[:num_horizons]):
            y_true = self.targets[:, i]
            y_pred = self.predictions[:, i]

            # Regression metrics
            mse = np.mean((y_pred - y_true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_pred - y_true))

            # R² (coefficient of determination)
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Directional accuracy
            direction_true = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            directional_accuracy = np.mean(direction_true == direction_pred)

            # Store metrics
            metrics[horizon] = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "directional_accuracy": float(directional_accuracy),
            }

        # Overall directional accuracy
        all_directions_true = np.sign(self.targets)
        all_directions_pred = np.sign(self.predictions)
        overall_accuracy = np.mean(all_directions_true == all_directions_pred)
        metrics["overall"] = {
            "directional_accuracy": float(overall_accuracy),
        }

        return metrics

    def print_report(self) -> None:
        """Print formatted evaluation report."""
        if self.metrics is None:
            logger.error("No metrics available. Run evaluate() first.")
            return

        logger.info("=" * 70)
        logger.info("Model Evaluation Results")
        logger.info("=" * 70)
        logger.info(f"Test Samples: {len(self.predictions):,}")
        logger.info("")
        logger.info("Metrics by Horizon:")
        logger.info("")

        for horizon in self.horizons:
            if horizon not in self.metrics:
                continue

            m = self.metrics[horizon]
            logger.info(f"Horizon: {horizon}")
            logger.info(f"  MSE: {m['mse']:.4f}")
            logger.info(f"  RMSE: {m['rmse']:.4f}")
            logger.info(f"  MAE: {m['mae']:.4f}")
            logger.info(f"  R²: {m['r2']:.4f}")
            logger.info(f"  Directional Accuracy: {m['directional_accuracy']*100:.1f}%")
            logger.info("")

        overall_acc = self.metrics["overall"]["directional_accuracy"]
        logger.info(f"Overall Directional Accuracy: {overall_acc*100:.1f}%")
        logger.info("=" * 70)

    def save_predictions(self, output_path: Path) -> None:
        """
        Save predictions to CSV file.

        Args:
            output_path (Path): Path to save predictions.
        """
        import pandas as pd

        # Create DataFrame
        data = {}
        for i, horizon in enumerate(self.horizons):
            if i >= self.predictions.shape[1]:
                break
            data[f"actual_{horizon}"] = self.targets[:, i]
            data[f"pred_{horizon}"] = self.predictions[:, i]
            data[f"error_{horizon}"] = self.predictions[:, i] - self.targets[:, i]

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        logger.info(f"Predictions saved to {output_path}")

    def plot_results(self, output_dir: Path) -> None:
        """
        Generate evaluation plots.

        Args:
            output_dir (Path): Directory to save plots.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
            output_dir.mkdir(parents=True, exist_ok=True)

            # 1. Prediction vs Actual scatter plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, horizon in enumerate(self.horizons):
                if i >= len(axes):
                    break

                ax = axes[i]
                y_true = self.targets[:, i]
                y_pred = self.predictions[:, i]

                # Scatter plot
                ax.scatter(y_true, y_pred, alpha=0.3, s=1)

                # Perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

                ax.set_xlabel("Actual Return (%)")
                ax.set_ylabel("Predicted Return (%)")
                ax.set_title(f"Horizon: {horizon}")

                # Add R² to plot
                r2 = self.metrics[horizon]["r2"]
                ax.text(
                    0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes, verticalalignment="top"
                )

            # Hide unused subplot
            if len(self.horizons) < len(axes):
                axes[-1].axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / "predictions_scatter.png", dpi=150)
            plt.close()

            # 2. Error distribution histograms
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for i, horizon in enumerate(self.horizons):
                if i >= len(axes):
                    break

                ax = axes[i]
                errors = self.predictions[:, i] - self.targets[:, i]

                ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
                ax.axvline(0, color="r", linestyle="--", linewidth=2)
                ax.set_xlabel("Prediction Error (%)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"Error Distribution: {horizon}")

                # Add MAE to plot
                mae = self.metrics[horizon]["mae"]
                ax.text(
                    0.05, 0.95, f"MAE = {mae:.4f}", transform=ax.transAxes, verticalalignment="top"
                )

            if len(self.horizons) < len(axes):
                axes[-1].axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / "error_distributions.png", dpi=150)
            plt.close()

            # 3. Directional accuracy bar chart
            fig, ax = plt.subplots(figsize=(10, 6))

            horizons_list = []
            accuracies = []
            for horizon in self.horizons:
                if horizon in self.metrics:
                    horizons_list.append(horizon)
                    accuracies.append(self.metrics[horizon]["directional_accuracy"] * 100)

            bars = ax.bar(horizons_list, accuracies, alpha=0.7, edgecolor="black")
            ax.axhline(50, color="r", linestyle="--", linewidth=2, label="Random (50%)")
            ax.set_xlabel("Prediction Horizon")
            ax.set_ylabel("Directional Accuracy (%)")
            ax.set_title("Directional Accuracy by Horizon")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{acc:.1f}%",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plt.savefig(output_dir / "directional_accuracy.png", dpi=150)
            plt.close()

            logger.info(f"Plots saved to {output_dir}")

        except ImportError:
            logger.warning("matplotlib not installed, skipping plots")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")


# =============================================================================
# Main Evaluation Function
# =============================================================================


async def main(args):
    """Main evaluation function."""
    setup_logging(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("LOB Price Prediction - Model Evaluation")
    logger.info("=" * 70)

    # Set device
    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model}")

    checkpoint = torch.load(args.model, map_location=device)

    model = LOBPricePredictionTCN(
        input_size=43,
        num_channels=[128, 128, 256, 256],
        kernel_size=3,
        dropout=0.2,
        num_horizons=5,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

    # Load data
    if args.synthetic:
        logger.info("Using synthetic data for testing")
        num_samples = 10000
        num_features = 43
        num_horizons = 5

        features = np.random.randn(num_samples, num_features).astype(np.float32)
        targets = np.random.randn(num_samples, num_horizons).astype(np.float32)

        # Use last 15% as test
        split_idx = int(num_samples * 0.85)
        X_test = features[split_idx:]
        y_test = targets[split_idx:]
    else:
        logger.info("Loading data from TimescaleDB...")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=args.days)

        db_config = {
            "host": settings.db_host,
            "port": settings.db_port,
            "database": settings.db_name,
            "user": settings.db_user,
            "password": settings.db_password,
        }

        features, targets = await load_data_from_db(
            symbol=args.symbol,
            start_time=start_time,
            end_time=end_time,
            db_config=db_config,
        )

        # Split (use test set only)
        _, _, (X_test, y_test) = create_train_val_test_split(features, targets)

    logger.info(f"Test data: {len(X_test)} samples")

    # Create test dataset (need normalization params from training)
    # For now, normalize on test data itself (in production, load from checkpoint)
    test_dataset = LOBDataset(
        X_test,
        y_test,
        sequence_length=args.sequence_length,
        normalize=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Evaluate
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        horizons=["1s", "5s", "10s", "30s", "60s"],
    )

    metrics = evaluator.evaluate()
    evaluator.print_report()

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        evaluator.save_predictions(output_dir / "predictions.csv")

        # Save metrics
        import json

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {output_dir / 'metrics.json'}")

        # Generate plots
        if args.plot:
            evaluator.plot_results(output_dir)

    logger.info("Evaluation complete!")


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained TCN model")

    # Model args
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")

    # Data args
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--sequence-length", type=int, default=100, help="Input sequence length")

    # Evaluation args
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    # Other
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    asyncio.run(main(args))
