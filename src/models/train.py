"""
Training script for TCN model on LOB data.

Handles full training loop including:
- Data loading from TimescaleDB
- Model initialization and checkpointing
- Training with validation monitoring
- Learning rate scheduling
- Early stopping
- Metric logging
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

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


class Trainer:
    """
    Training manager for TCN model.

    Handles the complete training lifecycle including data loading,
    model training, validation, checkpointing, and early stopping.

    Args:
        model (nn.Module): TCN model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on (CPU/MPS/CUDA).
        checkpoint_dir (Path): Directory to save checkpoints.
        early_stopping_patience (int): Epochs to wait before early stopping.

    Attributes:
        best_val_loss (float): Best validation loss seen.
        epochs_no_improve (int): Epochs since last improvement.
        history (dict): Training history (losses, metrics).

    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     criterion=nn.MSELoss(),
        ...     optimizer=optimizer,
        ...     device=device,
        ...     checkpoint_dir=Path('checkpoints'),
        ... )
        >>> trainer.train(num_epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        checkpoint_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10,
    ):
        """Initialize trainer."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stopping_patience = early_stopping_patience

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

        # History tracking
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (x, y) in enumerate(pbar):
            # Move to device
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(x)
            loss = self.criterion(predictions, y)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> float:
        """
        Validate on validation set.

        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validating"):
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)
                loss = self.criterion(predictions, y)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(
        self,
        filename: str,
        is_best: bool = False,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            filename (str): Checkpoint filename.
            is_best (bool): Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✓ Saved best model to {best_path}")

        logger.debug(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: Path) -> None:
        """
        Load model checkpoint.

        Args:
            filepath (Path): Path to checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(
        self,
        num_epochs: int,
        save_every: int = 10,
    ) -> Dict:
        """
        Full training loop.

        Args:
            num_epochs (int): Number of epochs to train.
            save_every (int): Save checkpoint every N epochs.

        Returns:
            Dict: Training history.
        """
        logger.info("=" * 70)
        logger.info("Starting Training")
        logger.info("=" * 70)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 70)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Track history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # Update learning rate based on validation loss
            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(
                    f"checkpoint_epoch{epoch + 1}.pth",
                    is_best=True,
                )
            else:
                self.epochs_no_improve += 1

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch + 1}.pth")

            # Early stopping
            if self.epochs_no_improve >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"(no improvement for {self.early_stopping_patience} epochs)"
                )
                break

        logger.info("=" * 70)
        logger.info("Training Complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 70)

        return self.history


# =============================================================================
# Main Training Function
# =============================================================================


async def main(args):
    """Main training function."""
    setup_logging(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("LOB Price Prediction - TCN Training")
    logger.info("=" * 70)

    # Set device
    if torch.backends.mps.is_available() and not args.cpu:
        device = torch.device("mps")
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data from database (or use synthetic for testing)
    if args.synthetic:
        logger.info("Using synthetic data for testing")
        num_samples = 10000
        num_features = 43
        num_horizons = 5

        features = np.random.randn(num_samples, num_features).astype(np.float32)
        targets = np.random.randn(num_samples, num_horizons).astype(np.float32)
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

    logger.info(f"Data loaded: {len(features)} samples")

    # Train/val/test split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_train_val_test_split(
        features, targets
    )

    # Create datasets
    train_dataset = LOBDataset(
        X_train,
        y_train,
        sequence_length=args.sequence_length,
        normalize=True,
    )

    mean, std = train_dataset.get_normalization_params()

    val_dataset = LOBDataset(
        X_val,
        y_val,
        sequence_length=args.sequence_length,
        normalize=True,
        mean=mean,
        std=std,
    )

    # _test_dataset = LOBDataset(
    #     X_test,
    #     y_test,
    #     sequence_length=args.sequence_length,
    #     normalize=True,
    #     mean=mean,
    #     std=std,
    # )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for MPS compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    model = LOBPricePredictionTCN(
        input_size=43,
        num_channels=[128, 128, 256, 256],
        kernel_size=3,
        dropout=args.dropout,
        num_horizons=5,
    )

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Create trainer
    checkpoint_dir = Path(f"data/models/{args.symbol}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=args.patience,
    )

    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Train
    history = trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
    )

    # Save final model
    trainer.save_checkpoint("final_model.pth")

    # Plot training curves (optional)
    if args.plot:
        plot_training_curves(history, checkpoint_dir)

    logger.info("Training pipeline complete!")


def plot_training_curves(history: Dict, output_dir: Path) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history (Dict): Training history.
        output_dir (Path): Directory to save plots.
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Training Loss", linewidth=2)
        ax.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (MSE)", fontsize=12)
        ax.set_title("Training Progress", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plot_path = output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training curves to {plot_path}")

    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCN model on LOB data")

    # Data args
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--days", type=int, default=7, help="Days of historical data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    # Model args
    parser.add_argument("--sequence-length", type=int, default=100, help="Input sequence length")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Training args
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (L2 reg)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Checkpoint args
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--plot", action="store_true", help="Plot training curves")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    asyncio.run(main(args))
