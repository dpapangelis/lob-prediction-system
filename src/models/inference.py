"""
Real-time inference engine.

Loads trained model and makes predictions on live or historical LOB data.
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from config.logging_config import get_logger, setup_logging

# from config.settings import settings
from src.data.binance_stream import BinanceDepthStream
from src.data.feature_engineering import LOBFeatureEngineering
from src.models.tcn import LOBPricePredictionTCN

logger = get_logger(__name__)


class LOBPredictor:
    """
    Real-time LOB price predictor.

    Maintains a rolling window of LOB features and generates predictions
    for multiple time horizons.

    Args:
        model_path (Path): Path to trained model checkpoint.
        sequence_length (int): Input sequence length (must match training).
        device (str): Device to run inference on ('cpu', 'mps', 'cuda').
        normalization_params (tuple, optional): (mean, std) for feature normalization.

    Attributes:
        model (nn.Module): Loaded TCN model.
        feature_buffer (list): Rolling buffer of feature vectors.
        predictions_history (list): History of predictions made.

    Example:
        >>> predictor = LOBPredictor('models/best_model.pth')
        >>>
        >>> # On each LOB update:
        >>> predictions = predictor.predict(lob_snapshot)
        >>> print(f"Predicted returns: {predictions}")
    """

    def __init__(
        self,
        model_path: Path,
        sequence_length: int = 100,
        device: str = "mps",
        normalization_params: Optional[tuple] = None,
    ):
        """Initialize predictor."""
        self.sequence_length = sequence_length
        self.device = torch.device(device)

        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = LOBPricePredictionTCN(
            input_size=43,
            num_channels=[128, 128, 256, 256],
            kernel_size=3,
            dropout=0.2,
            num_horizons=5,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded (epoch {checkpoint['epoch']})")

        # Normalization
        if normalization_params:
            self.mean, self.std = normalization_params
        else:
            logger.warning("No normalization params provided, using zeros/ones")
            self.mean = np.zeros(43)
            self.std = np.ones(43)

        # Feature engineering
        self.feature_engineer = LOBFeatureEngineering(levels=5, normalize=False)

        # Rolling buffer
        self.feature_buffer = []

        # Statistics
        self.predictions_made = 0
        self.predictions_history = []

        logger.info(f"Predictor initialized (device={device})")

    def update(self, lob_snapshot: dict) -> Optional[dict]:
        """
        Update with new LOB snapshot and make prediction if ready.

        Args:
            lob_snapshot (dict): LOB snapshot from exchange.

        Returns:
            dict: Predictions if buffer is full, None otherwise.
        """
        # Compute features
        features = self.feature_engineer.compute_features(lob_snapshot)

        # Add to buffer
        self.feature_buffer.append(features.flatten())

        # Keep only last sequence_length
        if len(self.feature_buffer) > self.sequence_length:
            self.feature_buffer.pop(0)

        # Make prediction if buffer full
        if len(self.feature_buffer) == self.sequence_length:
            return self._predict()

        return None

    def _predict(self) -> dict:
        """
        Make prediction on current buffer.

        Returns:
            dict: Predictions for all horizons.
        """
        # Stack features into sequence
        features = np.array(self.feature_buffer)  # (seq_len, features)

        # Normalize
        features = (features - self.mean) / self.std

        # Convert to tensor
        x = torch.from_numpy(features).float().unsqueeze(0)  # (1, seq_len, features)
        x = x.to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(x)  # (1, num_horizons)

        # Convert to numpy
        predictions = predictions.cpu().numpy().flatten()

        # Format output
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "predictions": {
                "1s": float(predictions[0]),
                "5s": float(predictions[1]),
                "10s": float(predictions[2]),
                "30s": float(predictions[3]),
                "60s": float(predictions[4]),
            },
        }

        self.predictions_made += 1
        self.predictions_history.append(result)

        # Keep only last 1000 predictions
        if len(self.predictions_history) > 1000:
            self.predictions_history.pop(0)

        return result

    def get_statistics(self) -> dict:
        """Get predictor statistics."""
        return {
            "predictions_made": self.predictions_made,
            "buffer_size": len(self.feature_buffer),
            "buffer_ready": len(self.feature_buffer) == self.sequence_length,
        }


class LivePredictionPipeline:
    """
    End-to-end live prediction pipeline.

    Connects to Binance WebSocket, processes LOB updates, and generates
    real-time predictions.

    Args:
        model_path (Path): Path to trained model.
        symbol (str): Trading symbol.
        callback (callable, optional): Callback for predictions.

    Example:
        >>> def handle_prediction(pred):
        ...     print(f"Prediction: {pred}")
        >>>
        >>> pipeline = LivePredictionPipeline(
        ...     model_path='models/best_model.pth',
        ...     symbol='BTCUSDT',
        ...     callback=handle_prediction
        ... )
        >>> await pipeline.start()
    """

    def __init__(
        self,
        model_path: Path,
        symbol: str = "BTCUSDT",
        callback=None,
    ):
        """Initialize pipeline."""
        self.symbol = symbol
        self.callback = callback

        # Create predictor
        self.predictor = LOBPredictor(model_path)

        # Create WebSocket stream
        self.stream = BinanceDepthStream(
            symbol=symbol,
            callback=self._process_lob_update,
            levels=5,
            use_testnet=False,
        )

        logger.info(f"Live prediction pipeline initialized for {symbol}")

    async def _process_lob_update(self, lob_data: dict) -> None:
        """Process LOB update and make prediction."""
        try:
            # Update predictor
            prediction = self.predictor.update(lob_data)

            # If prediction ready
            if prediction:
                # Log prediction
                preds = prediction["predictions"]
                logger.info(
                    f"Predictions: 1s={preds['1s']:+.4f}, "
                    f"5s={preds['5s']:+.4f}, "
                    f"10s={preds['10s']:+.4f}, "
                    f"30s={preds['30s']:+.4f}, "
                    f"60s={preds['60s']:+.4f}"
                )

                # Call callback if provided
                if self.callback:
                    await self.callback(prediction)

        except Exception as e:
            logger.error(f"Error processing LOB update: {e}", exc_info=True)

    async def start(self) -> None:
        """Start live prediction pipeline."""
        logger.info("Starting live prediction pipeline...")

        # Start stream
        await self.stream.start()

    async def stop(self) -> None:
        """Stop pipeline."""
        logger.info("Stopping live prediction pipeline...")

        # Stop stream
        await self.stream.stop()

        # Print statistics
        stats = self.predictor.get_statistics()
        logger.info(f"Total predictions made: {stats['predictions_made']}")


# =============================================================================
# Main
# =============================================================================


async def main(args):
    """Main inference function."""
    setup_logging(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("LOB Price Prediction - Live Inference")
    logger.info("=" * 70)

    # Create pipeline
    pipeline = LivePredictionPipeline(
        model_path=Path(args.model),
        symbol=args.symbol,
    )

    # Setup signal handlers for graceful shutdown
    import signal

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(pipeline.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await pipeline.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await pipeline.stop()

    logger.info("Inference pipeline shutdown complete")


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time LOB price prediction")

    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    asyncio.run(main(args))
