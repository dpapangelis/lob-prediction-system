"""
Real-time inference engine with database persistence.

Loads trained model and makes predictions on live LOB data, saving predictions
to the database for later evaluation against actual outcomes.
"""

import argparse
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import asyncpg
import numpy as np
import torch

from config.logging_config import get_logger, setup_logging
from config.settings import settings
from src.data.binance_stream import BinanceDepthStream
from src.data.feature_engineering import LOBFeatureEngineering
from src.models.tcn import LOBPricePredictionTCN

logger = get_logger(__name__)


class LOBPredictor:
    """
    Real-time LOB price predictor with database persistence.

    Maintains a rolling window of LOB features and generates predictions
    for multiple time horizons. Optionally saves predictions to database
    for later evaluation.

    Args:
        model_path (Path): Path to trained model checkpoint.
        sequence_length (int): Input sequence length (must match training).
        device (str): Device to run inference on ('cpu', 'mps', 'cuda').
        normalization_params (tuple, optional): (mean, std) for feature normalization.
        model_version (str): Version identifier for tracking.
        save_to_db (bool): Whether to save predictions to database.
        db_pool (asyncpg.Pool, optional): Database connection pool.

    Attributes:
        model (nn.Module): Loaded TCN model.
        feature_buffer (list): Rolling buffer of feature vectors.
        predictions_history (list): History of predictions made.

    Example:
        >>> predictor = LOBPredictor('models/best_model.pth')
        >>>
        >>> # On each LOB update:
        >>> predictions = await predictor.update(lob_snapshot, 'BTCUSDT')
        >>> print(f"Predicted returns: {predictions}")
    """

    def __init__(
        self,
        model_path: Path,
        sequence_length: int = 100,
        device: str = "mps",
        normalization_params: Optional[tuple] = None,
        model_version: str = "v1.0",
        save_to_db: bool = True,
        db_pool: Optional[asyncpg.Pool] = None,
    ):
        """Initialize predictor."""
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        self.model_version = model_version
        self.save_to_db = save_to_db
        self.db_pool = db_pool

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

        logger.info(f"Predictor initialized (device={device}, save_to_db={save_to_db})")

    async def _save_prediction_to_db(
        self,
        timestamp: datetime,
        symbol: str,
        predictions: dict,
        mid_price: float,
        spread_bps: float,
        volume_imbalance: float,
        inference_time_ms: float,
    ) -> None:
        """
        Save prediction to database.

        Args:
            timestamp: Prediction timestamp
            symbol: Trading symbol
            predictions: Dict of predictions by horizon
            mid_price: Current mid price
            spread_bps: Spread in basis points
            volume_imbalance: Volume imbalance ratio
            inference_time_ms: Inference time in milliseconds
        """
        if not self.save_to_db or self.db_pool is None:
            return

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO predictions (
                        time, symbol, model_version,
                        pred_1s, pred_5s, pred_10s, pred_30s, pred_60s,
                        mid_price, spread_bps, volume_imbalance,
                        inference_time_ms
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (time, symbol, model_version) DO NOTHING
                    """,
                    timestamp,
                    symbol,
                    self.model_version,
                    predictions["1s"],
                    predictions["5s"],
                    predictions["10s"],
                    predictions["30s"],
                    predictions["60s"],
                    mid_price,
                    spread_bps,
                    volume_imbalance,
                    inference_time_ms,
                )
                logger.debug(f"Saved prediction to database at {timestamp}")
        except Exception as e:
            logger.error(f"Error saving prediction to database: {e}")

    async def update(self, lob_snapshot: dict, symbol: str = "BTCUSDT") -> Optional[dict]:
        """
        Update with new LOB snapshot and make prediction if ready.

        Args:
            lob_snapshot (dict): LOB snapshot from exchange.
            symbol (str): Trading symbol.

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
            prediction = await self._predict(lob_snapshot, symbol)
            return prediction

        return None

    async def _predict(self, lob_snapshot: dict, symbol: str) -> dict:
        """
        Make prediction on current buffer.

        Args:
            lob_snapshot (dict): Current LOB snapshot.
            symbol (str): Trading symbol.

        Returns:
            dict: Predictions for all horizons.
        """
        # Start timing
        start_time = time.time()

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

        # Inference time
        inference_time_ms = (time.time() - start_time) * 1000

        # Get context from LOB (convert strings to float)
        best_bid = float(lob_snapshot["bids"][0][0])
        best_ask = float(lob_snapshot["asks"][0][0])
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000

        total_bid_volume = sum(float(bid[1]) for bid in lob_snapshot["bids"])
        total_ask_volume = sum(float(ask[1]) for ask in lob_snapshot["asks"])
        volume_imbalance = (total_bid_volume - total_ask_volume) / (
            total_bid_volume + total_ask_volume
        )

        # Format output
        timestamp = datetime.now(timezone.utc)
        result = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "mid_price": float(mid_price),
            "spread_bps": float(spread_bps),
            "volume_imbalance": float(volume_imbalance),
            "inference_time_ms": float(inference_time_ms),
            "predictions": {
                "1s": float(predictions[0]),
                "5s": float(predictions[1]),
                "10s": float(predictions[2]),
                "30s": float(predictions[3]),
                "60s": float(predictions[4]),
            },
        }

        # Save to database
        if self.save_to_db:
            await self._save_prediction_to_db(
                timestamp=timestamp,
                symbol=symbol,
                predictions=result["predictions"],
                mid_price=mid_price,
                spread_bps=spread_bps,
                volume_imbalance=volume_imbalance,
                inference_time_ms=inference_time_ms,
            )

        self.predictions_made += 1
        self.predictions_history.append(result)

        # Keep only last 1000 predictions in memory
        if len(self.predictions_history) > 1000:
            self.predictions_history.pop(0)

        return result

    def get_statistics(self) -> dict:
        """Get predictor statistics."""
        return {
            "predictions_made": self.predictions_made,
            "buffer_size": len(self.feature_buffer),
            "buffer_ready": len(self.feature_buffer) == self.sequence_length,
            "avg_inference_time_ms": (
                np.mean([p["inference_time_ms"] for p in self.predictions_history[-100:]])
                if self.predictions_history
                else 0
            ),
        }


class LivePredictionPipeline:
    """
    End-to-end live prediction pipeline with database persistence.

    Connects to Binance WebSocket, processes LOB updates, generates
    real-time predictions, and saves them to the database.

    Args:
        model_path (Path): Path to trained model.
        symbol (str): Trading symbol.
        model_version (str): Model version identifier.
        callback (callable, optional): Callback for predictions.
        save_to_db (bool): Whether to save predictions to database.

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
        model_version: str = "v1.0",
        callback=None,
        save_to_db: bool = True,
    ):
        """Initialize pipeline."""
        self.symbol = symbol
        self.model_version = model_version
        self.callback = callback
        self.save_to_db = save_to_db
        self.db_pool = None

        # Create predictor (without db_pool initially)
        self.predictor = LOBPredictor(
            model_path,
            model_version=model_version,
            save_to_db=save_to_db,
            db_pool=None,  # Will set after creating pool
        )

        # Create WebSocket stream
        self.stream = BinanceDepthStream(
            symbol=symbol,
            callback=self._process_lob_update,
            levels=5,
            use_testnet=False,
        )

        logger.info(f"Live prediction pipeline initialized for {symbol}")

    async def _init_db(self) -> None:
        """Initialize database connection pool."""
        if not self.save_to_db:
            return

        try:
            self.db_pool = await asyncpg.create_pool(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                min_size=1,
                max_size=5,
            )
            self.predictor.db_pool = self.db_pool
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.save_to_db = False

    async def _process_lob_update(self, lob_data: dict) -> None:
        """Process LOB update and make prediction."""
        try:
            # Update predictor
            prediction = await self.predictor.update(lob_data, self.symbol)

            # If prediction ready
            if prediction:
                # Log prediction
                preds = prediction["predictions"]
                logger.info(
                    f"Predictions: 1s={preds['1s']:+.4f}%, "
                    f"5s={preds['5s']:+.4f}%, "
                    f"10s={preds['10s']:+.4f}%, "
                    f"30s={preds['30s']:+.4f}%, "
                    f"60s={preds['60s']:+.4f}% "
                    f"[{prediction['inference_time_ms']:.1f}ms]"
                )

                # Call callback if provided
                if self.callback:
                    await self.callback(prediction)

        except Exception as e:
            logger.error(f"Error processing LOB update: {e}", exc_info=True)

    async def start(self) -> None:
        """Start live prediction pipeline."""
        logger.info("Starting live prediction pipeline...")

        # Initialize database
        if self.save_to_db:
            await self._init_db()

        # Start stream
        await self.stream.start()

    async def stop(self) -> None:
        """Stop pipeline."""
        logger.info("Stopping live prediction pipeline...")

        # Stop stream
        await self.stream.stop()

        # Close database pool
        if self.db_pool:
            await self.db_pool.close()
            logger.info("Database connection pool closed")

        # Print statistics
        stats = self.predictor.get_statistics()
        logger.info(f"Total predictions made: {stats['predictions_made']}")
        logger.info(f"Average inference time: {stats['avg_inference_time_ms']:.1f}ms")


# =============================================================================
# Main
# =============================================================================


async def main(args):
    """Main inference function."""
    setup_logging(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("LOB Price Prediction - Live Inference with DB Persistence")
    logger.info("=" * 70)

    # Create pipeline
    pipeline = LivePredictionPipeline(
        model_path=Path(args.model),
        symbol=args.symbol,
        model_version=args.model_version,
        save_to_db=not args.no_save,
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
    parser = argparse.ArgumentParser(
        description="Real-time LOB price prediction with DB persistence"
    )

    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument(
        "--model-version", type=str, default="v1.0", help="Model version identifier"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save predictions to database"
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    asyncio.run(main(args))
