"""
Unified production system: Live data collection + Real-time inference.

This combines data collection and inference into a single efficient process:
- Single WebSocket connection to Binance
- Immediate predictions on incoming data (no DB roundtrip)
- Parallel async saves to database
- Guaranteed timestamp alignment

Usage:
    python -m src.production.live_system --model data/models/BTCUSDT/best_model.pth
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


class UnifiedProductionSystem:
    """
    Unified production system combining data collection and inference.

    Optimized for minimum latency:
    - Single WebSocket connection
    - Immediate prediction on raw data
    - Parallel async database writes
    - No intermediate DB roundtrips

    Args:
        model_path (Path): Path to trained model checkpoint
        symbol (str): Trading symbol
        model_version (str): Model version identifier
        sequence_length (int): Input sequence length for model
        device (str): Device for inference ('mps', 'cuda', 'cpu')
        save_lob_data (bool): Whether to save LOB snapshots to database
        save_predictions (bool): Whether to save predictions to database

    Example:
        >>> system = UnifiedProductionSystem(
        ...     model_path=Path('models/best_model.pth'),
        ...     symbol='BTCUSDT',
        ... )
        >>> await system.start()
    """

    def __init__(
        self,
        model_path: Path,
        symbol: str = "BTCUSDT",
        model_version: str = "v1.0",
        sequence_length: int = 100,
        device: str = "mps",
        save_lob_data: bool = True,
        save_predictions: bool = True,
    ):
        """Initialize unified production system."""
        self.symbol = symbol
        self.model_version = model_version
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        self.save_lob_data = save_lob_data
        self.save_predictions = save_predictions

        # Database connection pool
        self.db_pool: Optional[asyncpg.Pool] = None

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

        # Normalization (TODO: save these in checkpoint)
        logger.warning("No normalization params in checkpoint, using zeros/ones")
        self.mean = np.zeros(43)
        self.std = np.ones(43)

        # Feature engineering
        self.feature_engineer = LOBFeatureEngineering(levels=5, normalize=False)

        # Rolling buffer for sequences
        self.feature_buffer = []

        # WebSocket stream
        self.stream = BinanceDepthStream(
            symbol=symbol,
            callback=self._process_lob_update,
            levels=5,
            use_testnet=False,
        )

        # Statistics
        self.total_updates = 0
        self.total_predictions = 0
        self.total_inference_time = 0
        self.total_save_time = 0

        logger.info(
            f"Unified production system initialized for {symbol} "
            f"(device={device}, save_lob={save_lob_data}, save_pred={save_predictions})"
        )

    async def _init_db(self) -> None:
        """Initialize database connection pool."""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                min_size=2,
                max_size=10,
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def _save_lob_data(
        self,
        timestamp: datetime,
        lob_snapshot: dict,
        features: np.ndarray,
    ) -> None:
        """
        Save LOB snapshot to database (async, non-blocking).

        Args:
            timestamp: Snapshot timestamp
            lob_snapshot: Raw LOB data from Binance
            features: Computed features (43-dimensional)
        """
        if not self.save_lob_data or self.db_pool is None:
            return

        try:
            # Extract prices and volumes
            bids = lob_snapshot["bids"]
            asks = lob_snapshot["asks"]

            # Convert to floats
            bid_prices = [float(bid[0]) for bid in bids[:5]]
            bid_volumes = [float(bid[1]) for bid in bids[:5]]
            ask_prices = [float(ask[0]) for ask in asks[:5]]
            ask_volumes = [float(ask[1]) for ask in asks[:5]]

            # Compute basic metrics
            mid_price = (bid_prices[0] + ask_prices[0]) / 2
            spread = ask_prices[0] - bid_prices[0]
            spread_bps = (spread / mid_price) * 10000

            total_bid_volume = sum(bid_volumes)
            total_ask_volume = sum(ask_volumes)
            volume_imbalance = (total_bid_volume - total_ask_volume) / (
                total_bid_volume + total_ask_volume
            )

            # Weighted mid price
            weighted_mid_price = (
                bid_prices[0] * ask_volumes[0] + ask_prices[0] * bid_volumes[0]
            ) / (bid_volumes[0] + ask_volumes[0])

            # Price range
            price_range = ask_prices[-1] - bid_prices[-1]

            # Depth imbalance (exponentially weighted)
            weights = [np.exp(-0.1 * i) for i in range(5)]
            weighted_bid_depth = sum(w * v for w, v in zip(weights, bid_volumes))
            weighted_ask_depth = sum(w * v for w, v in zip(weights, ask_volumes))
            depth_imbalance = (weighted_bid_depth - weighted_ask_depth) / (
                weighted_bid_depth + weighted_ask_depth
            )

            # Save to database (async, non-blocking)
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO lob_data (
                        time, symbol,
                        mid_price, spread, spread_bps,
                        bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
                        bid_price_2, bid_volume_2, ask_price_2, ask_volume_2,
                        bid_price_3, bid_volume_3, ask_price_3, ask_volume_3,
                        bid_price_4, bid_volume_4, ask_price_4, ask_volume_4,
                        bid_price_5, bid_volume_5, ask_price_5, ask_volume_5,
                        total_bid_volume, total_ask_volume, volume_imbalance,
                        weighted_mid_price, price_range, depth_imbalance
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25,
                        $26, $27, $28, $29, $30, $31
                    )
                    ON CONFLICT (time, symbol) DO NOTHING
                    """,
                    timestamp,
                    self.symbol,
                    mid_price,
                    spread,
                    spread_bps,
                    bid_prices[0],
                    bid_volumes[0],
                    ask_prices[0],
                    ask_volumes[0],
                    bid_prices[1],
                    bid_volumes[1],
                    ask_prices[1],
                    ask_volumes[1],
                    bid_prices[2],
                    bid_volumes[2],
                    ask_prices[2],
                    ask_volumes[2],
                    bid_prices[3],
                    bid_volumes[3],
                    ask_prices[3],
                    ask_volumes[3],
                    bid_prices[4],
                    bid_volumes[4],
                    ask_prices[4],
                    ask_volumes[4],
                    total_bid_volume,
                    total_ask_volume,
                    volume_imbalance,
                    weighted_mid_price,
                    price_range,
                    depth_imbalance,
                )
        except Exception as e:
            logger.error(f"Error saving LOB data: {e}")

    async def _save_prediction(
        self,
        timestamp: datetime,
        predictions: dict,
        mid_price: float,
        spread_bps: float,
        volume_imbalance: float,
        inference_time_ms: float,
    ) -> None:
        """
        Save prediction to database (async, non-blocking).

        Args:
            timestamp: Prediction timestamp
            predictions: Predictions dict by horizon
            mid_price: Current mid price
            spread_bps: Spread in basis points
            volume_imbalance: Volume imbalance ratio
            inference_time_ms: Inference time in milliseconds
        """
        if not self.save_predictions or self.db_pool is None:
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
                    self.symbol,
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
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")

    async def _process_lob_update(self, lob_snapshot: dict) -> None:
        """
        Process LOB update: compute features, make prediction, save both.

        This is the hot path - optimized for minimum latency.

        Args:
            lob_snapshot: Raw LOB data from Binance
        """
        try:
            update_start = time.time()
            self.total_updates += 1

            timestamp = datetime.now(timezone.utc)

            # Compute features (FAST: ~1ms)
            features = self.feature_engineer.compute_features(lob_snapshot)
            features_flat = features.flatten()

            # Add to buffer
            self.feature_buffer.append(features_flat)
            if len(self.feature_buffer) > self.sequence_length:
                self.feature_buffer.pop(0)

            # Extract context for logging and saving
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

            # Save LOB data (async, non-blocking)
            save_lob_task = asyncio.create_task(
                self._save_lob_data(timestamp, lob_snapshot, features)
            )

            # Make prediction if buffer is ready
            prediction_dict = None
            inference_time_ms = 0

            if len(self.feature_buffer) == self.sequence_length:
                # Prediction (FAST: ~10ms)
                inference_start = time.time()

                # Stack features
                features_seq = np.array(self.feature_buffer)  # (seq_len, 43)

                # Normalize
                features_seq = (features_seq - self.mean) / self.std

                # Convert to tensor
                x = torch.from_numpy(features_seq).float().unsqueeze(0)
                x = x.to(self.device)

                # Predict
                with torch.no_grad():
                    predictions = self.model(x)  # (1, 5)

                predictions = predictions.cpu().numpy().flatten()

                inference_time_ms = (time.time() - inference_start) * 1000
                self.total_inference_time += inference_time_ms
                self.total_predictions += 1

                # Format predictions
                prediction_dict = {
                    "1s": float(predictions[0]),
                    "5s": float(predictions[1]),
                    "10s": float(predictions[2]),
                    "30s": float(predictions[3]),
                    "60s": float(predictions[4]),
                }

                # Save prediction (async, non-blocking)
                save_pred_task = asyncio.create_task(
                    self._save_prediction(
                        timestamp,
                        prediction_dict,
                        mid_price,
                        spread_bps,
                        volume_imbalance,
                        inference_time_ms,
                    )
                )

                # Log prediction
                logger.info(
                    f"Predictions: 1s={prediction_dict['1s']:+.4f}%, "
                    f"5s={prediction_dict['5s']:+.4f}%, "
                    f"10s={prediction_dict['10s']:+.4f}%, "
                    f"30s={prediction_dict['30s']:+.4f}%, "
                    f"60s={prediction_dict['60s']:+.4f}% "
                    f"[{inference_time_ms:.1f}ms] "
                    f"mid={mid_price:.2f} spread={spread_bps:.2f}bps"
                )

                # Wait for saves to complete (optional, for timing)
                await asyncio.gather(save_lob_task, save_pred_task)
            else:
                # Just wait for LOB save
                await save_lob_task

                # Log buffer status occasionally
                if self.total_updates % 10 == 0:
                    logger.info(
                        f"Buffering: {len(self.feature_buffer)}/{self.sequence_length} "
                        f"(need {self.sequence_length - len(self.feature_buffer)} more)"
                    )

            # Total processing time
            total_time_ms = (time.time() - update_start) * 1000
            self.total_save_time += total_time_ms

        except Exception as e:
            logger.error(f"Error processing LOB update: {e}", exc_info=True)

    async def start(self) -> None:
        """Start the unified production system."""
        logger.info("=" * 70)
        logger.info("Starting Unified Production System")
        logger.info("=" * 70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Model version: {self.model_version}")
        logger.info(f"Save LOB data: {self.save_lob_data}")
        logger.info(f"Save predictions: {self.save_predictions}")
        logger.info("=" * 70)

        # Initialize database
        await self._init_db()

        # Start WebSocket stream
        logger.info("Starting WebSocket stream...")
        await self.stream.start()

    async def stop(self) -> None:
        """Stop the system gracefully."""
        logger.info("=" * 70)
        logger.info("Stopping Unified Production System")
        logger.info("=" * 70)

        # Stop stream
        await self.stream.stop()

        # Close database pool
        if self.db_pool:
            await self.db_pool.close()

        # Print statistics
        logger.info(f"Total LOB updates processed: {self.total_updates:,}")
        logger.info(f"Total predictions made: {self.total_predictions:,}")

        if self.total_predictions > 0:
            avg_inference = self.total_inference_time / self.total_predictions
            logger.info(f"Average inference time: {avg_inference:.1f}ms")

        if self.total_updates > 0:
            avg_total = self.total_save_time / self.total_updates
            logger.info(f"Average total processing time: {avg_total:.1f}ms")

        logger.info("=" * 70)
        logger.info("System stopped")


# =============================================================================
# Main
# =============================================================================


async def main(args):
    """Main entry point."""
    setup_logging(log_level=args.log_level)

    # Create system
    system = UnifiedProductionSystem(
        model_path=Path(args.model),
        symbol=args.symbol,
        model_version=args.model_version,
        sequence_length=args.sequence_length,
        device=args.device,
        save_lob_data=not args.no_save_lob,
        save_predictions=not args.no_save_predictions,
    )

    # Setup signal handlers for graceful shutdown
    import signal

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(system.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await system.stop()


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified production system: Live data + Real-time inference"
    )

    # Required
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")

    # Optional
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--model-version", type=str, default="v1.0", help="Model version identifier (default: v1.0)"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=100, help="Input sequence length (default: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device for inference (default: mps)",
    )
    parser.add_argument(
        "--no-save-lob", action="store_true", help="Do not save LOB data to database"
    )
    parser.add_argument(
        "--no-save-predictions", action="store_true", help="Do not save predictions to database"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    asyncio.run(main(args))
