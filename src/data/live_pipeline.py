"""
End-to-end live data pipeline.

Integrates WebSocket streaming, feature engineering, and database writing
for continuous LOB data collection and storage.
"""

import asyncio
import signal
from datetime import datetime, timezone
from typing import Any

from config.logging_config import get_logger
from config.settings import settings
from src.data.binance_stream import BinanceDepthStream
from src.data.db_writer import TimescaleDBWriter
from src.data.feature_engineering import LOBFeatureEngineering

logger = get_logger(__name__)


class LiveDataPipeline:
    """
    Complete live data pipeline from WebSocket to database.

    Orchestrates the full data flow:
    1. Receive LOB updates via WebSocket
    2. Compute features
    3. Write to TimescaleDB

    Args:
        symbol (str): Trading pair to stream (e.g., 'BTCUSDT').
        levels (int, optional): LOB depth levels. Defaults to 5.
        use_testnet (bool, optional): Use Binance testnet. Defaults to False.

    Attributes:
        symbol (str): Trading pair being streamed.
        stream (BinanceDepthStream): WebSocket client.
        feature_engineer (LOBFeatureEngineering): Feature computation.
        db_writer (TimescaleDBWriter): Database writer.
        snapshots_processed (int): Total snapshots processed.
        snapshots_written (int): Total snapshots written to DB.

    Example:
        >>> pipeline = LiveDataPipeline('BTCUSDT')
        >>> await pipeline.start()
        >>> # ... runs until stopped ...
        >>> await pipeline.stop()

    Note:
        Designed to run continuously. Use Ctrl+C or SIGTERM to stop gracefully.
    """

    def __init__(
        self,
        symbol: str,
        levels: int = 5,
        use_testnet: bool = False,
    ):
        """Initialize live data pipeline."""
        self.symbol = symbol.upper()
        self.levels = levels
        self.use_testnet = use_testnet

        # Components
        self.stream = BinanceDepthStream(
            symbol=symbol,
            callback=self._process_lob_update,
            levels=levels,
            use_testnet=use_testnet,
        )

        self.feature_engineer = LOBFeatureEngineering(
            levels=levels,
            normalize=False,
        )

        self.db_writer = TimescaleDBWriter(
            pool_size=5,
            batch_size=100,
        )

        # Statistics
        self.snapshots_processed = 0
        self.snapshots_written = 0
        self.snapshots_failed = 0

        # Control flag
        self._running = False

        logger.info(
            f"Initialized LiveDataPipeline for {symbol} "
            f"(levels={levels}, testnet={use_testnet})"
        )

    async def _process_lob_update(self, data: dict[str, Any]) -> None:
        """
        Process incoming LOB update.

        Callback for WebSocket stream. Computes features and writes to database.

        Args:
            data (dict): Raw LOB data from Binance.
        """
        try:
            self.snapshots_processed += 1

            # Extract timestamp
            timestamp = data.get("local_time", datetime.now(timezone.utc))

            # Parse bids and asks
            bids = []
            asks = []

            for bid in data.get("bids", [])[: self.levels]:
                if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                    bids.append((float(bid[0]), float(bid[1])))

            for ask in data.get("asks", [])[: self.levels]:
                if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                    asks.append((float(ask[0]), float(ask[1])))

            if not bids or not asks:
                logger.warning("Received LOB update with no bids or asks")
                return

            # Write to database
            success = await self.db_writer.write_lob_snapshot(
                symbol=self.symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks,
            )

            if success:
                self.snapshots_written += 1
            else:
                self.snapshots_failed += 1

            # Log progress periodically
            if self.snapshots_processed % 100 == 0:
                logger.info(
                    f"Pipeline stats: {self.snapshots_processed} processed, "
                    f"{self.snapshots_written} written, "
                    f"{self.snapshots_failed} failed"
                )

        except Exception as e:
            self.snapshots_failed += 1
            logger.error(f"Error processing LOB update: {e}", exc_info=True)

    async def start(self) -> None:
        """
        Start the live data pipeline.

        Connects to database and begins streaming data. Runs until stop() is called.

        Example:
            >>> pipeline = LiveDataPipeline('BTCUSDT')
            >>> await pipeline.start()
        """
        try:
            self._running = True

            # Connect to database
            logger.info("Connecting to database...")
            await self.db_writer.connect()

            # Start WebSocket stream
            logger.info(f"Starting live data pipeline for {self.symbol}")
            await self.stream.start()

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            await self.stop()

    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        if not self._running:
            return  # Already stopped

        logger.info("Stopping live data pipeline...")
        self._running = False

        # Stop stream first (stops new messages)
        await self.stream.stop()

        # Small delay to let any in-flight messages finish
        await asyncio.sleep(0.5)

        # Then close database
        await self.db_writer.close()

        # Log final statistics
        logger.info("=" * 60)
        logger.info("Pipeline Statistics:")
        logger.info(f"  Total snapshots processed: {self.snapshots_processed}")
        logger.info(f"  Successfully written: {self.snapshots_written}")
        logger.info(f"  Failed: {self.snapshots_failed}")
        success_rate = (
            (self.snapshots_written / self.snapshots_processed * 100)
            if self.snapshots_processed > 0
            else 0
        )
        logger.info(f"  Success rate: {success_rate:.2f}%")
        logger.info("=" * 60)

    def get_statistics(self) -> dict:
        """
        Get pipeline statistics.

        Returns:
            dict: Statistics including counts and rates.
        """
        return {
            "symbol": self.symbol,
            "snapshots_processed": self.snapshots_processed,
            "snapshots_written": self.snapshots_written,
            "snapshots_failed": self.snapshots_failed,
            "success_rate": (
                (self.snapshots_written / self.snapshots_processed)
                if self.snapshots_processed > 0
                else 0
            ),
        }


# =============================================================================
# Main Entry Point
# =============================================================================


async def main():
    """Run live data pipeline with graceful shutdown."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    logger.info("=" * 60)
    logger.info("LOB Prediction System - Live Data Pipeline")
    logger.info("=" * 60)

    # Create pipeline
    pipeline = LiveDataPipeline(
        symbol=settings.binance_symbols[0],  # Use first symbol from config
        levels=settings.lob_depth_levels,
        use_testnet=settings.binance_testnet,
    )

    # Setup signal handlers for graceful shutdown
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

    logger.info("Pipeline shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
