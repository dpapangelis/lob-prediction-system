"""
Real-time LOB data collection with feature engineering.

This module integrates WebSocket streaming with feature engineering to
collect, process, and buffer limit order book data for model training
and inference.
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from config.logging_config import get_logger
from config.settings import settings
from src.data.binance_stream import BinanceDepthStream
from src.data.feature_engineering import LOBFeatureEngineering

logger = get_logger(__name__)


class LOBDataCollector:
    """
    Collect and process real-time LOB data with feature engineering.

    Integrates WebSocket streaming with feature computation to maintain
    a rolling buffer of processed LOB snapshots. Data can be accessed
    for real-time inference or batch training.

    Args:
        symbol (str): Trading pair to collect (e.g., 'BTCUSDT').
        levels (int, optional): LOB depth levels. Defaults to 5.
        buffer_size (int, optional): Maximum snapshots to keep in memory.
            Defaults to 10000 (~3 hours at 1 update/sec).
        use_testnet (bool, optional): Use Binance testnet. Defaults to False.

    Attributes:
        symbol (str): Trading pair being collected.
        feature_engineer (LOBFeatureEngineering): Feature computation engine.
        data_buffer (deque): Rolling buffer of processed data points.
        stream (BinanceDepthStream): WebSocket client instance.

    Example:
        >>> collector = LOBDataCollector('BTCUSDT', buffer_size=1000)
        >>> await collector.start()
        >>> # ... collect data ...
        >>> df = collector.get_dataframe()
        >>> print(f"Collected {len(df)} snapshots")
        >>> await collector.stop()

    Note:
        Data is kept in memory. For long-running collection, consider
        periodically saving to disk or database to avoid memory issues.
    """

    def __init__(
        self,
        symbol: str,
        levels: int = 5,
        buffer_size: int = 10000,
        use_testnet: bool = False,
    ):
        """Initialize data collector."""
        self.symbol = symbol.upper()
        self.levels = levels
        self.buffer_size = buffer_size
        self.use_testnet = use_testnet

        # Feature engineering
        self.feature_engineer = LOBFeatureEngineering(
            levels=levels, normalize=False  # We'll normalize during training
        )

        # Data buffer: stores (timestamp, features, raw_data) tuples
        self.data_buffer: deque = deque(maxlen=buffer_size)

        # Statistics
        self.snapshots_received = 0
        self.snapshots_processed = 0
        self.errors = 0

        # WebSocket stream
        self.stream = BinanceDepthStream(
            symbol=symbol,
            callback=self._process_snapshot,
            levels=levels,
            use_testnet=use_testnet,
        )

        logger.info(
            f"Initialized LOBDataCollector for {symbol} "
            f"(buffer_size={buffer_size}, levels={levels})"
        )

    async def _process_snapshot(self, data: dict[str, Any]) -> None:
        """
        Process incoming LOB snapshot.

        Callback function for WebSocket stream. Computes features and
        stores in buffer.

        Args:
            data (dict): Raw LOB data from Binance.
        """
        try:
            self.snapshots_received += 1

            # Extract timestamp
            timestamp = data.get("local_time", datetime.utcnow())

            # Compute features
            features = self.feature_engineer.compute_features(data)

            # Store in buffer (timestamp, features, raw_data)
            self.data_buffer.append(
                {
                    "timestamp": timestamp,
                    "features": features[0],  # Remove batch dimension
                    "mid_price": (
                        (float(data["bids"][0][0]) + float(data["asks"][0][0])) / 2
                        if data.get("bids") and data.get("asks")
                        else None
                    ),
                    "best_bid": float(data["bids"][0][0]) if data.get("bids") else None,
                    "best_ask": float(data["asks"][0][0]) if data.get("asks") else None,
                    "symbol": self.symbol,
                }
            )

            self.snapshots_processed += 1

            # Log progress periodically
            if self.snapshots_processed % 100 == 0:
                logger.info(
                    f"Processed {self.snapshots_processed} snapshots "
                    f"(buffer size: {len(self.data_buffer)})"
                )

        except Exception as e:
            self.errors += 1
            logger.error(f"Error processing snapshot: {e}", exc_info=True)

    async def start(self) -> None:
        """
        Start collecting data.

        Begins WebSocket stream and data processing. This method blocks
        until stop() is called.

        Example:
            >>> collector = LOBDataCollector('BTCUSDT')
            >>> task = asyncio.create_task(collector.start())
            >>> await asyncio.sleep(60)  # Collect for 1 minute
            >>> await collector.stop()
            >>> await task
        """
        logger.info(f"Starting data collection for {self.symbol}")
        await self.stream.start()

    async def stop(self) -> None:
        """
        Stop collecting data.

        Gracefully stops the WebSocket stream and logs collection statistics.
        """
        logger.info(f"Stopping data collection for {self.symbol}")
        await self.stream.stop()

        logger.info(
            f"Collection complete: {self.snapshots_processed} snapshots processed, "
            f"{self.errors} errors, buffer size: {len(self.data_buffer)}"
        )

    def get_dataframe(
        self, include_features: bool = True, include_raw: bool = False
    ) -> pd.DataFrame:
        """
        Export collected data as a pandas DataFrame.

        Args:
            include_features (bool, optional): Include computed features.
                Defaults to True.
            include_raw (bool, optional): Include raw LOB data (large).
                Defaults to False.

        Returns:
            pd.DataFrame: Collected data with timestamps as index.

        Example:
            >>> df = collector.get_dataframe()
            >>> print(df.columns)
            ['timestamp', 'mid_price', 'mid_price', 'spread_bps', ...]

        Note:
            Returns empty DataFrame if no data collected.
        """
        if not self.data_buffer:
            logger.warning("No data in buffer")
            return pd.DataFrame()

        # Extract data from buffer
        records = []
        for item in self.data_buffer:
            record = {
                "timestamp": item["timestamp"],
                "symbol": item["symbol"],
                "mid_price": item["mid_price"],
                "best_bid": item["best_bid"],
                "best_ask": item["best_ask"],
            }

            if include_features:
                # Add all features
                for i, feature_name in enumerate(self.feature_engineer.feature_names):
                    record[feature_name] = item["features"][i]

            records.append(record)

        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)

        logger.info(f"Exported {len(df)} rows to DataFrame")
        return df

    def get_latest_snapshot(self) -> Optional[dict]:
        """
        Get the most recent snapshot from buffer.

        Returns:
            dict: Latest snapshot with timestamp, features, and prices,
                or None if buffer is empty.

        Example:
            >>> snapshot = collector.get_latest_snapshot()
            >>> if snapshot:
            ...     print(f"Latest mid-price: {snapshot['mid_price']}")
        """
        if not self.data_buffer:
            return None
        return self.data_buffer[-1]

    def get_statistics(self) -> dict:
        """
        Get collection statistics.

        Returns:
            dict: Statistics including snapshot counts, errors, and buffer info.
        """
        return {
            "symbol": self.symbol,
            "snapshots_received": self.snapshots_received,
            "snapshots_processed": self.snapshots_processed,
            "errors": self.errors,
            "buffer_size": len(self.data_buffer),
            "buffer_capacity": self.buffer_size,
            "buffer_usage_pct": (len(self.data_buffer) / self.buffer_size) * 100,
        }

    def save_to_csv(self, filepath: str) -> None:
        """
        Save collected data to CSV file.

        Args:
            filepath (str): Path to output CSV file.

        Example:
            >>> collector.save_to_csv('data/raw/btcusdt_lob.csv')
        """
        df = self.get_dataframe()
        df.to_csv(filepath)
        logger.info(f"Saved {len(df)} rows to {filepath}")


# =============================================================================
# Testing
# =============================================================================


async def main():
    """Test data collector with live data."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    logger.info("=== Testing LOB Data Collector ===")

    # Create collector
    collector = LOBDataCollector(
        symbol="BTCUSDT",
        levels=settings.lob_depth_levels,
        buffer_size=100,  # Small buffer for testing
        use_testnet=settings.binance_testnet,
    )

    # Start collection
    task = asyncio.create_task(collector.start())

    try:
        logger.info("Collecting data for 30 seconds...")
        await asyncio.sleep(30)

        # Get statistics
        stats = collector.get_statistics()
        logger.info("\nCollection Statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Get DataFrame
        df = collector.get_dataframe()
        logger.info(f"\nCollected DataFrame shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns[:10])}...")  # First 10 columns

        # Show sample data
        logger.info("\nSample data (first 3 rows):")
        print(
            df[["symbol", "mid_price", "spread_bps", "total_volume_imbalance"]].head(3).to_string()
        )

        # Save to CSV
        output_path = settings.data_dir / "raw" / "test_collection.csv"
        collector.save_to_csv(str(output_path))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await collector.stop()
        await task

    logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(main())
