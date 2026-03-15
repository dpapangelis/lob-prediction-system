"""
TimescaleDB writer for LOB data and features.

This module handles writing limit order book data, computed features,
and predictions to TimescaleDB for persistence and historical analysis.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import asyncpg
import numpy as np
from asyncpg import Pool

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class TimescaleDBWriter:
    """
    Async writer for LOB data to TimescaleDB.

    Handles connection pooling, batch writes, and error recovery for
    streaming LOB data into TimescaleDB hypertables.

    Args:
        pool_size (int, optional): Database connection pool size. Defaults to 5.
        batch_size (int, optional): Number of records to batch before writing.
            Defaults to 100.

    Attributes:
        pool (Optional[Pool]): AsyncPG connection pool.
        batch_buffer (list): Buffer for batching writes.
        writes_completed (int): Total successful writes.
        writes_failed (int): Total failed writes.

    Example:
        >>> writer = TimescaleDBWriter()
        >>> await writer.connect()
        >>>
        >>> # Write single snapshot
        >>> await writer.write_lob_snapshot(
        ...     symbol='BTCUSDT',
        ...     timestamp=datetime.now(timezone.utc),
        ...     features=feature_array,
        ...     raw_data={'bids': [...], 'asks': [...]}
        ... )
        >>>
        >>> await writer.close()

    Note:
        Uses asyncpg for high-performance async PostgreSQL access.
        Batches writes for efficiency while maintaining data integrity.
    """

    def __init__(self, pool_size: int = 5, batch_size: int = 100):
        """Initialize database writer."""
        self.pool: Optional[Pool] = None
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.batch_buffer: list[dict[str, Any]] = []

        # Statistics
        self.writes_completed = 0
        self.writes_failed = 0

        logger.info(
            f"Initialized TimescaleDBWriter " f"(pool_size={pool_size}, batch_size={batch_size})"
        )

    async def connect(self) -> None:
        """
        Establish connection pool to TimescaleDB.

        Creates an asyncpg connection pool for efficient database access.

        Raises:
            ConnectionError: If unable to connect to database.
        """
        try:
            logger.info(f"Connecting to TimescaleDB at {settings.db_host}:{settings.db_port}")

            self.pool = await asyncpg.create_pool(
                host=settings.db_host,
                port=settings.db_port,
                database=settings.db_name,
                user=settings.db_user,
                password=settings.db_password,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=60,
            )

            # Test connection
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version()")
                logger.info(f"Connected to: {version}")

            logger.info("TimescaleDB connection pool established")

        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise ConnectionError(f"Database connection failed: {e}")

    async def close(self) -> None:
        """
        Close database connection pool.

        Flushes any remaining batched writes and closes all connections.
        """
        try:
            # Flush remaining batch
            if self.batch_buffer:
                await self.flush_batch()

            if self.pool:
                await self.pool.close()
                logger.info("TimescaleDB connection pool closed")

            logger.info(
                f"Total writes: {self.writes_completed} succeeded, " f"{self.writes_failed} failed"
            )

        except Exception as e:
            logger.error(f"Error closing database connection: {e}")

    async def write_lob_snapshot(
        self,
        symbol: str,
        timestamp: datetime,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        features: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Write a single LOB snapshot to database.

        Args:
            symbol (str): Trading pair symbol.
            timestamp (datetime): Snapshot timestamp (must be timezone-aware).
            bids (list): List of (price, volume) tuples for bid side.
            asks (list): List of (price, volume) tuples for ask side.
            features (np.ndarray, optional): Computed feature vector.

        Returns:
            bool: True if write succeeded, False otherwise.

        Example:
            >>> await writer.write_lob_snapshot(
            ...     symbol='BTCUSDT',
            ...     timestamp=datetime.now(timezone.utc),
            ...     bids=[(100.0, 5.5), (99.9, 3.2)],
            ...     asks=[(100.1, 4.8), (100.2, 6.2)]
            ... )

        Note:
            Timestamps must be timezone-aware (UTC recommended).
        """
        try:
            if not self.pool:
                raise ConnectionError("Database not connected")

            # Ensure timestamp is timezone-aware
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            # Extract best bid/ask
            best_bid_price = bids[0][0] if bids else None
            best_bid_vol = bids[0][1] if bids else None
            best_ask_price = asks[0][0] if asks else None
            best_ask_vol = asks[0][1] if asks else None

            # Calculate mid-price and spread
            if best_bid_price and best_ask_price:
                mid_price = (best_bid_price + best_ask_price) / 2
                spread = best_ask_price - best_bid_price
                imbalance = (
                    (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
                    if best_bid_vol and best_ask_vol
                    else None
                )
            else:
                mid_price = spread = imbalance = None

            # Insert into lob_data table
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO lob_data (
                        time, symbol,
                        bid_price_1, bid_volume_1,
                        bid_price_2, bid_volume_2,
                        bid_price_3, bid_volume_3,
                        ask_price_1, ask_volume_1,
                        ask_price_2, ask_volume_2,
                        ask_price_3, ask_volume_3,
                        mid_price, spread, imbalance
                    ) VALUES (
                        $1, $2,
                        $3, $4, $5, $6, $7, $8,
                        $9, $10, $11, $12, $13, $14,
                        $15, $16, $17
                    )
                    """,
                    timestamp,
                    symbol,
                    bids[0][0] if len(bids) > 0 else None,
                    bids[0][1] if len(bids) > 0 else None,
                    bids[1][0] if len(bids) > 1 else None,
                    bids[1][1] if len(bids) > 1 else None,
                    bids[2][0] if len(bids) > 2 else None,
                    bids[2][1] if len(bids) > 2 else None,
                    asks[0][0] if len(asks) > 0 else None,
                    asks[0][1] if len(asks) > 0 else None,
                    asks[1][0] if len(asks) > 1 else None,
                    asks[1][1] if len(asks) > 1 else None,
                    asks[2][0] if len(asks) > 2 else None,
                    asks[2][1] if len(asks) > 2 else None,
                    mid_price,
                    spread,
                    imbalance,
                )

            self.writes_completed += 1
            return True

        except Exception as e:
            self.writes_failed += 1
            logger.error(f"Failed to write LOB snapshot: {e}")
            return False

    async def flush_batch(self) -> None:
        """
        Flush batched writes to database.

        Writes all buffered records in a single transaction for efficiency.
        """
        if not self.batch_buffer:
            return

        if not self.pool:
            logger.error("Cannot flush batch: database not connected")
            return

        batch = list(self.batch_buffer)
        self.batch_buffer.clear()

        try:
            logger.debug(f"Flushing batch of {len(batch)} records")

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    for record in batch:
                        await conn.execute(
                            """
                            INSERT INTO lob_data (
                                time, symbol,
                                bid_price_1, bid_volume_1,
                                bid_price_2, bid_volume_2,
                                bid_price_3, bid_volume_3,
                                ask_price_1, ask_volume_1,
                                ask_price_2, ask_volume_2,
                                ask_price_3, ask_volume_3,
                                mid_price, spread, imbalance
                            ) VALUES (
                                $1, $2,
                                $3, $4, $5, $6, $7, $8,
                                $9, $10, $11, $12, $13, $14,
                                $15, $16, $17
                            )
                            ON CONFLICT (time, symbol) DO NOTHING
                            """,
                            record["timestamp"],
                            record["symbol"],
                            record.get("bid_price_1"),
                            record.get("bid_volume_1"),
                            record.get("bid_price_2"),
                            record.get("bid_volume_2"),
                            record.get("bid_price_3"),
                            record.get("bid_volume_3"),
                            record.get("ask_price_1"),
                            record.get("ask_volume_1"),
                            record.get("ask_price_2"),
                            record.get("ask_volume_2"),
                            record.get("ask_price_3"),
                            record.get("ask_volume_3"),
                            record.get("mid_price"),
                            record.get("spread"),
                            record.get("imbalance"),
                        )

            self.writes_completed += len(batch)
            logger.debug(f"Successfully flushed {len(batch)} records")

        except Exception as e:
            self.writes_failed += len(batch)
            logger.error(f"Failed to flush batch: {e}")


# =============================================================================
# Testing
# =============================================================================


async def main():
    """Test database writer."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    logger.info("=== Testing TimescaleDB Writer ===")

    # Create writer
    writer = TimescaleDBWriter(pool_size=2)

    try:
        # Connect
        await writer.connect()

        # Write test snapshot
        timestamp = datetime.now(timezone.utc)
        bids = [(101000.0, 5.5), (100999.5, 3.2), (100999.0, 8.1)]
        asks = [(101000.5, 4.8), (101001.0, 6.2), (101001.5, 3.9)]

        logger.info("Writing test snapshot...")
        success = await writer.write_lob_snapshot(
            symbol="BTCUSDT",
            timestamp=timestamp,
            bids=bids,
            asks=asks,
        )

        if success:
            logger.info("✓ Test snapshot written successfully")
        else:
            logger.error("✗ Failed to write test snapshot")

        # Query back
        async with writer.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM lob_data WHERE symbol = $1 ORDER BY time DESC LIMIT 1", "BTCUSDT"
            )
            if row:
                logger.info(f"✓ Retrieved: {dict(row)}")

    finally:
        await writer.close()

    logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(main())
