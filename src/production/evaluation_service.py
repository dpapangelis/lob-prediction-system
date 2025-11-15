"""
Continuous evaluation service for predictions.

Runs in the background and automatically evaluates predictions
as they become ready, creating outcomes for real-time metrics.

Usage:
    python -m src.production.evaluation_service
"""

import asyncio
from datetime import datetime, timedelta, timezone

import asyncpg

from config.logging_config import get_logger, setup_logging
from config.settings import settings

logger = get_logger(__name__)


class EvaluationService:
    """
    Continuous evaluation service.

    Polls for predictions that are ready to be evaluated and
    matches them with actual outcomes from LOB data.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        model_version: str = "v1.0",
        poll_interval: int = 5,  # seconds
    ):
        """Initialize evaluation service."""
        self.symbol = symbol
        self.model_version = model_version
        self.poll_interval = poll_interval
        self.db_pool = None

        self.horizons = {
            "1s": 1,
            "5s": 5,
            "10s": 10,
            "30s": 30,
            "60s": 60,
        }

        self.total_evaluated = 0
        self.start_time = datetime.now(timezone.utc)

    async def _init_db(self):
        """Initialize database connection pool."""
        self.db_pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=2,
            max_size=5,
        )
        logger.info("Database connection pool initialized")

    async def _find_actual_price(
        self,
        conn: asyncpg.Connection,
        prediction_time: datetime,
        horizon_seconds: int,
    ) -> float:
        """
        Find actual mid price at prediction_time + horizon.

        Args:
            conn: Database connection
            prediction_time: When prediction was made
            horizon_seconds: How many seconds in the future

        Returns:
            Actual mid price, or None if not found
        """
        target_time = prediction_time + timedelta(seconds=horizon_seconds)

        # Find closest LOB data within ±2 seconds
        row = await conn.fetchrow(
            """
            SELECT mid_price, time
            FROM lob_data
            WHERE symbol = $1
              AND time >= $2
              AND time <= $3
            ORDER BY ABS(EXTRACT(EPOCH FROM (time - $4)))
            LIMIT 1
            """,
            self.symbol,
            target_time - timedelta(seconds=2),
            target_time + timedelta(seconds=2),
            target_time,
        )

        return float(row["mid_price"]) if row else None

    async def _evaluate_predictions(self):
        """Evaluate all pending predictions."""
        async with self.db_pool.acquire() as conn:
            # Find predictions that haven't been evaluated yet
            # Look back 2 minutes to catch any we missed
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=2)

            predictions = await conn.fetch(
                """
                SELECT p.time, p.pred_1s, p.pred_5s, p.pred_10s, p.pred_30s, p.pred_60s,
                       p.mid_price
                FROM predictions p
                LEFT JOIN prediction_outcomes po
                    ON p.time = po.prediction_time
                    AND p.symbol = po.symbol
                    AND p.model_version = po.model_version
                    AND po.horizon = '60s'  -- Check longest horizon
                WHERE p.symbol = $1
                  AND p.model_version = $2
                  AND p.time >= $3
                  AND po.prediction_time IS NULL  -- Not yet evaluated
                ORDER BY p.time
                """,
                self.symbol,
                self.model_version,
                cutoff_time,
            )

            if not predictions:
                return

            logger.info(f"Found {len(predictions)} predictions to evaluate")

            outcomes_to_insert = []

            for pred in predictions:
                pred_time = pred["time"]
                initial_price = float(pred["mid_price"])

                # Evaluate each horizon
                for horizon_name, horizon_seconds in self.horizons.items():
                    # Check if this horizon is ready to evaluate
                    elapsed = (datetime.now(timezone.utc) - pred_time).total_seconds()
                    if elapsed < horizon_seconds:
                        continue  # Not ready yet

                    # Get predicted return
                    pred_key = f"pred_{horizon_name}"
                    predicted_return = float(pred[pred_key])

                    # Find actual price
                    actual_price = await self._find_actual_price(conn, pred_time, horizon_seconds)

                    if actual_price is None:
                        continue  # No LOB data available

                    # Calculate actual return
                    actual_return = ((actual_price - initial_price) / initial_price) * 100

                    # Calculate metrics
                    error = predicted_return - actual_return
                    absolute_error = abs(error)
                    squared_error = error**2
                    direction_correct = (predicted_return * actual_return) > 0

                    # Calculate outcome_time (when we actually measured the outcome)
                    outcome_time = pred_time + timedelta(seconds=horizon_seconds)
                    # evaluated_at = datetime.now(timezone.utc)

                    outcomes_to_insert.append(
                        (
                            pred_time,  # $1 prediction_time
                            self.symbol,  # $2 symbol
                            self.model_version,  # $3 model_version
                            horizon_name,  # $4 horizon
                            predicted_return,  # $5 predicted_return
                            actual_return,  # $6 actual_return
                            error,  # $7 error
                            absolute_error,  # $8 absolute_error  <- SWAPPED
                            squared_error,  # $9 squared_error   <- SWAPPED
                            direction_correct,  # $10 direction_correct
                            outcome_time,  # $11 outcome_time
                        )
                    )

            # Batch insert outcomes
            if outcomes_to_insert:
                await conn.executemany(
                    """
                    INSERT INTO prediction_outcomes (
                        prediction_time, symbol, model_version, horizon,
                        predicted_return, actual_return, error, squared_error,
                        absolute_error, direction_correct, outcome_time
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (prediction_time, symbol, model_version, horizon)
                    DO NOTHING
                    """,
                    outcomes_to_insert,
                )

                self.total_evaluated += len(outcomes_to_insert)
                logger.info(f"✓ Evaluated {len(outcomes_to_insert)} outcomes")

    async def run(self):
        """Run the evaluation service continuously."""
        logger.info("=" * 70)
        logger.info("Starting Continuous Evaluation Service")
        logger.info("=" * 70)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Model version: {self.model_version}")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info("=" * 70)

        await self._init_db()

        try:
            while True:
                try:
                    await self._evaluate_predictions()
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}", exc_info=True)

                # Log stats every 60 seconds
                elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                if int(elapsed) % 60 == 0 and elapsed > 0:
                    rate = self.total_evaluated / elapsed * 60
                    logger.info(
                        f"Stats: {self.total_evaluated} outcomes evaluated " f"({rate:.1f}/min)"
                    )

                await asyncio.sleep(self.poll_interval)

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            if self.db_pool:
                await self.db_pool.close()
            logger.info("=" * 70)
            logger.info("Evaluation service stopped")
            logger.info(f"Total outcomes evaluated: {self.total_evaluated}")
            logger.info("=" * 70)


async def main():
    """Main entry point."""
    setup_logging(log_level="INFO")

    service = EvaluationService(
        symbol="BTCUSDT",
        model_version="v1.0",
        poll_interval=5,
    )

    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
