"""
Evaluate predictions against actual outcomes.

This script:
1. Reads predictions from the database
2. Computes actual returns at each horizon
3. Saves prediction outcomes for analysis
4. Generates performance reports

Usage:
    python scripts/evaluate_predictions.py --hours 24
    python scripts/evaluate_predictions.py --start "2024-11-10 00:00:00" --end "2024-11-10 23:59:59"
"""

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

import asyncpg
import numpy as np
import pandas as pd

from config.logging_config import get_logger, setup_logging
from config.settings import settings

# from typing import Optional


logger = get_logger(__name__)


async def compute_actual_returns(
    conn: asyncpg.Connection,
    symbol: str,
    prediction_time: datetime,
    horizons: dict,
) -> dict:
    """
    Compute actual returns for each horizon.

    Args:
        conn: Database connection
        symbol: Trading symbol
        prediction_time: Time prediction was made
        horizons: Dict of horizon names to seconds (e.g., {'1s': 1, '5s': 5})

    Returns:
        Dict of horizon names to (actual_return, outcome_time) tuples
    """
    actual_returns = {}

    # Get mid price at prediction time
    pred_price_row = await conn.fetchrow(
        """
        SELECT mid_price FROM lob_data
        WHERE symbol = $1 AND time <= $2
        ORDER BY time DESC
        LIMIT 1
        """,
        symbol,
        prediction_time,
    )

    if not pred_price_row:
        logger.warning(f"No LOB data found at prediction time {prediction_time}")
        return {}

    pred_price = pred_price_row["mid_price"]

    # For each horizon, get price after horizon seconds
    for horizon_name, horizon_seconds in horizons.items():
        future_time = prediction_time + timedelta(seconds=horizon_seconds)

        future_price_row = await conn.fetchrow(
            """
            SELECT mid_price, time FROM lob_data
            WHERE symbol = $1 AND time >= $2
            ORDER BY time ASC
            LIMIT 1
            """,
            symbol,
            future_time,
        )

        if future_price_row:
            future_price = future_price_row["mid_price"]
            outcome_time = future_price_row["time"]

            # Calculate return as percentage
            actual_return = ((future_price - pred_price) / pred_price) * 100
            actual_returns[horizon_name] = (actual_return, outcome_time)
        else:
            logger.debug(f"No future price found for {horizon_name} at {future_time}")
            actual_returns[horizon_name] = (None, None)

    return actual_returns


async def evaluate_predictions(
    symbol: str,
    model_version: str,
    start_time: datetime,
    end_time: datetime,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Evaluate all predictions in time range.

    Args:
        symbol: Trading symbol
        model_version: Model version to evaluate
        start_time: Start of evaluation period
        end_time: End of evaluation period
        batch_size: Number of predictions to process in each batch

    Returns:
        DataFrame with evaluation results
    """
    horizons = {
        "1s": 1,
        "5s": 5,
        "10s": 10,
        "30s": 30,
        "60s": 60,
    }

    # Connect to database
    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    try:
        # Get all predictions in time range
        predictions = await conn.fetch(
            """
            SELECT * FROM predictions
            WHERE symbol = $1
              AND model_version = $2
              AND time >= $3
              AND time <= $4
            ORDER BY time ASC
            """,
            symbol,
            model_version,
            start_time,
            end_time,
        )

        logger.info(f"Found {len(predictions)} predictions to evaluate")

        if len(predictions) == 0:
            logger.warning("No predictions found in specified time range")
            return pd.DataFrame()

        results = []

        # Process in batches
        for i in range(0, len(predictions), batch_size):
            batch = predictions[i : i + batch_size]

            for pred_row in batch:
                prediction_time = pred_row["time"]

                # Compute actual returns
                actual_returns = await compute_actual_returns(
                    conn,
                    symbol,
                    prediction_time,
                    horizons,
                )

                # For each horizon
                for horizon_name in horizons.keys():
                    pred_col = f"pred_{horizon_name}"
                    predicted_return = pred_row[pred_col]

                    actual_data = actual_returns.get(horizon_name)
                    if actual_data and actual_data[0] is not None:
                        actual_return, outcome_time = actual_data

                        # Calculate metrics
                        error = predicted_return - actual_return
                        squared_error = error**2
                        absolute_error = abs(error)

                        # Directional accuracy
                        direction_correct = (
                            (predicted_return > 0 and actual_return > 0)
                            or (predicted_return < 0 and actual_return < 0)
                            or (predicted_return == 0 and actual_return == 0)
                        )

                        results.append(
                            {
                                "prediction_time": prediction_time,
                                "symbol": symbol,
                                "model_version": model_version,
                                "horizon": horizon_name,
                                "predicted_return": predicted_return,
                                "actual_return": actual_return,
                                "error": error,
                                "squared_error": squared_error,
                                "absolute_error": absolute_error,
                                "direction_correct": direction_correct,
                                "outcome_time": outcome_time,
                            }
                        )

            # Log progress
            logger.info(
                f"Processed {min(i+batch_size, len(predictions))}/{len(predictions)} predictions"
            )

        # Create DataFrame
        df = pd.DataFrame(results)

        # Save to database
        if len(results) > 0:
            logger.info(f"Saving {len(results)} outcomes to database...")

            await conn.executemany(
                """
                INSERT INTO prediction_outcomes (
                    prediction_time, symbol, model_version, horizon,
                    predicted_return, actual_return, error, squared_error,
                    absolute_error, direction_correct, outcome_time
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (prediction_time, symbol, model_version, horizon)
                DO UPDATE SET
                    actual_return = EXCLUDED.actual_return,
                    error = EXCLUDED.error,
                    squared_error = EXCLUDED.squared_error,
                    absolute_error = EXCLUDED.absolute_error,
                    direction_correct = EXCLUDED.direction_correct,
                    outcome_time = EXCLUDED.outcome_time,
                    evaluated_at = NOW()
                """,
                [
                    (
                        r["prediction_time"],
                        r["symbol"],
                        r["model_version"],
                        r["horizon"],
                        r["predicted_return"],
                        r["actual_return"],
                        r["error"],
                        r["squared_error"],
                        r["absolute_error"],
                        r["direction_correct"],
                        r["outcome_time"],
                    )
                    for r in results
                ],
            )

            logger.info(f"✓ Saved {len(results)} outcomes to database")

        return df

    finally:
        await conn.close()


async def generate_report(df: pd.DataFrame) -> None:
    """
    Generate evaluation report.

    Args:
        df: DataFrame with evaluation results
    """
    if len(df) == 0:
        logger.warning("No data to generate report")
        return

    logger.info("=" * 70)
    logger.info("Prediction Evaluation Report")
    logger.info("=" * 70)
    logger.info(f"Total predictions evaluated: {len(df):,}")
    logger.info(f"Time range: {df['prediction_time'].min()} to {df['prediction_time'].max()}")
    logger.info("")

    # Group by horizon
    for horizon in ["1s", "5s", "10s", "30s", "60s"]:
        horizon_df = df[df["horizon"] == horizon]

        if len(horizon_df) == 0:
            continue

        # Compute metrics
        mse = horizon_df["squared_error"].mean()
        rmse = np.sqrt(mse)
        mae = horizon_df["absolute_error"].mean()

        # R²
        ss_res = horizon_df["squared_error"].sum()
        ss_tot = ((horizon_df["actual_return"] - horizon_df["actual_return"].mean()) ** 2).sum()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Directional accuracy
        dir_acc = horizon_df["direction_correct"].mean() * 100

        # Precision and recall
        true_positives = (
            (horizon_df["predicted_return"] > 0) & (horizon_df["actual_return"] > 0)
        ).sum()
        false_positives = (
            (horizon_df["predicted_return"] > 0) & (horizon_df["actual_return"] <= 0)
        ).sum()
        false_negatives = (
            (horizon_df["predicted_return"] <= 0) & (horizon_df["actual_return"] > 0)
        ).sum()

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )

        logger.info(f"Horizon: {horizon}")
        logger.info(f"  Samples: {len(horizon_df):,}")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.4f}%")
        logger.info(f"  MAE: {mae:.4f}%")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  Directional Accuracy: {dir_acc:.2f}%")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info("")

    # Overall statistics
    overall_dir_acc = df["direction_correct"].mean() * 100
    logger.info(f"Overall Directional Accuracy: {overall_dir_acc:.2f}%")
    logger.info("=" * 70)


async def save_performance_metrics(
    symbol: str,
    model_version: str,
    df: pd.DataFrame,
    window_size_hours: int = 1,
) -> None:
    """
    Aggregate and save performance metrics to model_performance table.

    Args:
        symbol: Trading symbol
        model_version: Model version
        df: DataFrame with outcomes
        window_size_hours: Window size for aggregation in hours
    """
    if len(df) == 0:
        return

    logger.info(f"Computing aggregated metrics with {window_size_hours}h windows...")

    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    try:
        # Group by time windows and horizon
        df["time_window"] = df["prediction_time"].dt.floor(f"{window_size_hours}H")

        for horizon in df["horizon"].unique():
            horizon_df = df[df["horizon"] == horizon]

            for window_time, window_df in horizon_df.groupby("time_window"):
                # Regression metrics
                mse = window_df["squared_error"].mean()
                rmse = np.sqrt(mse)
                mae = window_df["absolute_error"].mean()

                ss_res = window_df["squared_error"].sum()
                ss_tot = (
                    (window_df["actual_return"] - window_df["actual_return"].mean()) ** 2
                ).sum()
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Classification metrics
                directional_accuracy = window_df["direction_correct"].mean()

                true_positives = (
                    (window_df["predicted_return"] > 0) & (window_df["actual_return"] > 0)
                ).sum()
                true_negatives = (
                    (window_df["predicted_return"] <= 0) & (window_df["actual_return"] <= 0)
                ).sum()
                false_positives = (
                    (window_df["predicted_return"] > 0) & (window_df["actual_return"] <= 0)
                ).sum()
                false_negatives = (
                    (window_df["predicted_return"] <= 0) & (window_df["actual_return"] > 0)
                ).sum()

                precision = (
                    true_positives / (true_positives + false_positives)
                    if (true_positives + false_positives) > 0
                    else None
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else None
                )
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if precision and recall and (precision + recall) > 0
                    else None
                )

                # Insert into model_performance table
                await conn.execute(
                    """
                    INSERT INTO model_performance (
                        time, symbol, model_version, horizon,
                        num_predictions, mse, rmse, mae, r_squared,
                        directional_accuracy, precision_score, recall_score, f1_score,
                        true_positives, true_negatives, false_positives, false_negatives
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
                    )
                    ON CONFLICT (time, symbol, model_version, horizon)
                    DO UPDATE SET
                        num_predictions = EXCLUDED.num_predictions,
                        mse = EXCLUDED.mse,
                        rmse = EXCLUDED.rmse,
                        mae = EXCLUDED.mae,
                        r_squared = EXCLUDED.r_squared,
                        directional_accuracy = EXCLUDED.directional_accuracy,
                        precision_score = EXCLUDED.precision_score,
                        recall_score = EXCLUDED.recall_score,
                        f1_score = EXCLUDED.f1_score,
                        true_positives = EXCLUDED.true_positives,
                        true_negatives = EXCLUDED.true_negatives,
                        false_positives = EXCLUDED.false_positives,
                        false_negatives = EXCLUDED.false_negatives,
                        computed_at = NOW()
                    """,
                    window_time,
                    symbol,
                    model_version,
                    horizon,
                    len(window_df),
                    float(mse),
                    float(rmse),
                    float(mae),
                    float(r_squared),
                    float(directional_accuracy),
                    float(precision) if precision else None,
                    float(recall) if recall else None,
                    float(f1) if f1 else None,
                    int(true_positives),
                    int(true_negatives),
                    int(false_positives),
                    int(false_negatives),
                )

        logger.info("✓ Saved aggregated metrics to model_performance table")

    finally:
        await conn.close()


async def main(args):
    """Main evaluation function."""
    setup_logging(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("Prediction Evaluation - Matching Predictions with Actuals")
    logger.info("=" * 70)

    # Determine time range
    if args.start and args.end:
        start_time = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end_time = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    else:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=args.hours)

    logger.info(f"Evaluating predictions from {start_time} to {end_time}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Model version: {args.model_version}")
    logger.info("")

    # Evaluate
    df = await evaluate_predictions(
        symbol=args.symbol,
        model_version=args.model_version,
        start_time=start_time,
        end_time=end_time,
        batch_size=args.batch_size,
    )

    if len(df) == 0:
        logger.warning("No predictions found or no matching LOB data")
        return

    # Generate report
    await generate_report(df)

    # Save aggregated metrics
    if args.save_metrics:
        await save_performance_metrics(
            symbol=args.symbol,
            model_version=args.model_version,
            df=df,
            window_size_hours=args.window_hours,
        )

    # Save to CSV
    if args.output:
        df.to_csv(args.output, index=False)
        logger.info(f"✓ Results saved to {args.output}")

    logger.info("")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions against actual outcomes")

    # Time range
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--hours", type=int, default=24, help="Hours of predictions to evaluate (default: 24)"
    )
    time_group.add_argument(
        "--start", type=str, help="Start time (ISO format: 2024-11-10 00:00:00)"
    )
    parser.add_argument("--end", type=str, help="End time (ISO format, used with --start)")

    # What to evaluate
    parser.add_argument(
        "--symbol", type=str, default="BTCUSDT", help="Trading symbol (default: BTCUSDT)"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0",
        help="Model version to evaluate (default: v1.0)",
    )

    # Processing
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for processing (default: 100)"
    )

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save aggregated metrics to model_performance table",
    )
    parser.add_argument(
        "--window-hours",
        type=int,
        default=1,
        help="Window size for aggregated metrics in hours (default: 1)",
    )

    # Other
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Validate
    if args.start and not args.end:
        parser.error("--end is required when --start is specified")

    asyncio.run(main(args))
