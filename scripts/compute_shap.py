"""
Compute SHAP values for predictions and save to database.

This script:
1. Loads a trained model
2. Reads predictions from database
3. Computes SHAP values explaining each prediction
4. Saves SHAP values to shap_values table
5. Generates visualizations

Usage:
    python scripts/compute_shap.py --model models/best_model.pth --samples 100
"""

import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg
import numpy as np

from config.logging_config import get_logger, setup_logging
from config.settings import settings
from src.data.feature_engineering import LOBFeatureEngineering
from src.models.shap_explainer import FEATURE_NAMES, create_shap_explainer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = get_logger(__name__)


async def load_background_data(
    symbol: str,
    num_samples: int = 1000,
) -> np.ndarray:
    """
    Load background data for SHAP from database and compute features.

    Uses the same feature engineering as the model to ensure consistency.

    Args:
        symbol: Trading symbol
        num_samples: Number of samples to use as background

    Returns:
        Background data array of shape (num_samples, 43)
    """
    logger.info(f"Loading {num_samples} background samples from database...")

    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    # Initialize feature engineer
    feature_engineer = LOBFeatureEngineering(levels=5, normalize=False)

    try:
        # Get recent LOB data with bids and asks
        rows = await conn.fetch(
            """
            SELECT
                bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
                bid_price_2, bid_volume_2, ask_price_2, ask_volume_2,
                bid_price_3, bid_volume_3, ask_price_3, ask_volume_3,
                bid_price_4, bid_volume_4, ask_price_4, ask_volume_4,
                bid_price_5, bid_volume_5, ask_price_5, ask_volume_5
            FROM lob_data
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT $2
            """,
            symbol,
            num_samples,
        )

        if len(rows) == 0:
            raise ValueError(f"No LOB data found for {symbol}")

        logger.info(f"✓ Loaded {len(rows)} background samples")

        # Convert to LOB snapshot format and compute features
        all_features = []

        for row in rows:
            # Reconstruct LOB snapshot
            lob_snapshot = {
                "bids": [
                    [float(row["bid_price_1"]), float(row["bid_volume_1"])],
                    [float(row["bid_price_2"]), float(row["bid_volume_2"])],
                    [float(row["bid_price_3"]), float(row["bid_volume_3"])],
                    [float(row["bid_price_4"]), float(row["bid_volume_4"])],
                    [float(row["bid_price_5"]), float(row["bid_volume_5"])],
                ],
                "asks": [
                    [float(row["ask_price_1"]), float(row["ask_volume_1"])],
                    [float(row["ask_price_2"]), float(row["ask_volume_2"])],
                    [float(row["ask_price_3"]), float(row["ask_volume_3"])],
                    [float(row["ask_price_4"]), float(row["ask_volume_4"])],
                    [float(row["ask_price_5"]), float(row["ask_volume_5"])],
                ],
            }

            # Compute features using the same method as production
            features = feature_engineer.compute_features(lob_snapshot)
            all_features.append(features.flatten())

        features_array = np.array(all_features, dtype=np.float32)

        logger.info(f"✓ Features shape: {features_array.shape}")
        logger.info(
            f"First sample shape: {all_features[0].shape if len(all_features) > 0 else 'N/A'}"
        )
        logger.info(f"Feature min: {features_array.min()}, max: {features_array.max()}")

        assert features_array.shape[1] == 43, f"Expected 43 features, got {features_array.shape[1]}"

        logger.info(f"✓ Features shape: {features_array.shape}")
        assert features_array.shape[1] == 43, f"Expected 43 features, got {features_array.shape[1]}"

        return features_array

    finally:
        await conn.close()


async def compute_and_save_shap(
    model_path: Path,
    symbol: str,
    model_version: str,
    num_samples: int,
    num_background: int,
    horizons: list,
):
    """
    Compute SHAP values and save to database.

    Args:
        model_path: Path to trained model
        symbol: Trading symbol
        model_version: Model version
        num_samples: Number of predictions to explain
        num_background: Number of background samples for SHAP
        horizons: List of horizons to explain
    """
    # Load background data
    background_data = await load_background_data(symbol, num_background)

    # Create SHAP explainer
    logger.info("Creating SHAP explainer...")
    explainer = create_shap_explainer(
        model_path=model_path,
        background_data=background_data,
        sequence_length=100,
        device="mps",
    )

    # Connect to database
    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    try:
        # Get recent predictions
        logger.info(f"Loading {num_samples} predictions to explain...")
        predictions = await conn.fetch(
            """
            SELECT time FROM predictions
            WHERE symbol = $1 AND model_version = $2
            ORDER BY time DESC
            LIMIT $3
            """,
            symbol,
            model_version,
            num_samples,
        )

        if len(predictions) == 0:
            logger.warning("No predictions found to explain")
            return

        logger.info(f"✓ Found {len(predictions)} predictions")

        # Get corresponding LOB data for these predictions
        prediction_times = [p["time"] for p in predictions]

        # For each prediction, we need the LOB data at that time
        # Use the SAME feature engineering as background data
        logger.info("Loading LOB data for predictions...")

        # Initialize feature engineer (CRITICAL: same as background)
        feature_engineer = LOBFeatureEngineering(levels=5, normalize=False)

        all_shap_records = []

        for i, pred_time in enumerate(prediction_times):
            if i % 10 == 0:
                logger.info(f"Processing prediction {i+1}/{len(prediction_times)}")

            # Get LOB data at prediction time (get bid/ask prices and volumes)
            lob_row = await conn.fetchrow(
                """
                SELECT
                    bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
                    bid_price_2, bid_volume_2, ask_price_2, ask_volume_2,
                    bid_price_3, bid_volume_3, ask_price_3, ask_volume_3,
                    bid_price_4, bid_volume_4, ask_price_4, ask_volume_4,
                    bid_price_5, bid_volume_5, ask_price_5, ask_volume_5
                FROM lob_data
                WHERE symbol = $1 AND time <= $2
                ORDER BY time DESC
                LIMIT 1
                """,
                symbol,
                pred_time,
            )

            if not lob_row:
                logger.warning(f"No LOB data for prediction at {pred_time}")
                continue

            # Reconstruct LOB snapshot (SAME format as background data)
            lob_snapshot = {
                "bids": [
                    [float(lob_row["bid_price_1"]), float(lob_row["bid_volume_1"])],
                    [float(lob_row["bid_price_2"]), float(lob_row["bid_volume_2"])],
                    [float(lob_row["bid_price_3"]), float(lob_row["bid_volume_3"])],
                    [float(lob_row["bid_price_4"]), float(lob_row["bid_volume_4"])],
                    [float(lob_row["bid_price_5"]), float(lob_row["bid_volume_5"])],
                ],
                "asks": [
                    [float(lob_row["ask_price_1"]), float(lob_row["ask_volume_1"])],
                    [float(lob_row["ask_price_2"]), float(lob_row["ask_volume_2"])],
                    [float(lob_row["ask_price_3"]), float(lob_row["ask_volume_3"])],
                    [float(lob_row["ask_price_4"]), float(lob_row["ask_volume_4"])],
                    [float(lob_row["ask_price_5"]), float(lob_row["ask_volume_5"])],
                ],
            }

            # Compute features using LOBFeatureEngineering (SAME as background)
            features_2d = feature_engineer.compute_features(lob_snapshot)
            features = features_2d.flatten()  # (43,)

            # Debug first sample
            if i == 0:
                logger.info(f"First prediction features shape: {features.shape}")

            # Compute SHAP values for each horizon
            for horizon in horizons:
                logger.info(f"  Computing SHAP for {horizon} horizon...")
                shap_values = explainer.explain(
                    features.reshape(1, -1),  # (1, 43)
                    horizon=horizon,
                    nsamples=50,  # Faster computation
                )

                # Get base value
                base_value = explainer.explainers[horizon].expected_value

                # Save to database
                for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                    all_shap_records.append(
                        (
                            pred_time,
                            symbol,
                            model_version,
                            horizon,
                            feat_name,
                            float(shap_values[0, feat_idx]),
                            float(features[feat_idx]),
                            float(base_value),
                        )
                    )

        # Batch insert SHAP values
        if all_shap_records:
            logger.info(f"Saving {len(all_shap_records)} SHAP values to database...")
            await conn.executemany(
                """
                INSERT INTO shap_values (
                    prediction_time, symbol, model_version, horizon,
                    feature_name, shap_value, feature_value, base_value
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (prediction_time, symbol, model_version, horizon, feature_name)
                DO UPDATE SET
                    shap_value = EXCLUDED.shap_value,
                    feature_value = EXCLUDED.feature_value,
                    base_value = EXCLUDED.base_value
                """,
                all_shap_records,
            )
            logger.info("✓ SHAP values saved to database")

    finally:
        await conn.close()


async def main(args):
    """Main function."""
    setup_logging(log_level=args.log_level)

    logger.info("=" * 70)
    logger.info("SHAP Feature Importance Computation")
    logger.info("=" * 70)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Model: {model_path}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Model version: {args.model_version}")
    logger.info(f"Samples to explain: {args.samples}")
    logger.info(f"Background samples: {args.background}")
    logger.info(f"Horizons: {args.horizons}")
    logger.info("")

    # Compute and save SHAP values
    await compute_and_save_shap(
        model_path=model_path,
        symbol=args.symbol,
        model_version=args.model_version,
        num_samples=args.samples,
        num_background=args.background,
        horizons=args.horizons,
    )

    # Query top features
    logger.info("")
    logger.info("=" * 70)
    logger.info("Top Features by Horizon")
    logger.info("=" * 70)

    conn = await asyncpg.connect(
        host=settings.db_host,
        port=settings.db_port,
        database=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )

    try:
        for horizon in args.horizons:
            top_features = await conn.fetch(
                """
                SELECT feature_name, AVG(ABS(shap_value)) as avg_abs_shap
                FROM shap_values
                WHERE symbol = $1
                  AND model_version = $2
                  AND horizon = $3
                GROUP BY feature_name
                ORDER BY avg_abs_shap DESC
                LIMIT 10
                """,
                args.symbol,
                args.model_version,
                horizon,
            )

            logger.info(f"\n{horizon} Horizon:")
            for i, row in enumerate(top_features, 1):
                logger.info(f"  {i}. {row['feature_name']}: {row['avg_abs_shap']:.6f}")

    finally:
        await conn.close()

    logger.info("")
    logger.info("=" * 70)
    logger.info("SHAP computation complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SHAP feature importance for predictions")

    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--model-version", type=str, default="v1.0", help="Model version")
    parser.add_argument("--samples", type=int, default=10, help="Number of predictions to explain")
    parser.add_argument("--background", type=int, default=500, help="Background samples for SHAP")
    parser.add_argument("--horizons", nargs="+", default=["1s", "10s"], help="Horizons to explain")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    asyncio.run(main(args))
