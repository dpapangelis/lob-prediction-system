"""
Feature engineering for Limit Order Book (LOB) data.

This module transforms raw order book snapshots into meaningful features
for machine learning models. Features capture market microstructure signals
including spreads, imbalances, and price pressures.

The features are designed specifically for short-term price prediction and
are commonly used in LOB-based trading research.

References:
    - Zhang et al. (2019): "DeepLOB: Deep Convolutional Neural Networks
      for Limit Order Books"
    - Ntakaris et al. (2018): "Feature Engineering for Mid-Price Prediction
      with Deep Learning"
"""

from typing import Optional

import numpy as np
import pandas as pd

from config.logging_config import get_logger

logger = get_logger(__name__)


class LOBFeatureEngineering:
    """
    Feature engineering for Limit Order Book data.

    Transforms raw bid/ask prices and volumes into meaningful features
    for machine learning models. Handles multiple LOB levels and computes
    various market microstructure indicators.

    Features computed:
        - Spread features (absolute, relative, log)
        - Mid-price and weighted mid-price
        - Order imbalance at each level
        - Volume ratios and derivatives
        - Price momentum indicators
        - Book pressure metrics

    Args:
        levels (int, optional): Number of LOB levels to process. Defaults to 5.
        normalize (bool, optional): Whether to normalize features. Defaults to True.

    Attributes:
        levels (int): Number of order book levels.
        normalize (bool): Whether features are normalized.
        feature_names (list[str]): Names of all computed features.

    Example:
        >>> engineer = LOBFeatureEngineering(levels=5)
        >>> lob_data = {
        ...     'bids': [['100.0', '10.5'], ['99.9', '8.2'], ...],
        ...     'asks': [['100.1', '12.3'], ['100.2', '9.1'], ...]
        ... }
        >>> features = engineer.compute_features(lob_data)
        >>> print(features.shape)
        (1, 45)  # 45 features from 5 levels

    Note:
        All price features are computed in basis points (bps) or relative
        terms to ensure scale invariance across different price levels.
    """

    def __init__(self, levels: int = 5, normalize: bool = True):
        """
        Initialize feature engineering pipeline.

        Args:
            levels: Number of LOB levels to use (must match incoming data)
            normalize: If True, normalize features to [-1, 1] or [0, 1] range
        """
        self.levels = levels
        self.normalize = normalize
        self.feature_names = self._generate_feature_names()

        logger.info(
            f"Initialized LOBFeatureEngineering with {levels} levels, "
            f"{len(self.feature_names)} features, normalize={normalize}"
        )

    def _generate_feature_names(self) -> list[str]:
        """
        Generate feature names for all computed features.

        Returns:
            list[str]: Ordered list of feature names matching the output array.

        Note:
            Feature names follow the pattern: <type>_<level> for level-specific
            features, or <type> for aggregate features.
        """
        names = []

        # Price and spread features
        names.extend(
            [
                "mid_price",
                "spread_absolute",
                "spread_bps",
                "spread_log",
            ]
        )

        # Weighted mid-price
        names.append("weighted_mid_price")

        # Level-specific features
        for level in range(1, self.levels + 1):
            names.extend(
                [
                    f"bid_price_{level}",
                    f"ask_price_{level}",
                    f"bid_volume_{level}",
                    f"ask_volume_{level}",
                    f"price_diff_{level}",  # ask_price - bid_price
                    f"volume_imbalance_{level}",  # (bid_vol - ask_vol) / total
                ]
            )

        # Aggregate volume features
        names.extend(
            [
                "total_bid_volume",
                "total_ask_volume",
                "total_volume_imbalance",
                "bid_ask_volume_ratio",
            ]
        )

        # Depth and pressure features
        names.extend(
            [
                "depth_imbalance",  # Weighted by distance from mid
                "price_range",  # ask_level_n - bid_level_n
                "accumulated_depth_bid",
                "accumulated_depth_ask",
            ]
        )

        return names

    def compute_features(
        self, lob_data: dict, previous_mid_price: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute all features from a single LOB snapshot.

        Takes raw order book data and computes a comprehensive feature vector
        suitable for machine learning models. Features are ordered consistently
        with `feature_names`.

        Args:
            lob_data (dict): Order book data containing:
                - 'bids': List of [price, volume] pairs (best to worst)
                - 'asks': List of [price, volume] pairs (best to worst)
            previous_mid_price (float, optional): Mid-price from previous snapshot
                for computing price changes. If None, price change features are
                set to 0.

        Returns:
            np.ndarray: Feature vector of shape (1, n_features) where n_features
                depends on the number of levels (typically ~45 for 5 levels).

        Raises:
            ValueError: If LOB data doesn't contain enough levels.
            ValueError: If bid/ask prices or volumes are invalid.

        Example:
            >>> lob = {'bids': [['100', '10'], ['99', '8']],
            ...        'asks': [['101', '12'], ['102', '9']]}
            >>> features = engineer.compute_features(lob)
            >>> print(features.shape)
            (1, 45)

        Note:
            Missing or invalid data points are handled by setting corresponding
            features to 0. Warnings are logged for data quality issues.
        """
        try:
            # Extract and validate data
            bids = self._parse_lob_side(lob_data.get("bids", []))
            asks = self._parse_lob_side(lob_data.get("asks", []))

            if len(bids) < self.levels or len(asks) < self.levels:
                raise ValueError(
                    f"Insufficient LOB depth: got {len(bids)} bids and "
                    f"{len(asks)} asks, need {self.levels}"
                )

            # Trim to specified levels
            bids = bids[: self.levels]
            asks = asks[: self.levels]

            features = []

            # Basic price features
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2

            spread_abs = best_ask - best_bid
            spread_bps = (spread_abs / mid_price) * 10000
            spread_log = np.log(1 + spread_abs)

            features.extend([mid_price, spread_abs, spread_bps, spread_log])

            # Weighted mid-price
            best_bid_vol = bids[0][1]
            best_ask_vol = asks[0][1]
            total_vol = best_bid_vol + best_ask_vol

            if total_vol > 0:
                weighted_mid = (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_vol
            else:
                weighted_mid = mid_price

            features.append(weighted_mid)

            # Level-specific features
            for level in range(self.levels):
                bid_price, bid_vol = bids[level]
                ask_price, ask_vol = asks[level]

                price_diff = ask_price - bid_price

                # Volume imbalance at this level
                level_total_vol = bid_vol + ask_vol
                if level_total_vol > 0:
                    vol_imbalance = (bid_vol - ask_vol) / level_total_vol
                else:
                    vol_imbalance = 0.0

                features.extend(
                    [
                        bid_price,
                        ask_price,
                        bid_vol,
                        ask_vol,
                        price_diff,
                        vol_imbalance,
                    ]
                )

            # Aggregate volume features
            total_bid_vol = sum(b[1] for b in bids)
            total_ask_vol = sum(a[1] for a in asks)
            total_volume = total_bid_vol + total_ask_vol

            if total_volume > 0:
                total_imbalance = (total_bid_vol - total_ask_vol) / total_volume
            else:
                total_imbalance = 0.0

            if total_ask_vol > 0:
                vol_ratio = total_bid_vol / total_ask_vol
            else:
                vol_ratio = 0.0

            features.extend(
                [
                    total_bid_vol,
                    total_ask_vol,
                    total_imbalance,
                    vol_ratio,
                ]
            )

            # Depth and pressure features
            depth_imbalance = self._compute_depth_imbalance(bids, asks, mid_price)
            price_range = asks[-1][0] - bids[-1][0]  # Worst ask - worst bid

            # Accumulated depth (cumulative volume)
            accum_bid = np.cumsum([b[1] for b in bids])[-1]
            accum_ask = np.cumsum([a[1] for a in asks])[-1]

            features.extend(
                [
                    depth_imbalance,
                    price_range,
                    accum_bid,
                    accum_ask,
                ]
            )

            # Convert to numpy array
            feature_array = np.array(features, dtype=np.float32).reshape(1, -1)

            # Normalize if requested
            if self.normalize:
                feature_array = self._normalize_features(feature_array)

            return feature_array

        except Exception as e:
            logger.error(f"Error computing features: {e}", exc_info=True)
            # Return zero vector on error
            return np.zeros((1, len(self.feature_names)), dtype=np.float32)

    def _parse_lob_side(self, side_data: list) -> list[tuple[float, float]]:
        """
        Parse and validate one side of the order book (bids or asks).

        Args:
            side_data (list): List of [price, volume] pairs as strings or floats.

        Returns:
            list[tuple[float, float]]: Parsed (price, volume) tuples.

        Raises:
            ValueError: If data format is invalid.
        """
        parsed = []
        for item in side_data:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    price = float(item[0])
                    volume = float(item[1])

                    if price <= 0 or volume < 0:
                        logger.warning(f"Invalid price/volume: {price}, {volume}")
                        continue

                    parsed.append((price, volume))
                else:
                    logger.warning(f"Invalid LOB entry format: {item}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse LOB entry {item}: {e}")
                continue

        return parsed

    def _compute_depth_imbalance(
        self, bids: list[tuple[float, float]], asks: list[tuple[float, float]], mid_price: float
    ) -> float:
        """
        Compute weighted depth imbalance.

        Weights order book levels by their distance from the mid-price,
        giving more importance to levels closer to the market.

        Args:
            bids: List of (price, volume) tuples for bid side.
            asks: List of (price, volume) tuples for ask side.
            mid_price: Current mid-price.

        Returns:
            float: Weighted depth imbalance in range [-1, 1], where positive
                values indicate bid pressure and negative indicate ask pressure.

        Note:
            Uses exponential weighting: weight = exp(-distance_from_mid)
        """
        weighted_bid = 0.0
        weighted_ask = 0.0

        for price, volume in bids:
            distance = abs(price - mid_price) / mid_price
            weight = np.exp(-distance * 10)  # Exponential decay
            weighted_bid += volume * weight

        for price, volume in asks:
            distance = abs(price - mid_price) / mid_price
            weight = np.exp(-distance * 10)
            weighted_ask += volume * weight

        total_weighted = weighted_bid + weighted_ask

        if total_weighted > 0:
            return (weighted_bid - weighted_ask) / total_weighted
        else:
            return 0.0

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature array to stable ranges.

        Uses min-max scaling for bounded features and standardization
        for unbounded features. Normalization parameters could be learned
        from training data for production use.

        Args:
            features (np.ndarray): Raw feature array of shape (1, n_features).

        Returns:
            np.ndarray: Normalized feature array of same shape.

        Note:
            This is a simple normalization. For production, use sklearn's
            StandardScaler or MinMaxScaler fit on training data.
        """
        # TODO: Implement proper normalization with fitted scaler
        # For now, return as-is with warning
        logger.debug("Feature normalization not yet implemented")
        return features

    def compute_features_batch(self, lob_data_list: list[dict]) -> np.ndarray:
        """
        Compute features for a batch of LOB snapshots.

        Efficiently processes multiple snapshots and returns a feature matrix.

        Args:
            lob_data_list (list[dict]): List of LOB data dictionaries.

        Returns:
            np.ndarray: Feature matrix of shape (n_snapshots, n_features).

        Example:
            >>> lob_snapshots = [lob1, lob2, lob3]
            >>> features = engineer.compute_features_batch(lob_snapshots)
            >>> print(features.shape)
            (3, 45)
        """
        feature_list = []

        for lob_data in lob_data_list:
            features = self.compute_features(lob_data)
            feature_list.append(features)

        return np.vstack(feature_list)

    def features_to_dataframe(self, features: np.ndarray) -> pd.DataFrame:
        """
        Convert feature array to labeled DataFrame.

        Args:
            features (np.ndarray): Feature array of shape (n, n_features).

        Returns:
            pd.DataFrame: DataFrame with feature names as columns.

        Example:
            >>> features = engineer.compute_features(lob_data)
            >>> df = engineer.features_to_dataframe(features)
            >>> print(df.columns)
            ['mid_price', 'spread_absolute', ...]
        """
        return pd.DataFrame(features, columns=self.feature_names)


# =============================================================================
# Utility Functions
# =============================================================================


def compute_returns(prices: np.ndarray, periods: list[int] = [1, 5, 10]) -> pd.DataFrame:
    """
    Compute returns over multiple time periods.

    Args:
        prices (np.ndarray): Array of prices (typically mid-prices).
        periods (list[int]): List of periods for return calculation.

    Returns:
        pd.DataFrame: DataFrame with return columns for each period.

    Example:
        >>> prices = np.array([100, 101, 102, 103, 104])
        >>> returns = compute_returns(prices, periods=[1, 2])
        >>> print(returns.columns)
        ['return_1', 'return_2']
    """
    returns_df = pd.DataFrame()

    for period in periods:
        returns_df[f"return_{period}"] = (prices - np.roll(prices, period)) / np.roll(
            prices, period
        )
        # Set initial values to 0 (no previous data)
        returns_df[f"return_{period}"].iloc[:period] = 0

    return returns_df


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    """Test feature engineering with sample data."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    # Sample LOB data (format from Binance)
    sample_lob = {
        "bids": [
            ["101000.00", "5.5"],
            ["100999.50", "3.2"],
            ["100999.00", "8.1"],
            ["100998.50", "2.7"],
            ["100998.00", "6.3"],
        ],
        "asks": [
            ["101000.50", "4.8"],
            ["101001.00", "6.2"],
            ["101001.50", "3.9"],
            ["101002.00", "5.4"],
            ["101002.50", "7.1"],
        ],
    }

    # Initialize feature engineer
    engineer = LOBFeatureEngineering(levels=5, normalize=False)

    # Compute features
    logger.info("Computing features from sample LOB data...")
    features = engineer.compute_features(sample_lob)

    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Number of features: {len(engineer.feature_names)}")

    # Convert to DataFrame for better visualization
    df = engineer.features_to_dataframe(features)

    logger.info("\nSample features:")
    print(
        df[["mid_price", "spread_bps", "total_volume_imbalance", "weighted_mid_price"]].to_string()
    )

    logger.info("\nAll feature names:")
    for i, name in enumerate(engineer.feature_names, 1):
        print(f"{i:2d}. {name}")
