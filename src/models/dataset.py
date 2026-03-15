"""
PyTorch Dataset for LOB data.

Handles loading time-series data from TimescaleDB, creating sequences,
and preparing batches for training the TCN model.
"""

from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from config.logging_config import get_logger

logger = get_logger(__name__)


class LOBDataset(Dataset):
    """
    PyTorch Dataset for Limit Order Book time-series data.

    Loads data from TimescaleDB and creates sliding window sequences
    for temporal modeling. Each sample is a sequence of LOB features
    with corresponding future price targets at multiple horizons.

    Args:
        data (np.ndarray): Feature matrix of shape (num_samples, num_features).
        targets (np.ndarray): Target matrix of shape (num_samples, num_horizons).
        sequence_length (int, optional): Length of input sequences. Defaults to 100.
        stride (int, optional): Stride between consecutive sequences. Defaults to 1.
        normalize (bool, optional): Whether to normalize features. Defaults to True.

    Attributes:
        sequences (np.ndarray): All sequences of shape (num_sequences, seq_len, features).
        targets (np.ndarray): Corresponding targets of shape (num_sequences, num_horizons).
        mean (np.ndarray): Feature means (if normalized).
        std (np.ndarray): Feature standard deviations (if normalized).

    Example:
        >>> # Load data from database
        >>> features, targets = load_data_from_db()
        >>>
        >>> # Create dataset
        >>> dataset = LOBDataset(
        ...     data=features,
        ...     targets=targets,
        ...     sequence_length=100,
        ...     normalize=True
        ... )
        >>>
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=64, shuffle=True)
        >>>
        >>> for x, y in loader:
        ...     predictions = model(x)
        ...     loss = criterion(predictions, y)

    Note:
        Normalization is fit only on the data provided. For train/val/test splits,
        fit normalization on training data and apply to validation/test.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 100,
        stride: int = 1,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        """Initialize LOB dataset."""
        self.sequence_length = sequence_length
        self.stride = stride

        # Store raw data
        self.data = data
        self.targets = targets

        # Normalization parameters
        self.normalize = normalize
        if normalize:
            if mean is not None and std is not None:
                # Use provided normalization (for val/test)
                self.mean = mean
                self.std = std
            else:
                # Fit normalization (for training)
                self.mean = np.mean(data, axis=0)
                self.std = np.std(data, axis=0) + 1e-8  # Avoid division by zero

            # Apply normalization
            self.data_normalized = (data - self.mean) / self.std
        else:
            self.data_normalized = data
            self.mean = None
            self.std = None

        # Create sequences
        self.sequences, self.sequence_targets = self._create_sequences()

        logger.info(
            f"Created LOBDataset: {len(self)} sequences, "
            f"seq_len={sequence_length}, stride={stride}, "
            f"normalized={normalize}"
        )

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences from data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sequences and their targets.

        Example:
            >>> data = [0, 1, 2, 3, 4, 5]
            >>> seq_len = 3, stride = 1
            >>> sequences = [[0,1,2], [1,2,3], [2,3,4]]
            >>> targets = [3, 4, 5]  # next value after each sequence
        """
        sequences = []
        seq_targets = []

        # Slide window across data
        for i in range(0, len(self.data_normalized) - self.sequence_length, self.stride):
            # Extract sequence
            seq = self.data_normalized[i : i + self.sequence_length]

            # Target is at the END of sequence (predict future from current state)
            target_idx = i + self.sequence_length - 1

            # Only include if we have valid target
            if target_idx < len(self.targets):
                sequences.append(seq)
                seq_targets.append(self.targets[target_idx])

        sequences = np.array(sequences, dtype=np.float32)
        seq_targets = np.array(seq_targets, dtype=np.float32)

        logger.debug(f"Created {len(sequences)} sequences from {len(self.data)} samples")

        return sequences, seq_targets

    def __len__(self) -> int:
        """Return number of sequences in dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence and its target.

        Args:
            idx (int): Index of sequence.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - sequence: Shape (seq_len, features)
                - target: Shape (num_horizons,)
        """
        sequence = torch.from_numpy(self.sequences[idx])
        target = torch.from_numpy(self.sequence_targets[idx])

        return sequence, target

    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get normalization parameters (mean, std).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Mean and standard deviation.

        Use case:
            >>> train_dataset = LOBDataset(train_data, train_targets)
            >>> mean, std = train_dataset.get_normalization_params()
            >>>
            >>> # Apply same normalization to validation
            >>> val_dataset = LOBDataset(
            ...     val_data, val_targets, mean=mean, std=std
            ... )
        """
        if self.mean is None or self.std is None:
            raise ValueError("Dataset was not normalized")

        return self.mean, self.std


# =============================================================================
# Data Loading Utilities
# =============================================================================


async def load_data_from_db(
    symbol: str,
    start_time: datetime,
    end_time: datetime,
    db_config: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load LOB data and compute targets from TimescaleDB.

    Args:
        symbol (str): Trading pair symbol.
        start_time (datetime): Start of data range.
        end_time (datetime): End of data range.
        db_config (dict): Database connection configuration.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - features: Shape (num_samples, 43)
            - targets: Shape (num_samples, 5) - future returns at each horizon

    Note:
        Targets are computed as percentage returns:
        return_h = (price_{t+h} - price_t) / price_t * 100
    """
    import asyncpg

    logger.info(f"Loading data for {symbol} from {start_time} to {end_time}")

    # Connect to database
    conn = await asyncpg.connect(**db_config)

    try:
        # Query all columns used to reconstruct the 43 features
        query = """
        SELECT
            time,
            mid_price, spread, spread_bps,
            weighted_mid_price,
            bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
            bid_price_2, bid_volume_2, ask_price_2, ask_volume_2,
            bid_price_3, bid_volume_3, ask_price_3, ask_volume_3,
            bid_price_4, bid_volume_4, ask_price_4, ask_volume_4,
            bid_price_5, bid_volume_5, ask_price_5, ask_volume_5,
            total_bid_volume, total_ask_volume, volume_imbalance,
            price_range, depth_imbalance
        FROM lob_data
        WHERE symbol = $1
          AND time >= $2
          AND time <= $3
        ORDER BY time ASC
        """

        rows = await conn.fetch(query, symbol, start_time, end_time)

        if not rows:
            raise ValueError(f"No data found for {symbol} in specified range")

        logger.info(f"Loaded {len(rows)} snapshots from database")

        # Convert to numpy arrays
        mid_prices = np.array([float(row["mid_price"]) for row in rows])

        # Build the full 43-feature vector matching LOBFeatureEngineering order:
        #   [mid_price, spread, spread_bps, spread_log,
        #    weighted_mid_price,
        #    (bid_price_i, ask_price_i, bid_volume_i, ask_volume_i,
        #     price_diff_i, volume_imbalance_i) for i in 1..5,
        #    total_bid_volume, total_ask_volume, total_volume_imbalance,
        #    bid_ask_volume_ratio,
        #    depth_imbalance, price_range,
        #    accumulated_depth_bid, accumulated_depth_ask]
        def _row_to_features(row):
            mid = float(row["mid_price"])
            spread = float(row["spread"])
            spread_bps = float(row["spread_bps"])
            spread_log = float(np.log(1 + spread))
            weighted_mid = float(row["weighted_mid_price"])

            feats = [mid, spread, spread_bps, spread_log, weighted_mid]

            for lvl in range(1, 6):
                bp = float(row[f"bid_price_{lvl}"])
                ap = float(row[f"ask_price_{lvl}"])
                bv = float(row[f"bid_volume_{lvl}"])
                av = float(row[f"ask_volume_{lvl}"])
                price_diff = ap - bp
                total_vol = bv + av
                vol_imbalance = (bv - av) / total_vol if total_vol > 0 else 0.0
                feats.extend([bp, ap, bv, av, price_diff, vol_imbalance])

            tbv = float(row["total_bid_volume"])
            tav = float(row["total_ask_volume"])
            total_vol = tbv + tav
            total_imbalance = float(row["volume_imbalance"])
            vol_ratio = tbv / tav if tav > 0 else 0.0
            feats.extend([tbv, tav, total_imbalance, vol_ratio])

            depth_imb = float(row["depth_imbalance"])
            price_range = float(row["price_range"])
            # accumulated depth = total volume per side (already queried)
            feats.extend([depth_imb, price_range, tbv, tav])

            return feats

        features = np.array(
            [_row_to_features(row) for row in rows],
            dtype=np.float32,
        )

        # Compute targets (future returns)
        targets = compute_targets(mid_prices, horizons=[1, 5, 10, 30, 60])

        logger.info(f"Features shape: {features.shape}, Targets shape: {targets.shape}")

        return features, targets

    finally:
        await conn.close()


def compute_targets(
    prices: np.ndarray,
    horizons: list[int] = [1, 5, 10, 30, 60],
) -> np.ndarray:
    """
    Compute future return targets at multiple horizons.

    Args:
        prices (np.ndarray): Mid-price time series of shape (num_samples,).
        horizons (list[int]): List of forecast horizons in timesteps.

    Returns:
        np.ndarray: Target returns of shape (num_samples, num_horizons).
            Returns are percentage: (price_future - price_now) / price_now * 100

    Example:
        >>> prices = np.array([100, 101, 102, 103, 104])
        >>> targets = compute_targets(prices, horizons=[1, 2])
        >>> # targets[0] = [(101-100)/100*100, (102-100)/100*100] = [1.0, 2.0]

    Note:
        For samples near the end where future horizon is unavailable,
        target is set to NaN (these samples should be excluded from training).
    """
    num_samples = len(prices)
    num_horizons = len(horizons)

    targets = np.full((num_samples, num_horizons), np.nan, dtype=np.float32)

    for i, horizon in enumerate(horizons):
        for t in range(num_samples - horizon):
            # Compute percentage return
            current_price = prices[t]
            future_price = prices[t + horizon]

            if current_price > 0:  # Avoid division by zero
                return_pct = ((future_price - current_price) / current_price) * 100
                targets[t, i] = return_pct

    # Remove samples with NaN targets (can't predict beyond data end)
    valid_mask = ~np.isnan(targets).any(axis=1)
    targets = targets[valid_mask]

    logger.info(
        f"Computed targets for {len(targets)} samples "
        f"(dropped {num_samples - len(targets)} due to insufficient future data)"
    )

    return targets


def create_train_val_test_split(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    """
    Split data into train/validation/test sets (chronological split).

    Args:
        features (np.ndarray): Feature matrix.
        targets (np.ndarray): Target matrix.
        train_ratio (float): Fraction of data for training.
        val_ratio (float): Fraction of data for validation.
        test_ratio (float): Fraction of data for testing.

    Returns:
        Tuple of (train, val, test), where each is (features, targets).

    Note:
        Split is chronological (not random) to avoid temporal leakage.

        Timeline:
        |←―――――― train (70%) ―――――→|←― val (15%) ―→|←― test (15%) ―→|
        t=0                      t=0.7           t=0.85          t=1.0

    Important:
        Financial data is non-stationary, so random splits would leak
        future information into training. Always use chronological splits.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    num_samples = len(features)

    # Calculate split indices
    train_end = int(num_samples * train_ratio)
    val_end = int(num_samples * (train_ratio + val_ratio))

    # Split features
    X_train = features[:train_end]
    X_val = features[train_end:val_end]
    X_test = features[val_end:]

    # Split targets
    y_train = targets[:train_end]
    y_val = targets[train_end:val_end]
    y_test = targets[val_end:]

    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    """Test dataset creation."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    logger.info("=== Testing LOBDataset ===")

    # Create synthetic data
    num_samples = 1000
    num_features = 43
    num_horizons = 5

    np.random.seed(42)
    features = np.random.randn(num_samples, num_features).astype(np.float32)
    targets = np.random.randn(num_samples, num_horizons).astype(np.float32)

    logger.info(f"Synthetic data: {num_samples} samples, {num_features} features")

    # Create dataset
    dataset = LOBDataset(
        data=features,
        targets=targets,
        sequence_length=100,
        stride=1,
        normalize=True,
    )

    logger.info(f"Dataset created: {len(dataset)} sequences")

    # Test __getitem__
    x, y = dataset[0]
    logger.info(f"Sample shape: x={x.shape}, y={y.shape}")
    logger.info(f"Sample x range: [{x.min():.3f}, {x.max():.3f}]")
    logger.info(f"Sample y: {y}")

    # Test DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch_x, batch_y = next(iter(loader))
    logger.info(f"Batch shape: x={batch_x.shape}, y={batch_y.shape}")

    # Test train/val/test split
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = create_train_val_test_split(
        features, targets
    )

    logger.info(f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Create datasets with proper normalization
    train_dataset = LOBDataset(X_train, y_train, normalize=True)
    mean, std = train_dataset.get_normalization_params()

    val_dataset = LOBDataset(X_val, y_val, normalize=True, mean=mean, std=std)
    test_dataset = LOBDataset(X_test, y_test, normalize=True, mean=mean, std=std)

    logger.info("✓ All tests passed!")
