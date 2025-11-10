"""
SHAP (SHapley Additive exPlanations) explainability for LOB prediction models.

Provides feature importance analysis to understand which LOB features
drive the model's predictions at different time horizons.

SHAP values explain:
- Which features contribute most to predictions
- How feature values influence predictions (positive/negative)
- Feature interactions and their effects

References:
- Lundberg & Lee (2017): "A unified approach to interpreting model predictions"
- SHAP documentation: https://shap.readthedocs.io/
"""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from config.logging_config import get_logger

logger = get_logger(__name__)


# Feature names for LOB data (43 features)
FEATURE_NAMES = [
    # Price features (3)
    "mid_price",
    "spread",
    "spread_bps",
    # Level 1 (4)
    "bid_price_1",
    "bid_volume_1",
    "ask_price_1",
    "ask_volume_1",
    # Level 2 (4)
    "bid_price_2",
    "bid_volume_2",
    "ask_price_2",
    "ask_volume_2",
    # Level 3 (4)
    "bid_price_3",
    "bid_volume_3",
    "ask_price_3",
    "ask_volume_3",
    # Level 4 (4)
    "bid_price_4",
    "bid_volume_4",
    "ask_price_4",
    "ask_volume_4",
    # Level 5 (4)
    "bid_price_5",
    "bid_volume_5",
    "ask_price_5",
    "ask_volume_5",
    # Aggregated features (6)
    "total_bid_volume",
    "total_ask_volume",
    "volume_imbalance",
    "weighted_mid_price",
    "price_range",
    "depth_imbalance",
    # Derived price features (5)
    "price_level_1",
    "price_level_2",
    "price_level_3",
    "price_level_4",
    "price_level_5",
    # Volume ratios (5)
    "volume_ratio_1",
    "volume_ratio_2",
    "volume_ratio_3",
    "volume_ratio_4",
    "volume_ratio_5",
]


class LOBModelWrapper(nn.Module):
    """
    Wrapper for LOB prediction model to work with SHAP.

    SHAP expects models that take numpy arrays and return numpy arrays.
    This wrapper handles the conversion and sequence formatting.

    Args:
        model (nn.Module): Trained LOB prediction model
        sequence_length (int): Input sequence length
        device (torch.device): Device for inference
        horizon_index (int): Which horizon to explain (0-4)

    Example:
        >>> model = load_trained_model()
        >>> wrapper = LOBModelWrapper(model, sequence_length=100, horizon_index=0)
        >>> predictions = wrapper(features)  # Returns predictions for 1s horizon
    """

    def __init__(
        self,
        model: nn.Module,
        sequence_length: int,
        device: torch.device,
        horizon_index: int = 0,
    ):
        """Initialize wrapper."""
        super(LOBModelWrapper, self).__init__()
        self.model = model
        self.sequence_length = sequence_length
        self.device = device
        self.horizon_index = horizon_index

        self.model.eval()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for SHAP.

        Args:
            x: Features of shape (batch, features) or (batch, seq_len, features)

        Returns:
            Predictions for specified horizon of shape (batch,)
        """
        # Debug logging
        logger.debug(f"LOBModelWrapper received input shape: {x.shape}")

        # Ensure x is at least 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        # Convert to tensor
        if len(x.shape) == 2:
            # (batch, features) -> need to create sequences
            # For simplicity, repeat the same features sequence_length times
            x = np.repeat(x[:, np.newaxis, :], self.sequence_length, axis=1)

        # Ensure correct dtype
        x = x.astype(np.float32)

        logger.debug(f"After reshaping, shape: {x.shape}")

        x_tensor = torch.from_numpy(x).float().to(self.device)

        # Predict
        with torch.no_grad():
            predictions = self.model(x_tensor)  # (batch, num_horizons)

        # Extract specified horizon
        predictions = predictions[:, self.horizon_index]

        return predictions.cpu().numpy()


class SHAPExplainer:
    """
    SHAP explainer for LOB prediction models.

    Provides feature importance analysis using SHAP values, which measure
    the marginal contribution of each feature to the prediction.

    Args:
        model (nn.Module): Trained model
        background_data (np.ndarray): Background dataset for SHAP
        sequence_length (int): Input sequence length
        device (str): Device for inference

    Attributes:
        explainers (dict): SHAP explainers for each horizon
        feature_names (list): Names of features

    Example:
        >>> explainer = SHAPExplainer(model, background_data)
        >>> shap_values = explainer.explain(test_data, horizon='1s')
        >>> explainer.plot_summary(shap_values, horizon='1s')
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        sequence_length: int = 100,
        device: str = "mps",
    ):
        """Initialize SHAP explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Install with: pip install shap")

        self.model = model
        self.background_data = background_data
        self.sequence_length = sequence_length
        self.device = torch.device(device)
        self.feature_names = FEATURE_NAMES

        # Create explainers for each horizon
        self.explainers = {}
        self.horizon_names = ["1s", "5s", "10s", "30s", "60s"]

        logger.info("Initializing SHAP explainers for all horizons...")
        for i, horizon in enumerate(self.horizon_names):
            logger.info(f"  Creating explainer for {horizon} horizon...")

            # Create model wrapper for this horizon
            wrapped_model = LOBModelWrapper(
                model=model,
                sequence_length=sequence_length,
                device=self.device,
                horizon_index=i,
            )

            # Create SHAP explainer (using KernelExplainer for model-agnostic approach)
            # Note: This is slower but works with any model,
            # min(100, len(background_data)) to avoid index errors
            num_background = min(100, len(background_data))
            self.explainers[horizon] = shap.KernelExplainer(
                wrapped_model.forward,
                background_data[:num_background],
            )

        logger.info("✓ SHAP explainers initialized")

    def explain(
        self,
        data: np.ndarray,
        horizon: str = "1s",
        nsamples: int = 100,
    ) -> np.ndarray:
        """
        Compute SHAP values for given data.

        Args:
            data: Input data of shape (n_samples, n_features)
            horizon: Which horizon to explain ('1s', '5s', etc.)
            nsamples: Number of samples for SHAP computation (trade-off: accuracy vs speed)

        Returns:
            SHAP values of shape (n_samples, n_features)
        """
        if horizon not in self.explainers:
            raise ValueError(f"Invalid horizon: {horizon}")

        logger.info(f"Computing SHAP values for {len(data)} samples at {horizon} horizon...")
        start_time = time.time()

        explainer = self.explainers[horizon]
        shap_values = explainer.shap_values(data, nsamples=nsamples)

        elapsed = time.time() - start_time
        logger.info(f"✓ SHAP computation complete ({elapsed:.1f}s)")

        return shap_values

    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most important features based on mean absolute SHAP values.

        Args:
            shap_values: SHAP values from explain()
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples, sorted by importance
        """
        # Mean absolute SHAP value for each feature
        importance = np.abs(shap_values).mean(axis=0)

        # Sort by importance
        indices = np.argsort(importance)[::-1][:top_k]

        return [(self.feature_names[i], importance[i]) for i in indices]

    def plot_summary(
        self,
        shap_values: np.ndarray,
        data: np.ndarray,
        horizon: str,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Generate SHAP summary plot.

        Args:
            shap_values: SHAP values from explain()
            data: Corresponding input data
            horizon: Horizon name for title
            output_path: Path to save plot (if None, displays interactively)
        """
        try:
            import matplotlib.pyplot as plt

            logger.info(f"Generating SHAP summary plot for {horizon} horizon...")

            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                data,
                feature_names=self.feature_names,
                show=False,
            )
            plt.title(f"SHAP Feature Importance - {horizon} Horizon")
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"✓ Plot saved to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")

    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        data: np.ndarray,
        sample_index: int,
        horizon: str,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Generate SHAP waterfall plot for a single prediction.

        Shows how each feature contributes to moving the prediction from
        the base value (expected value) to the final prediction.

        Args:
            shap_values: SHAP values from explain()
            data: Corresponding input data
            sample_index: Which sample to visualize
            horizon: Horizon name for title
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            logger.info(f"Generating waterfall plot for sample {sample_index}...")

            # Create explanation object
            explainer = self.explainers[horizon]
            explanation = shap.Explanation(
                values=shap_values[sample_index],
                base_values=explainer.expected_value,
                data=data[sample_index],
                feature_names=self.feature_names,
            )

            # Create waterfall plot
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(explanation, show=False)
            plt.title(f"SHAP Waterfall Plot - {horizon} Horizon (Sample {sample_index})")
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"✓ Plot saved to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")

    def plot_dependence(
        self,
        shap_values: np.ndarray,
        data: np.ndarray,
        feature_name: str,
        horizon: str,
        output_path: Optional[Path] = None,
    ) -> None:
        """
        Generate SHAP dependence plot for a specific feature.

        Shows how the feature value affects the SHAP value (prediction impact).

        Args:
            shap_values: SHAP values from explain()
            data: Corresponding input data
            feature_name: Which feature to visualize
            horizon: Horizon name for title
            output_path: Path to save plot
        """
        try:
            import matplotlib.pyplot as plt

            if feature_name not in self.feature_names:
                raise ValueError(f"Invalid feature name: {feature_name}")

            feature_index = self.feature_names.index(feature_name)

            logger.info(f"Generating dependence plot for {feature_name}...")

            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_index,
                shap_values,
                data,
                feature_names=self.feature_names,
                show=False,
            )
            plt.title(f"SHAP Dependence - {feature_name} ({horizon} Horizon)")
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                logger.info(f"✓ Plot saved to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")


def create_shap_explainer(
    model_path: Path,
    background_data: np.ndarray,
    sequence_length: int = 100,
    device: str = "mps",
) -> SHAPExplainer:
    """
    Factory function to create SHAP explainer from model checkpoint.

    Args:
        model_path: Path to trained model checkpoint
        background_data: Background dataset for SHAP
        sequence_length: Input sequence length
        device: Device for inference

    Returns:
        Initialized SHAPExplainer

    Example:
        >>> explainer = create_shap_explainer(
        ...     Path('models/best_model.pth'),
        ...     background_data,
        ... )
    """
    from src.models.tcn import LOBPricePredictionTCN

    # Load model
    logger.info(f"Loading model from {model_path}")
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj)

    model = LOBPricePredictionTCN(
        input_size=43,
        num_channels=[128, 128, 256, 256],
        kernel_size=3,
        dropout=0.2,
        num_horizons=5,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device_obj)
    model.eval()

    logger.info(f"Model loaded (epoch {checkpoint['epoch']})")

    # Create explainer
    explainer = SHAPExplainer(
        model=model,
        background_data=background_data,
        sequence_length=sequence_length,
        device=device,
    )

    return explainer


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    """Test SHAP explainer with synthetic data."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    if not SHAP_AVAILABLE:
        logger.error("SHAP not installed. Install with: pip install shap")
        exit(1)

    logger.info("=== Testing SHAP Explainer ===")

    # Create synthetic data
    logger.info("Creating synthetic data...")
    n_background = 500
    n_test = 10
    n_features = 43

    background_data = np.random.randn(n_background, n_features).astype(np.float32)
    test_data = np.random.randn(n_test, n_features).astype(np.float32)

    # Create dummy model
    logger.info("Creating dummy model...")
    from src.models.tcn import LOBPricePredictionTCN

    model = LOBPricePredictionTCN(
        input_size=43,
        num_channels=[64, 64],  # Smaller for testing
        kernel_size=3,
        dropout=0.2,
        num_horizons=5,
    )
    model.eval()

    # Create explainer
    logger.info("Creating SHAP explainer...")
    explainer = SHAPExplainer(
        model=model,
        background_data=background_data,
        sequence_length=10,  # Shorter for testing
        device="cpu",
    )

    # Compute SHAP values
    logger.info("Computing SHAP values...")
    shap_values = explainer.explain(
        test_data,
        horizon="1s",
        nsamples=50,  # Fewer samples for speed
    )

    # Get feature importance
    logger.info("Top 5 most important features:")
    top_features = explainer.get_feature_importance(shap_values, top_k=5)
    for i, (feature, importance) in enumerate(top_features, 1):
        logger.info(f"  {i}. {feature}: {importance:.4f}")

    logger.info("✓ SHAP explainer test complete!")
