"""
Temporal Convolutional Network (TCN) for LOB price prediction.

This module implements a TCN architecture optimized for time-series prediction
from limit order book features. The model uses dilated causal convolutions
with residual connections to capture temporal dependencies at multiple scales.

References:
    - Bai et al. (2018): "An Empirical Evaluation of Generic Convolutional
      and Recurrent Networks for Sequence Modeling"
    - Zhang et al. (2019): "DeepLOB: Deep Convolutional Neural Networks
      for Limit Order Books"
"""

from typing import List

import torch
import torch.nn as nn

# import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from config.logging_config import get_logger

logger = get_logger(__name__)


class TemporalBlock(nn.Module):
    """
    Single temporal block with dilated convolutions and residual connection.

    A temporal block consists of two dilated causal convolutional layers
    with weight normalization, dropout, and a residual connection. This is
    the fundamental building block of the TCN.

    Args:
        n_inputs (int): Number of input channels.
        n_outputs (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        dilation (int): Dilation factor for the convolution.
        dropout (float): Dropout probability.

    Attributes:
        conv1 (nn.Conv1d): First dilated causal convolution.
        conv2 (nn.Conv1d): Second dilated causal convolution.
        downsample (nn.Conv1d): Optional 1x1 conv for residual if channels change.
        relu (nn.ReLU): Activation function.
        dropout (nn.Dropout): Dropout layer.

    Note:
        Uses weight normalization for training stability and causal padding
        to prevent information leakage from future timesteps.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        """Initialize temporal block."""
        super(TemporalBlock, self).__init__()

        # Calculate padding for causal convolution
        # Ensures output has same length as input
        self.padding = (kernel_size - 1) * dilation

        # First convolutional layer with weight norm
        conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
        )
        self.conv1 = weight_norm(conv1)

        # Second convolutional layer
        conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
        )
        self.conv2 = weight_norm(conv2)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection: 1x1 conv if channel dimensions change
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # Initialize weights
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize convolutional layer weights."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, seq_len).

        Returns:
            torch.Tensor: Output tensor of same shape with residual added.

        Note:
            Applies causal convolutions, so output at time t only depends
            on inputs up to time t (no future information).
        """
        # First conv block
        out = self.conv1(x)
        # Remove future-looking padding (causal)
        out = out[:, :, : -self.padding] if self.padding != 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, : -self.padding] if self.padding != 0 else out
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.

    A TCN consists of a stack of temporal blocks with exponentially
    increasing dilation factors. This allows the network to have a very
    large receptive field while maintaining computational efficiency.

    Args:
        num_inputs (int): Number of input features.
        num_channels (List[int]): List of channel sizes for each layer.
            Length determines network depth.
        kernel_size (int, optional): Convolutional kernel size. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.2.

    Attributes:
        network (nn.Sequential): Sequential container of temporal blocks.

    Example:
        >>> tcn = TemporalConvNet(
        ...     num_inputs=43,
        ...     num_channels=[128, 128, 256, 256],
        ...     kernel_size=3,
        ...     dropout=0.2
        ... )
        >>> x = torch.randn(32, 43, 100)  # (batch, features, seq_len)
        >>> out = tcn(x)
        >>> print(out.shape)  # (32, 256, 100)

    Note:
        The receptive field grows exponentially with depth:
        receptive_field = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(depth))
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        """Initialize TCN."""
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i  # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, features, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch, channels[-1], seq_len).
        """
        return self.network(x)


class LOBPricePredictionTCN(nn.Module):
    """
    Complete TCN model for LOB-based price prediction.

    This model takes a sequence of LOB features and predicts future price
    movements at multiple time horizons. It uses a TCN backbone followed
    by dense layers for multi-horizon prediction.

    The architecture is sized for ambitious performance while maintaining
    training efficiency on consumer hardware (M2 MacBook). With ~920K
    parameters, it has sufficient capacity for complex pattern learning
    while avoiding overfitting on moderately-sized datasets.

    Args:
        input_size (int): Number of input features (e.g., 43 for LOB features).
        num_channels (List[int], optional): TCN layer channels.
            Defaults to [128, 128, 256, 256] for ~920K params.
        kernel_size (int, optional): Convolutional kernel size. Defaults to 3.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        num_horizons (int, optional): Number of prediction horizons. Defaults to 5.

    Attributes:
        tcn (TemporalConvNet): TCN backbone.
        fc (nn.Linear): Final dense layer for predictions.

    Example:
        >>> model = LOBPricePredictionTCN(input_size=43, num_horizons=5)
        >>> x = torch.randn(32, 100, 43)  # (batch, seq_len, features)
        >>> predictions = model(x)
        >>> print(predictions.shape)  # (32, 5) - one prediction per horizon

    Note:
        Input should be normalized features. Model outputs raw predictions
        that need to be denormalized for interpretation. The model predicts
        price changes (returns), not absolute prices.
    """

    def __init__(
        self,
        input_size: int,
        num_channels: List[int] = [128, 128, 256, 256],  # ~920K params
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_horizons: int = 5,
    ):
        """Initialize LOB price prediction TCN."""
        super(LOBPricePredictionTCN, self).__init__()

        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_horizons = num_horizons

        # TCN backbone
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Output layer: map from last TCN channel to prediction horizons
        self.fc = nn.Linear(num_channels[-1], num_horizons)

        # Calculate and log model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            f"Initialized LOBPricePredictionTCN: "
            f"input={input_size}, channels={num_channels}, "
            f"horizons={num_horizons}"
        )
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Receptive field: {self.get_receptive_field()} timesteps")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for price prediction.

        Args:
            x (torch.Tensor): Input features of shape (batch, seq_len, features).

        Returns:
            torch.Tensor: Predictions of shape (batch, num_horizons).
                Each column represents predicted price change for a different
                time horizon (e.g., [1s, 5s, 10s, 30s, 60s]).

        Note:
            Uses the last timestep's representation for prediction, as we're
            predicting future prices from the current state.
        """
        # TCN expects (batch, features, seq_len)
        x = x.transpose(1, 2)

        # Pass through TCN
        tcn_out = self.tcn(x)  # (batch, channels, seq_len)

        # Use last timestep for prediction
        last_timestep = tcn_out[:, :, -1]  # (batch, channels)

        # Predict for all horizons
        predictions = self.fc(last_timestep)  # (batch, num_horizons)

        return predictions

    def get_receptive_field(self) -> int:
        """
        Calculate the receptive field of the TCN.

        Returns:
            int: Number of timesteps the model can "see" backwards.

        Note:
            Useful for determining minimum sequence length for training.
            With default settings (kernel_size=3, 4 layers), receptive
            field is 61 timesteps.
        """
        depth = len(self.num_channels)
        receptive_field = 1 + 2 * (self.kernel_size - 1) * sum(2**i for i in range(depth))
        return receptive_field

    def count_parameters(self) -> dict:
        """
        Get detailed parameter count breakdown.

        Returns:
            dict: Dictionary with parameter statistics.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count by component
        tcn_params = sum(p.numel() for p in self.tcn.parameters())
        fc_params = sum(p.numel() for p in self.fc.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "tcn_backbone": tcn_params,
            "output_layer": fc_params,
        }


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    """Test TCN model architecture."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    logger.info("=" * 70)
    logger.info("Testing LOB Price Prediction TCN Model")
    logger.info("=" * 70)

    # Model parameters
    input_size = 43  # Number of LOB features
    seq_len = 100  # Lookback window
    batch_size = 32
    num_horizons = 5

    # Create model
    model = LOBPricePredictionTCN(
        input_size=input_size,
        num_channels=[128, 128, 256, 256],  # ~920K params
        kernel_size=3,
        dropout=0.2,
        num_horizons=num_horizons,
    )

    # Print detailed parameter info
    param_stats = model.count_parameters()
    logger.info("\nParameter Breakdown:")
    logger.info(f"  TCN Backbone: {param_stats['tcn_backbone']:,} params")
    logger.info(f"  Output Layer: {param_stats['output_layer']:,} params")
    logger.info(f"  Total: {param_stats['total']:,} params")

    # Test forward pass
    logger.info("\nTesting Forward Pass:")
    logger.info(f"  Input shape: (batch={batch_size}, seq_len={seq_len}, features={input_size})")

    x = torch.randn(batch_size, seq_len, input_size)

    with torch.no_grad():
        predictions = model(x)

    logger.info(f"  Output shape: {predictions.shape}")
    logger.info(f"  Sample predictions (first batch): {predictions[0].numpy()}")

    # Memory estimate
    param_size_mb = param_stats["total"] * 4 / (1024 * 1024)  # float32 = 4 bytes
    logger.info(f"\nModel size: ~{param_size_mb:.1f} MB")

    # Test on MPS (Apple Silicon) if available
    if torch.backends.mps.is_available():
        logger.info("\nTesting on Apple Silicon (MPS):")
        model_mps = model.to("mps")
        x_mps = x.to("mps")

        import time

        start = time.time()
        with torch.no_grad():
            _ = model_mps(x_mps)
        elapsed = (time.time() - start) * 1000

        logger.info(f"  Inference time: {elapsed:.2f}ms for batch of {batch_size}")
        logger.info(f"  Per sample: {elapsed/batch_size:.2f}ms")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Model architecture test passed!")
    logger.info("=" * 70)
