"""
Custom loss functions for financial time series prediction.

Implements specialized loss functions beyond standard MSE that account for
financial-specific considerations like directional accuracy, asymmetric costs,
and risk-adjusted returns.
"""

# from typing import Optional

import torch
import torch.nn as nn

from config.logging_config import get_logger

# import torch.nn.functional as F


logger = get_logger(__name__)


class DirectionalLoss(nn.Module):
    """
    Loss function that penalizes incorrect direction predictions.

    Combines MSE with directional accuracy penalty to encourage the model
    to predict the correct sign (up/down) in addition to magnitude.

    Args:
        alpha (float): Weight for directional component. Default: 0.5
            - alpha=0: Pure MSE (magnitude only)
            - alpha=1: Pure directional (sign only)
            - alpha=0.5: Balanced

    Mathematical Formulation:
        L = (1 - alpha) * MSE + alpha * DirectionalPenalty

        DirectionalPenalty = mean(1 - sign(pred) * sign(true))
        - Correct direction: penalty = 0
        - Wrong direction: penalty = 2

    Example:
        >>> criterion = DirectionalLoss(alpha=0.3)
        >>> predictions = torch.tensor([[0.5, -0.2, 0.1]])
        >>> targets = torch.tensor([[0.3, 0.1, 0.2]])
        >>> loss = criterion(predictions, targets)

    Reference:
        Dixon, M. F. (2018). Sequence classification of the limit order book
        using recurrent neural networks. *Journal of Computational Science*, 24, 277-286.
    """

    def __init__(self, alpha: float = 0.5):
        """Initialize directional loss."""
        super(DirectionalLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute directional loss.

        Args:
            predictions: Predicted returns of shape (batch, num_horizons)
            targets: Actual returns of shape (batch, num_horizons)

        Returns:
            Scalar loss value
        """
        # MSE component
        mse_loss = self.mse(predictions, targets)

        # Directional component
        # sign(pred) * sign(true) = +1 if same direction, -1 if opposite
        direction_agreement = torch.sign(predictions) * torch.sign(targets)

        # Convert to penalty: 1 - agreement
        # Same direction: 1 - 1 = 0 (no penalty)
        # Opposite: 1 - (-1) = 2 (high penalty)
        directional_penalty = 1 - direction_agreement
        directional_loss = directional_penalty.mean()

        # Combined loss
        total_loss = (1 - self.alpha) * mse_loss + self.alpha * directional_loss

        return total_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss that penalizes over-predictions more than under-predictions.

    In trading, over-predicting returns (predicting large gains that don't materialize)
    can be more costly than under-predicting. This loss reflects that asymmetry.

    Args:
        beta (float): Asymmetry parameter. Default: 2.0
            - beta > 1: Penalize over-predictions more
            - beta < 1: Penalize under-predictions more
            - beta = 1: Symmetric (equivalent to MAE)

    Mathematical Formulation:
        For each prediction error e = pred - true:
        L(e) = |e|^beta  if e > 0 (over-prediction)
               |e|       if e <= 0 (under-prediction)

    Example:
        >>> criterion = AsymmetricLoss(beta=2.0)
        >>> # Over-prediction penalized quadratically
        >>> # Under-prediction penalized linearly

    Use Case:
        When false positives (predicting profit when loss occurs) are more
        costly than false negatives (missing profitable opportunities).

    Reference:
        Christoffersen, P. F., & Diebold, F. X. (1997). Optimal prediction under
        asymmetric loss. *Econometric Theory*, 13(6), 808-817.
    """

    def __init__(self, beta: float = 2.0):
        """Initialize asymmetric loss."""
        super(AsymmetricLoss, self).__init__()
        self.beta = beta

        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute asymmetric loss.

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            Scalar loss value
        """
        errors = predictions - targets

        # Over-predictions (positive errors)
        over_pred_mask = errors > 0
        over_pred_loss = torch.abs(errors[over_pred_mask]) ** self.beta

        # Under-predictions (negative errors)
        under_pred_mask = errors <= 0
        under_pred_loss = torch.abs(errors[under_pred_mask])

        # Combine
        total_loss = (over_pred_loss.sum() + under_pred_loss.sum()) / errors.numel()

        return total_loss


class SharpeRatioLoss(nn.Module):
    """
    Loss function based on Sharpe ratio (risk-adjusted returns).

    Instead of minimizing prediction error, this loss maximizes the Sharpe ratio
    of returns from a hypothetical trading strategy based on predictions.

    Strategy: Go long when predicted return > 0, short when < 0

    Args:
        risk_free_rate (float): Risk-free rate (annualized). Default: 0.0

    Mathematical Formulation:
        Returns[t] = sign(pred[t]) * actual[t]
        Sharpe = (mean(Returns) - risk_free_rate) / std(Returns)
        Loss = -Sharpe  (minimize negative Sharpe = maximize Sharpe)

    Example:
        >>> criterion = SharpeRatioLoss()
        >>> # Model learns to maximize trading strategy Sharpe ratio

    Note:
        This is an advanced loss function. Use with caution:
        - Requires large batch sizes (for stable mean/std estimates)
        - May be unstable early in training
        - Consider using after pre-training with MSE

    Reference:
        Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement.
        *IEEE transactions on neural Networks*, 12(4), 875-889.
    """

    def __init__(self, risk_free_rate: float = 0.0):
        """Initialize Sharpe ratio loss."""
        super(SharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio loss.

        Args:
            predictions: Predicted returns (batch, num_horizons)
            targets: Actual returns (batch, num_horizons)

        Returns:
            Scalar loss value
        """
        # Trading strategy returns: sign of prediction * actual return
        strategy_returns = torch.sign(predictions) * targets

        # Compute Sharpe ratio for each horizon
        sharpe_ratios = []
        for h in range(strategy_returns.shape[1]):
            returns_h = strategy_returns[:, h]
            mean_return = returns_h.mean()
            std_return = returns_h.std() + 1e-8  # Add epsilon for stability

            sharpe_h = (mean_return - self.risk_free_rate) / std_return
            sharpe_ratios.append(sharpe_h)

        # Average Sharpe across horizons
        avg_sharpe = torch.stack(sharpe_ratios).mean()

        # Return negative (we minimize loss = maximize Sharpe)
        return -avg_sharpe


class QuantileLoss(nn.Module):
    """
    Quantile regression loss for probabilistic forecasting.

    Instead of predicting a single point estimate, quantile regression predicts
    multiple quantiles of the distribution (e.g., 10th, 50th, 90th percentiles).

    Args:
        quantiles (list): List of quantiles to predict. Default: [0.1, 0.5, 0.9]

    Mathematical Formulation:
        For quantile q:
        L_q(e) = q * e        if e >= 0 (under-prediction)
                 (q - 1) * e  if e < 0 (over-prediction)

        Where e = actual - predicted

    Example:
        >>> criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        >>> # Model outputs 3 values per horizon (10th, 50th, 90th percentile)
        >>> predictions = torch.randn(32, 5, 3)  # (batch, horizons, quantiles)
        >>> targets = torch.randn(32, 5)

    Use Case:
        Risk management - need confidence intervals, not just point estimates

    Reference:
        Koenker, R., & Bassett Jr, G. (1978). Regression quantiles.
        *Econometrica*, 46(1), 33-50.
    """

    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        """Initialize quantile loss."""
        super(QuantileLoss, self).__init__()
        self.quantiles = torch.tensor(quantiles)

        for q in quantiles:
            if not 0 < q < 1:
                raise ValueError(f"Quantiles must be in (0, 1), got {q}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute quantile loss.

        Args:
            predictions: Predicted quantiles of shape (batch, horizons, num_quantiles)
            targets: Actual values of shape (batch, horizons)

        Returns:
            Scalar loss value
        """
        # Expand targets to match predictions
        targets_expanded = targets.unsqueeze(-1).expand_as(predictions)

        # Compute errors
        errors = targets_expanded - predictions

        # Move quantiles to same device
        quantiles = self.quantiles.to(predictions.device)

        # Quantile loss
        loss = torch.max(quantiles * errors, (quantiles - 1) * errors)

        return loss.mean()


class HuberLoss(nn.Module):
    """
    Huber loss - robust to outliers.

    Combines the best of MSE (smooth gradients) and MAE (outlier robustness).
    Behaves like MSE for small errors, MAE for large errors.

    Args:
        delta (float): Threshold for switching from quadratic to linear.
            Default: 1.0

    Mathematical Formulation:
        L(e) = 0.5 * e^2                if |e| <= delta
               delta * (|e| - 0.5*delta) if |e| > delta

    Example:
        >>> criterion = HuberLoss(delta=1.0)
        >>> # Robust to flash crashes and extreme price movements

    Use Case:
        Financial data with occasional extreme outliers (flash crashes)
        that shouldn't dominate the loss.

    Reference:
        Huber, P. J. (1992). Robust estimation of a location parameter.
        In *Breakthroughs in statistics* (pp. 492-518). Springer.
    """

    def __init__(self, delta: float = 1.0):
        """Initialize Huber loss."""
        super(HuberLoss, self).__init__()
        self.delta = delta

        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Huber loss.

        Args:
            predictions: Predicted values
            targets: True values

        Returns:
            Scalar loss value
        """
        errors = predictions - targets
        abs_errors = torch.abs(errors)

        # Quadratic for small errors
        quadratic = 0.5 * errors**2

        # Linear for large errors
        linear = self.delta * (abs_errors - 0.5 * self.delta)

        # Use quadratic if |error| <= delta, else linear
        loss = torch.where(abs_errors <= self.delta, quadratic, linear)

        return loss.mean()


# =============================================================================
# Loss Function Factory
# =============================================================================


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.

    Args:
        loss_name (str): Name of loss function.
            Options: 'mse', 'mae', 'huber', 'directional', 'asymmetric',
                     'sharpe', 'quantile'
        **kwargs: Additional arguments for loss function.

    Returns:
        Loss function module.

    Example:
        >>> criterion = get_loss_function('directional', alpha=0.3)
        >>> criterion = get_loss_function('mse')
    """
    loss_functions = {
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": HuberLoss,
        "directional": DirectionalLoss,
        "asymmetric": AsymmetricLoss,
        "sharpe": SharpeRatioLoss,
        "quantile": QuantileLoss,
    }

    if loss_name.lower() not in loss_functions:
        raise ValueError(
            f"Unknown loss function: {loss_name}. " f"Available: {list(loss_functions.keys())}"
        )

    loss_class = loss_functions[loss_name.lower()]
    return loss_class(**kwargs)


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    """Test loss functions."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    logger.info("=== Testing Custom Loss Functions ===")

    # Create sample data
    batch_size = 32
    num_horizons = 5

    predictions = torch.randn(batch_size, num_horizons)
    targets = torch.randn(batch_size, num_horizons)

    # Test each loss function
    logger.info("\n1. Standard MSE:")
    mse_loss = nn.MSELoss()
    loss = mse_loss(predictions, targets)
    logger.info(f"   Loss: {loss.item():.4f}")

    logger.info("\n2. Directional Loss (alpha=0.5):")
    directional_loss = DirectionalLoss(alpha=0.5)
    loss = directional_loss(predictions, targets)
    logger.info(f"   Loss: {loss.item():.4f}")

    logger.info("\n3. Asymmetric Loss (beta=2.0):")
    asymmetric_loss = AsymmetricLoss(beta=2.0)
    loss = asymmetric_loss(predictions, targets)
    logger.info(f"   Loss: {loss.item():.4f}")

    logger.info("\n4. Sharpe Ratio Loss:")
    sharpe_loss = SharpeRatioLoss()
    loss = sharpe_loss(predictions, targets)
    logger.info(f"   Loss: {loss.item():.4f}")

    logger.info("\n5. Huber Loss (delta=1.0):")
    huber_loss = HuberLoss(delta=1.0)
    loss = huber_loss(predictions, targets)
    logger.info(f"   Loss: {loss.item():.4f}")

    logger.info("\n6. Quantile Loss:")
    quantile_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    # For quantile loss, predictions need extra dimension
    predictions_quantile = torch.randn(batch_size, num_horizons, 3)
    loss = quantile_loss(predictions_quantile, targets)
    logger.info(f"   Loss: {loss.item():.4f}")

    logger.info("\n7. Loss Function Factory:")
    criterion = get_loss_function("directional", alpha=0.3)
    loss = criterion(predictions, targets)
    logger.info(f"   Directional (alpha=0.3): {loss.item():.4f}")

    logger.info("\n✓ All loss functions tested successfully!")
