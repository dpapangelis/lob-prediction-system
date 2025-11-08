"""
Application configuration management using Pydantic settings.
Loads from environment variables with validation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Project Paths
    # ==========================================================================
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    model_checkpoint_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent / "data" / "models"
    )

    # ==========================================================================
    # Binance API Configuration
    # ==========================================================================
    binance_api_key: str = Field(default="", description="Binance API key")
    binance_api_secret: str = Field(default="", description="Binance API secret")
    binance_testnet: bool = Field(default=False, description="Use Binance testnet")
    
    # Trading pairs to stream
    binance_symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT"],
        description="Symbols to stream LOB data for"
    )
    
    # Limit order book depth
    lob_depth_levels: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of LOB levels to fetch"
    )

    # ==========================================================================
    # Database Configuration (TimescaleDB)
    # ==========================================================================
    db_host: str = Field(default="localhost", description="Database host")
    db_port: int = Field(default=5432, description="Database port")
    db_name: str = Field(default="lob_prediction", description="Database name")
    db_user: str = Field(default="postgres", description="Database user")
    db_password: str = Field(default="postgres", description="Database password")
    
    @property
    def database_url(self) -> str:
        """Construct database URL for SQLAlchemy."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def async_database_url(self) -> str:
        """Construct async database URL for asyncpg."""
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    # ==========================================================================
    # Redis Configuration
    # ==========================================================================
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    device: Literal["cpu", "mps", "cuda"] = Field(
        default="mps",
        description="Device for PyTorch (mps for M1/M2, cuda for NVIDIA)"
    )
    
    # TCN Hyperparameters (can be overridden)
    tcn_num_channels: list[int] = Field(
        default=[64, 64, 128, 128],
        description="Channel sizes for TCN layers"
    )
    tcn_kernel_size: int = Field(default=3, ge=2, description="TCN kernel size")
    tcn_dropout: float = Field(default=0.2, ge=0.0, le=0.5, description="TCN dropout rate")
    
    # Training
    batch_size: int = Field(default=64, ge=1, description="Training batch size")
    learning_rate: float = Field(default=1e-3, gt=0, description="Learning rate")
    max_epochs: int = Field(default=100, ge=1, description="Maximum training epochs")
    
    # Inference
    prediction_horizons: list[int] = Field(
        default=[1, 5, 10, 30, 60],
        description="Prediction horizons in seconds"
    )

    # ==========================================================================
    # Experiment Tracking (Weights & Biases)
    # ==========================================================================
    wandb_api_key: str = Field(default="", description="W&B API key")
    wandb_project: str = Field(
        default="lob-prediction-system",
        description="W&B project name"
    )
    wandb_entity: str = Field(default="", description="W&B entity (username/team)")
    use_wandb: bool = Field(default=False, description="Enable W&B logging")

    # ==========================================================================
    # Logging Configuration
    # ==========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Path | None = Field(
        default=None,
        description="Log file path (None for stdout only)"
    )

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, ge=1024, le=65535, description="FastAPI port")
    
    # Dashboard
    dashboard_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Streamlit dashboard port"
    )

    # ==========================================================================
    # Data Processing Configuration
    # ==========================================================================
    feature_lookback_window: int = Field(
        default=100,
        ge=10,
        description="Number of timesteps to look back for features"
    )
    
    normalization_method: Literal["zscore", "minmax", "robust"] = Field(
        default="zscore",
        description="Feature normalization method"
    )

    # ==========================================================================
    # System Configuration
    # ==========================================================================
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Number of worker processes/threads"
    )
    
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("model_checkpoint_dir", "data_dir")
    @classmethod
    def ensure_directory_exists(cls, v: Path) -> Path:
        """Create directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_dump_safe(self) -> dict:
        """Dump settings without sensitive information."""
        dump = self.model_dump()
        # Redact sensitive fields
        sensitive_fields = ["binance_api_key", "binance_api_secret", "wandb_api_key", "db_password"]
        for field in sensitive_fields:
            if field in dump:
                dump[field] = "***REDACTED***"
        return dump


# =============================================================================
# Global Settings Instance
# =============================================================================
settings = Settings()


# =============================================================================
# Helper Functions
# =============================================================================
def get_settings() -> Settings:
    """
    Dependency injection helper for FastAPI.
    
    Usage:
        @app.get("/config")
        def get_config(settings: Settings = Depends(get_settings)):
            return settings.model_dump_safe()
    """
    return settings


if __name__ == "__main__":
    # Test configuration
    print("=== Application Settings ===")
    print(f"Project Root: {settings.project_root}")
    print(f"Database URL: {settings.database_url}")
    print(f"Redis URL: {settings.redis_url}")
    print(f"Device: {settings.device}")
    print(f"Symbols: {settings.binance_symbols}")
    print(f"\n=== Safe Dump (Sensitive Fields Redacted) ===")
    import json
    print(json.dumps(settings.model_dump_safe(), indent=2, default=str))