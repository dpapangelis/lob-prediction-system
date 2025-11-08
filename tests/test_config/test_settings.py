"""Tests for configuration management."""

from config.settings import Settings, get_settings


def test_settings_initialization():
    """Test that settings initialize with defaults."""
    settings = Settings()

    assert settings.device in ["cpu", "mps", "cuda"]
    assert settings.db_port == 5432
    assert settings.redis_port == 6379
    assert len(settings.binance_symbols) > 0


def test_database_url_construction():
    """Test database URL is constructed correctly."""
    settings = Settings()

    url = settings.database_url
    assert url.startswith("postgresql://")
    assert "localhost" in url or settings.db_host in url
    assert str(settings.db_port) in url


def test_async_database_url():
    """Test async database URL construction."""
    settings = Settings()

    url = settings.async_database_url
    assert url.startswith("postgresql+asyncpg://")


def test_redis_url_construction():
    """Test Redis URL construction."""
    settings = Settings()

    url = settings.redis_url
    assert url.startswith("redis://")
    assert str(settings.redis_port) in url


def test_settings_safe_dump():
    """Test that sensitive fields are redacted in safe dump."""
    settings = Settings(
        binance_api_key="secret_key", binance_api_secret="secret_secret", db_password="db_pass"
    )

    safe_dump = settings.model_dump_safe()

    assert safe_dump["binance_api_key"] == "***REDACTED***"
    assert safe_dump["binance_api_secret"] == "***REDACTED***"
    assert safe_dump["db_password"] == "***REDACTED***"


def test_get_settings_dependency():
    """Test FastAPI dependency injection helper."""
    settings = get_settings()

    assert isinstance(settings, Settings)
