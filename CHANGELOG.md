# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffold with proper directory structure
- Extensible TimescaleDB schema for LOB data, predictions, and explainability
- Docker Compose setup for TimescaleDB and Redis
- Pydantic-based configuration management with validation
- Centralized logging configuration
- Binance WebSocket client for real-time LOB data streaming
- Support for future TST and FinBERT model extensions
- Comprehensive README with architecture and installation guide
- GitHub Actions CI/CD pipeline for automated testing, linting, and formatting
- Dependabot for automated dependency updates
- Pre-commit hooks for code quality enforcement
- Comprehensive test suite with pytest and coverage reporting
- Security scanning with Bandit and Safety
- QA documentation in README with all commands
- Convenience script `scripts/qa_check.sh` for running all QA checks
- Sample configuration tests

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- Critical: Swapped `absolute_error` and `squared_error` columns in evaluation service INSERT statement
- Live system now loads normalization parameters from model checkpoint instead of hardcoded zeros/ones
- Training pipeline now saves normalization mean/std in model checkpoints
- `load_data_from_db()` now queries and reconstructs all 43 features (was only 6)
- `flush_batch()` in database writer now performs actual transactional batch writes
- Clarified feature engineering normalization as intentional pass-through (handled by LOBDataset)
- Uncommented test dataset creation in training pipeline

### Security
- Environment variables properly handled via `.env` file
- Sensitive credentials redacted in config dumps

---

## [0.0.1] - 2024-11-08

### Added
- Project initialization
