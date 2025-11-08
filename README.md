# Lobium: A Real-Time Predictive System for Market Microstructure Forecasting

Real-time cryptocurrency price prediction using Temporal Convolutional Networks (TCN) with SHAP explainability on Limit Order Book (LOB) data.

## 🎯 Project Overview

This system implements a hierarchical machine learning architecture for short-term cryptocurrency price prediction:

- **MVP**: TCN model for short-term predictions (1-60 seconds) with SHAP explainability
- **Future Extension**: Time Series Transformer (TST) for longer-term predictions (5-60 minutes)
- **Stretch Goal**: FinBERT sentiment analysis integration for market news

### Key Features

- ✅ Real-time LOB data ingestion from Binance WebSocket API
- ✅ TimescaleDB for efficient time-series storage
- ✅ PyTorch TCN model optimized for Apple Silicon (MPS)
- ✅ SHAP-based model explainability
- ✅ Live dashboard for monitoring predictions
- ✅ Extensible schema for future model layers

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│                 Real-time Data Pipeline                 │
│                                                          │
│  Binance WebSocket → Feature Engineering → TCN Model   │
│         ↓                    ↓                  ↓        │
│    TimescaleDB          Redis Cache        Predictions  │
└─────────────────────────────────────────────────────────┘
                              ↓
                    ┌──────────────────┐
                    │  Dashboard (UI)  │
                    │  - Live prices   │
                    │  - Predictions   │
                    │  - SHAP values   │
                    └──────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Data Ingestion** | WebSocket (asyncio) | Stream LOB data from Binance |
| **Feature Engineering** | Pandas, NumPy | Compute LOB features (spread, imbalance, etc.) |
| **Model** | PyTorch TCN | Short-term price prediction |
| **Explainability** | SHAP | Feature importance analysis |
| **Database** | TimescaleDB (PostgreSQL) | Time-series data storage |
| **Cache** | Redis | Message queue & intermediate storage |
| **API** | FastAPI | REST endpoints for predictions |
| **Dashboard** | Streamlit | Real-time visualization |
| **Experiment Tracking** | Weights & Biases | ML experiment management |

---

## 📊 Database Schema

### Core Tables

- **`lob_data`**: Raw limit order book snapshots (bid/ask prices & volumes)
- **`tcn_predictions`**: TCN model predictions at multiple horizons (1s, 5s, 10s, 30s, 60s)
- **`tst_predictions`**: *(Future)* Transformer predictions for longer horizons
- **`shap_values`**: Feature importance explanations
- **`sentiment_data`**: *(Future)* FinBERT sentiment scores
- **`model_versions`**: Model metadata and versioning

### Continuous Aggregates

- **`lob_1min_agg`**: 1-minute OHLCV aggregates for dashboard
- **`tcn_accuracy_1min`**: Model performance metrics

See [`scripts/setup_database.sql`](scripts/setup_database.sql) for full schema.

---

## 🚀 Installation

### Prerequisites

- **Python 3.12+** (3.14 not yet supported by all dependencies)
- **Docker Desktop** (for TimescaleDB & Redis)
- **macOS** (tested on M2), Linux, or Windows with WSL2

### 1. Clone Repository
```bash
git clone https://github.com/dpapangelis/lob-prediction-system.git
cd lob-prediction-system
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### 3. Configure Environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings (optional for basic testing)
# Binance API keys only needed for authenticated endpoints
nano .env
```

### 4. Start Database Services
```bash
# Start TimescaleDB and Redis
docker-compose up -d

# Verify services are running
docker-compose ps

# Check database schema was created
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "\dt"
```

---

## 🧪 Development & Quality Assurance

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_data/test_binance_stream.py -v

# Run tests matching pattern
pytest tests/ -k "test_websocket" -v
```

Coverage report will be generated in `htmlcov/index.html` - open it in your browser to see detailed coverage.

### Code Formatting
```bash
# Check if code needs formatting (dry run)
black --check src/ tests/ config/

# Auto-format all code
black src/ tests/ config/

# Check import sorting
isort --check-only src/ tests/ config/

# Fix import sorting
isort src/ tests/ config/
```

### Linting
```bash
# Lint with flake8 (style violations)
flake8 src/ tests/ config/ --max-line-length=100 --extend-ignore=E203,W503

# Type checking with mypy
mypy src/ --ignore-missing-imports
```

### Security Scanning
```bash
# Check for known vulnerabilities in dependencies
safety check

# Static security analysis
bandit -r src/ -f screen
```

### All-in-One QA Check

Run everything before pushing:
```bash
# Format code
black src/ tests/ config/
isort src/ tests/ config/

# Run tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Lint
flake8 src/ tests/ config/ --max-line-length=100 --extend-ignore=E203,W503

# Type check
mypy src/ --ignore-missing-imports
```

### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically check code before every commit:
```bash
# Install pre-commit
pre-commit install

# Manually run on all files
pre-commit run --all-files
```

This will automatically run Black, isort, and other checks when you `git commit`.

---

## 🎮 Usage

### Test WebSocket Connection
```bash
# Stream real-time LOB data (runs for 10 seconds)
python -m src.data.binance_stream
```

### Train TCN Model
```bash
# Coming soon
python -m src.models.train --config config/tcn_config.yaml
```

### Run Inference
```bash
# Coming soon
python -m src.models.inference --symbol BTCUSDT
```

### Start Dashboard
```bash
# Coming soon
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure
```
lob-prediction-system/
├── config/                  # Configuration management
│   ├── settings.py         # Pydantic settings
│   └── logging_config.py   # Logging setup
│
├── data/                    # Data storage (gitignored)
│   ├── raw/                # Raw LOB data cache
│   ├── processed/          # Preprocessed features
│   └── models/             # Saved model checkpoints
│
├── src/
│   ├── data/               # Data ingestion & processing
│   │   ├── binance_stream.py    # WebSocket client
│   │   ├── feature_engineering.py
│   │   └── db_writer.py         # TimescaleDB writer
│   │
│   ├── models/             # ML models
│   │   ├── tcn.py          # TCN architecture
│   │   ├── train.py        # Training script
│   │   └── inference.py    # Real-time inference
│   │
│   ├── explainability/     # Model interpretation
│   │   └── shap_analysis.py
│   │
│   ├── api/                # REST API
│   │   └── main.py         # FastAPI app
│   │
│   └── dashboard/          # Visualization
│       └── app.py          # Streamlit dashboard
│
├── tests/                  # Unit & integration tests
├── notebooks/              # Jupyter notebooks for EDA
├── scripts/                # Utility scripts
│   └── setup_database.sql  # DB initialization
│
├── docs/                   # Additional documentation
├── docker-compose.yml      # Container orchestration
├── pyproject.toml          # Dependencies & config
└── README.md               # This file
```

---

## 🧪 Development

### Run Tests
```bash
pytest tests/ -v --cov=src
```

### Code Formatting
```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
flake8 src/ tests/
mypy src/
```

### Git Workflow
```bash
# Create feature branch
git checkout develop
git checkout -b feature/your-feature

# Make changes, then commit
git add .
git commit -m "feat: your feature description"

# Push and create PR
git push -u origin feature/your-feature
```

---

## 📈 Data Flow

### 1. Data Ingestion
```
Binance API → WebSocket → JSON Parser → Feature Engineering
```

### 2. Feature Engineering
```
Raw LOB → Spread, Imbalance, Volume → Normalization → Feature Vector
```

### 3. Model Inference
```
Feature Vector → TCN Model → Price Predictions (1s, 5s, 10s, 30s, 60s)
```

### 4. Explainability
```
Feature Vector + Predictions → SHAP → Feature Importance
```

### 5. Storage
```
All Data → TimescaleDB → Continuous Aggregates → Dashboard
```

---

## 🔬 Research Context

This project is part of a dissertation at **CITY College, University of York**, investigating:

1. **Effectiveness of TCN models** for short-term LOB-based price prediction
2. **Feature importance** in financial time-series via SHAP analysis
3. **Hierarchical prediction architectures** (TCN → TST) for multi-horizon forecasting
4. **Impact of sentiment analysis** (FinBERT) on prediction accuracy

---

## 📝 TODO / Roadmap

### Phase 1: MVP (Current)
- [x] Project scaffold & configuration
- [x] Database schema design
- [x] WebSocket data ingestion
- [ ] Feature engineering module
- [ ] TCN model implementation
- [ ] Training pipeline
- [ ] SHAP integration
- [ ] Basic dashboard

### Phase 2: Extensions
- [ ] TST model for long-term predictions
- [ ] Model ensemble (TCN + TST)
- [ ] Advanced evaluation metrics

### Phase 3: Stretch Goals
- [ ] FinBERT sentiment integration
- [ ] Multi-exchange support
- [ ] Reinforcement learning for trading strategy

---

## 📚 References

### Documentation
- [Binance WebSocket API](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [SHAP Documentation](https://shap.readthedocs.io/)

---

## 📄 License

All Rights Reserved © 2025 D. Papangelis

This code and all associated files are part of the dissertation project
“Lobium: A Real-Time Predictive System for Market Microstructure Forecasting”
submitted to CITY College, The University of York Europe Campus.

No permission is granted to copy, modify, distribute, or use this software
except for academic examination or evaluation related to the dissertation.

For all other uses, written permission from the author is required.

---

## 👤 Author

**Dimitris Papangelis**
Computer Science Student, CITY College, University of York
Dissertation Project 2024-2025

---

## 🙏 Acknowledgments

- Binance for free public market data API
- TimescaleDB team for excellent time-series database
- SHAP developers for explainability tools
