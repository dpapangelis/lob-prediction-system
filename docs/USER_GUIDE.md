# LOB Prediction System: User Guide

## Overview

This guide provides step-by-step instructions for setting up, running, and using the LOB Prediction System. Whether you're collecting data, training models, or making predictions, this manual covers everything you need to know.

**System Requirements:**
- macOS 12+ or Linux (Ubuntu 20.04+)
- 16GB RAM (8GB minimum)
- 50GB free disk space
- Python 3.12+
- Docker Desktop

---

## Table of Contents

1. [Initial Setup](#1-initial-setup)
2. [Data Collection](#2-data-collection)
3. [Model Training](#3-model-training)
4. [Model Evaluation](#4-model-evaluation)
5. [Real-Time Prediction](#5-real-time-prediction)
6. [Troubleshooting](#6-troubleshooting)
7. [Maintenance & Updates](#7-maintenance--updates)

---

## 1. Initial Setup

### 1.1 Clone Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/lob-prediction-system.git
cd lob-prediction-system

# Verify you're on the main branch
git branch
```

### 1.2 Environment Setup

**Create Python Virtual Environment:**
```bash
# Using Python 3.12
python3.12 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

# Verify Python version
python --version  # Should show Python 3.12.x
```

**Install Dependencies:**
```bash
# Install all dependencies
pip install -e ".[dev]"

# This installs:
# - Core dependencies (PyTorch, asyncpg, etc.)
# - Development tools (pytest, black, etc.)
# - CI dependencies (flake8, mypy, etc.)

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import asyncpg; print('asyncpg OK')"
```

### 1.3 Docker Services Setup

**Install Docker Desktop:**

- macOS: Download from [docker.com](https://www.docker.com/products/docker-desktop)
- Linux: Follow [official guide](https://docs.docker.com/engine/install/)

**Start Services:**
```bash
# Start TimescaleDB and Redis
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# NAME                  STATUS    PORTS
# lob-timescaledb       Up        0.0.0.0:5432->5432/tcp
# lob-redis             Up        0.0.0.0:6379->6379/tcp
```

**Initialize Database:**
```bash
# Create database schema
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -f /docker-entrypoint-initdb.d/init.sql

# Verify tables exist
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "\dt"

# Expected output:
#              List of relations
#  Schema |    Name    | Type  |  Owner
# --------+------------+-------+----------
#  public | lob_data   | table | postgres
```

### 1.4 Configuration

**Environment Variables:**
```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env  # or use your preferred editor
```

**Key Settings:**
```bash
# .env file
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=lob_prediction
DB_USER=postgres
DB_PASSWORD=postgres

# Binance
BINANCE_API_KEY=your_key_here  # Optional, only for private endpoints
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=false  # Set to true for testing

# Trading
SYMBOLS=BTCUSDT,ETHUSDT  # Comma-separated
LOB_DEPTH_LEVELS=5

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

**Test Configuration:**
```bash
# Verify settings load correctly
python -c "from config.settings import settings; print(settings.db_host)"

# Should print: localhost
```

---

## 2. Data Collection

### 2.1 Quick Start (Test Collection)

**Collect 5 Minutes of Data:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure Docker services are running
docker-compose ps

# Start data collection
python -m src.data.live_pipeline

# You should see:
# 2024-11-09 10:00:00 | INFO | Starting live data pipeline for BTCUSDT
# 2024-11-09 10:00:01 | INFO | Connected to Binance
# 2024-11-09 10:00:02 | INFO | Pipeline stats: 1 processed, 1 written, 0 failed
# ...

# Let it run for 5 minutes, then press Ctrl+C

# Expected output:
# Pipeline Statistics:
#   Total snapshots processed: 300
#   Successfully written: 300
#   Failed: 0
#   Success rate: 100.00%
```

**Verify Data in Database:**
```bash
# Check data was written
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT COUNT(*),
         MIN(time) as first_snapshot,
         MAX(time) as last_snapshot
  FROM lob_data
  WHERE symbol='BTCUSDT';
"

# Expected output:
#  count |         first_snapshot         |         last_snapshot
# -------+--------------------------------+-------------------------------
#    300 | 2024-11-09 10:00:00.123+00     | 2024-11-09 10:05:00.456+00
```

### 2.2 Long-Term Data Collection

**Background Collection (Recommended):**
```bash
# Using tmux (recommended for long sessions)
tmux new -s data-collection

# Inside tmux session:
source .venv/bin/activate
python -m src.data.live_pipeline

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t data-collection

# Alternative: Using nohup
nohup python -m src.data.live_pipeline > collection.log 2>&1 &

# Check it's running
ps aux | grep live_pipeline

# View logs
tail -f collection.log
```

**Collection Targets:**

| Duration | Snapshots | Use Case | Recommended |
|----------|-----------|----------|-------------|
| 1 hour | ~3,600 | Testing pipeline | ✓ For development |
| 1 day | ~86,400 | Initial training | ✓ Minimum for training |
| 1 week | ~600,000 | Good baseline | ✓✓ Recommended |
| 1 month | ~2,500,000 | Production-ready | ✓✓✓ Best results |

**Monitor Collection:**
```bash
# Real-time monitoring query
watch -n 5 'docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT
    symbol,
    COUNT(*) as snapshots,
    MAX(time) as latest,
    NOW() - MAX(time) as lag
  FROM lob_data
  GROUP BY symbol;
"'

# Check for gaps in data
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT
    time,
    LAG(time) OVER (ORDER BY time) as prev_time,
    time - LAG(time) OVER (ORDER BY time) as gap
  FROM lob_data
  WHERE symbol='BTCUSDT'
  ORDER BY time DESC
  LIMIT 20;
"

# Gaps > 5 seconds indicate collection issues
```

**Stop Collection:**
```bash
# If using tmux:
tmux attach -t data-collection
# Press Ctrl+C
# Then: exit

# If using nohup:
ps aux | grep live_pipeline
kill <PID>

# Check final statistics in logs
tail -100 collection.log
```

### 2.3 Data Quality Checks

**After Collection, Verify Quality:**
```bash
# Run quality check script
python scripts/check_data_quality.py --symbol BTCUSDT --days 7

# Checks performed:
# ✓ No missing timestamps (gaps < 5s)
# ✓ No duplicate timestamps
# ✓ All features within valid ranges
# ✓ No NULL values in critical columns
# ✓ Spread values reasonable (< 100 bps)
```

**Manual Checks:**
```bash
# Check for NULL values
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT
    COUNT(*) as total,
    COUNT(mid_price) as has_mid_price,
    COUNT(spread) as has_spread,
    COUNT(imbalance) as has_imbalance
  FROM lob_data
  WHERE symbol='BTCUSDT';
"

# All counts should be equal (no NULLs)

# Check value ranges
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT
    MIN(spread) as min_spread,
    MAX(spread) as max_spread,
    AVG(spread) as avg_spread,
    MIN(imbalance) as min_imbalance,
    MAX(imbalance) as max_imbalance
  FROM lob_data
  WHERE symbol='BTCUSDT';
"

# Expected ranges:
# spread: 0.01 - 10 USDT (usually < 1)
# imbalance: -1.0 to 1.0
```

---

## 3. Model Training

### 3.1 Quick Training Test (Synthetic Data)

**Before training on real data, test the pipeline:**
```bash
# Test with synthetic data (fast, 5 epochs)
python -m src.models.train \
  --synthetic \
  --epochs 5 \
  --batch-size 32 \
  --log-level INFO

# Expected output:
# Epoch 1/5 | Train Loss: 0.9928 | Val Loss: 0.9989 | LR: 0.001000
# Epoch 2/5 | Train Loss: 0.9897 | Val Loss: 1.0030 | LR: 0.001000
# ...
# Training Complete!
# Best validation loss: 0.9989

# Checkpoints saved to: data/models/BTCUSDT/
```

### 3.2 Training on Real Data

**Prerequisites:**

- At least 1 day of data collected (86,400+ snapshots)
- Docker services running
- Sufficient disk space for checkpoints (~100MB per checkpoint)

**Start Training:**
```bash
# Full training run
python -m src.models.train \
  --symbol BTCUSDT \
  --days 7 \
  --epochs 100 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --patience 10 \
  --save-every 10 \
  --log-level INFO

# Training will run for several hours
# Monitor progress in real-time
```

**Training Progress:**
```
======================================================================
LOB Price Prediction - TCN Training
======================================================================
Using device: mps
Loading data from TimescaleDB...
Data loaded: 604,800 samples
Data split: train=423,360, val=90,720, test=90,720

======================================================================
Starting Training
======================================================================
Epochs: 100
Train batches: 6,615
Val batches: 1,418
Device: mps
======================================================================

Epoch 1: 100%|████████████| 6615/6615 [05:23<00:00, 20.43it/s, loss=0.0847]
Validating: 100%|██████████| 1418/1418 [00:32<00:00, 43.68it/s]
Epoch 1/100 | Train Loss: 0.0863 | Val Loss: 0.0921 | LR: 0.001000
✓ Saved best model to data/models/BTCUSDT/best_model.pth

Epoch 2: 100%|████████████| 6615/6615 [05:21<00:00, 20.58it/s, loss=0.0712]
Validating: 100%|██████████| 1418/1418 [00:31<00:00, 44.12it/s]
Epoch 2/100 | Train Loss: 0.0728 | Val Loss: 0.0834 | LR: 0.001000
✓ Saved best model to data/models/BTCUSDT/best_model.pth

...

Epoch 45: 100%|███████████| 6615/6615 [05:19<00:00, 20.71it/s, loss=0.0234]
Validating: 100%|██████████| 1418/1418 [00:30<00:00, 45.23it/s]
Epoch 45/100 | Train Loss: 0.0241 | Val Loss: 0.0389 | LR: 0.000125

Early stopping triggered after 45 epochs (no improvement for 10 epochs)

======================================================================
Training Complete!
Best validation loss: 0.0378
======================================================================
```

**Expected Training Times:**

| Data Size | Epochs | Batch Size | Device | Time |
|-----------|--------|------------|--------|------|
| 1 day (86K) | 50 | 64 | M2 MPS | ~2 hours |
| 1 week (600K) | 50 | 64 | M2 MPS | ~11 hours |
| 1 month (2.5M) | 50 | 64 | M2 MPS | ~40 hours |

**Training Outputs:**
```
data/models/BTCUSDT/
├── best_model.pth           # Best validation loss model
├── checkpoint_epoch10.pth   # Periodic checkpoints
├── checkpoint_epoch20.pth
├── ...
├── final_model.pth          # Final epoch model
└── training_curves.png      # Loss plots (if --plot used)
```

### 3.3 Monitoring Training

**Check Training Progress (Another Terminal):**
```bash
# View training logs
tail -f logs/training.log

# Check GPU/CPU usage (macOS)
sudo powermetrics --samplers gpu_power,cpu_power -i 1000

# Check memory usage
watch -n 5 'ps aux | grep train.py'
```

**TensorBoard (Optional, Future):**
```bash
# Install tensorboard
pip install tensorboard

# Start tensorboard
tensorboard --logdir data/models/BTCUSDT/runs

# Open browser: http://localhost:6006
```

### 3.4 Resume Training

**If Training Interrupted:**
```bash
# Find latest checkpoint
ls -lh data/models/BTCUSDT/

# Resume from checkpoint
python -m src.models.train \
  --symbol BTCUSDT \
  --days 7 \
  --epochs 100 \
  --batch-size 64 \
  --resume data/models/BTCUSDT/checkpoint_epoch30.pth

# Training will continue from epoch 31
```

### 3.5 Hyperparameter Tuning (Advanced)

**Grid Search:**
```bash
# Create tuning script
cat > scripts/tune_hyperparameters.sh << 'EOF'
#!/bin/bash

for lr in 0.0001 0.0005 0.001 0.005; do
  for batch_size in 32 64 128; do
    for dropout in 0.1 0.2 0.3; do
      echo "Training with LR=$lr, Batch=$batch_size, Dropout=$dropout"
      python -m src.models.train \
        --symbol BTCUSDT \
        --days 7 \
        --epochs 30 \
        --batch-size $batch_size \
        --learning-rate $lr \
        --dropout $dropout \
        --log-level WARNING \
        2>&1 | tee "logs/tune_${lr}_${batch_size}_${dropout}.log"
    done
  done
done
EOF

# Make executable
chmod +x scripts/tune_hyperparameters.sh

# Run tuning (will take many hours!)
./scripts/tune_hyperparameters.sh

# Analyze results
python scripts/analyze_tuning_results.py
```

---

## 4. Model Evaluation

### 4.1 Evaluate Trained Model

**Run Evaluation Script:**
```bash
# Evaluate best model on test set
python -m src.models.evaluate \
  --model data/models/BTCUSDT/best_model.pth \
  --symbol BTCUSDT \
  --days 7

# Output:
# ======================================================================
# Model Evaluation Results
# ======================================================================
# Model: data/models/BTCUSDT/best_model.pth
# Test Samples: 89,320
#
# Metrics by Horizon:
#
# Horizon: 1s
#   MSE: 0.0234
#   RMSE: 0.1530
#   MAE: 0.1124
#   R²: 0.4521
#   Directional Accuracy: 68.4%
#
# Horizon: 5s
#   MSE: 0.0312
#   RMSE: 0.1767
#   MAE: 0.1389
#   R²: 0.3812
#   Directional Accuracy: 64.2%
#
# Horizon: 10s
#   MSE: 0.0389
#   RMSE: 0.1972
#   MAE: 0.1567
#   R²: 0.3245
#   Directional Accuracy: 61.8%
#
# Horizon: 30s
#   MSE: 0.0512
#   RMSE: 0.2263
#   MAE: 0.1834
#   R²: 0.2456
#   Directional Accuracy: 58.3%
#
# Horizon: 60s
#   MSE: 0.0678
#   RMSE: 0.2604
#   MAE: 0.2123
#   R²: 0.1789
#   Directional Accuracy: 55.7%
#
# Overall Directional Accuracy: 61.7%
# ======================================================================
```

**Interpretation:**

- **MSE (Mean Squared Error):** Lower is better (measures prediction error magnitude)
- **RMSE (Root MSE):** Same units as target (percentage returns)
- **MAE (Mean Absolute Error):** Average absolute prediction error
- **R² (R-squared):** Proportion of variance explained (0-1, higher is better)
- **Directional Accuracy:** % of times predicted direction was correct

**Good Results:**
- Directional accuracy > 60% (significantly better than random 50%)
- R² > 0.3 for short horizons (1s, 5s)
- Accuracy decreases with horizon (expected)

### 4.2 Generate Evaluation Plots
```bash
# Create visualizations
python -m src.models.evaluate \
  --model data/models/BTCUSDT/best_model.pth \
  --symbol BTCUSDT \
  --days 7 \
  --plot

# Generates:
# - Prediction vs actual scatter plots
# - Error distribution histograms
# - Directional accuracy by horizon
# - Confusion matrices
#
# Saved to: data/models/BTCUSDT/evaluation/
```

### 4.3 SHAP Explainability (Advanced)
```bash
# Generate SHAP feature importance
python -m src.models.explain \
  --model data/models/BTCUSDT/best_model.pth \
  --symbol BTCUSDT \
  --samples 1000

# Output:
# Top 10 Most Important Features:
# 1. volume_imbalance_1: 0.234
# 2. spread_bps: 0.187
# 3. weighted_mid_price: 0.156
# 4. total_volume_imbalance: 0.143
# 5. depth_imbalance: 0.128
# ...
#
# SHAP plots saved to: data/models/BTCUSDT/shap/
```

---

## 5. Real-Time Prediction

### 5.1 Live Inference Server (Future)

**Start Prediction Server:**
```bash
# Start FastAPI server
python -m src.api.server \
  --model data/models/BTCUSDT/best_model.pth \
  --port 8000

# Server starts at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

**Make Predictions (API):**
```bash
# Get latest prediction
curl http://localhost:8000/predict/BTCUSDT

# Response:
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-11-09T10:30:45.123Z",
  "predictions": {
    "1s": 0.023,
    "5s": 0.045,
    "10s": 0.067,
    "30s": 0.089,
    "60s": 0.112
  },
  "confidence": {
    "1s": 0.68,
    "5s": 0.64,
    "10s": 0.62,
    "30s": 0.58,
    "60s": 0.56
  }
}
```

### 5.2 Batch Prediction

**Predict on Historical Data:**
```bash
# Predict for specific time range
python -m src.models.predict \
  --model data/models/BTCUSDT/best_model.pth \
  --symbol BTCUSDT \
  --start "2024-11-01 00:00:00" \
  --end "2024-11-01 23:59:59" \
  --output predictions.csv

# Output file columns:
# timestamp, actual_1s, pred_1s, actual_5s, pred_5s, ...
```

---

## 6. Troubleshooting

### 6.1 Common Issues

**Issue: Docker Services Won't Start**
```bash
# Check Docker is running
docker ps

# If error, restart Docker Desktop

# Check port conflicts
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# If ports in use, kill processes or change ports in docker-compose.yml
```

**Issue: Database Connection Failed**
```bash
# Test connection
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction

# If fails, check container logs
docker logs lob-timescaledb

# Restart database
docker-compose restart timescaledb
```

**Issue: WebSocket Connection Errors**
```bash
# Check internet connection
ping binance.com

# Check firewall isn't blocking WebSocket
# If using VPN, try disabling

# Test WebSocket manually
python -c "
import asyncio
import websockets

async def test():
    async with websockets.connect('wss://stream.binance.com/ws/btcusdt@depth5') as ws:
        msg = await ws.recv()
        print('Connection OK:', msg[:100])

asyncio.run(test())
"
```

**Issue: Training Very Slow**
```bash
# Check device being used
python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('CUDA available:', torch.cuda.is_available())
"

# If using CPU instead of GPU:
# - M2 Mac: Ensure PyTorch 2.0+ installed
# - Linux: Install CUDA-enabled PyTorch

# Reduce batch size if out of memory
python -m src.models.train --batch-size 32  # Instead of 64
```

**Issue: Out of Memory**
```bash
# Check memory usage
# macOS:
vm_stat

# Linux:
free -h

# Solutions:
# 1. Reduce batch size
--batch-size 16

# 2. Reduce sequence length
--sequence-length 50

# 3. Use smaller model
# Edit src/models/tcn.py:
# num_channels=[64, 64, 128, 128]  # Instead of [128, 128, 256, 256]
```

### 6.2 Data Issues

**Issue: Missing Data / Gaps**
```bash
# Check for gaps
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT
    time,
    time - LAG(time) OVER (ORDER BY time) as gap
  FROM lob_data
  WHERE symbol='BTCUSDT'
    AND time - LAG(time) OVER (ORDER BY time) > interval '10 seconds'
  ORDER BY time;
"

# If gaps found:
# - Check collection logs for errors
# - Restart collection to fill gaps
# - Gaps > 1 hour: Exclude that period from training
```

**Issue: Corrupt Data**
```bash
# Check for invalid values
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT * FROM lob_data
  WHERE spread < 0 OR spread > 1000
     OR imbalance < -1 OR imbalance > 1
     OR mid_price < 1000 OR mid_price > 1000000
  LIMIT 10;
"

# If found, delete corrupt rows
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  DELETE FROM lob_data
  WHERE spread < 0 OR spread > 1000
     OR imbalance < -1 OR imbalance > 1;
"
```

### 6.3 Getting Help

**Logs Locations:**
```
logs/
├── app.log              # Main application log
├── training.log         # Training logs
├── collection.log       # Data collection logs
└── errors.log           # Error-only log
```

**Enable Debug Logging:**
```bash
# Temporary (command line)
python -m src.data.live_pipeline --log-level DEBUG

# Permanent (edit .env)
LOG_LEVEL=DEBUG
```

**Report Issues:**
```bash
# Collect diagnostic info
python scripts/diagnostic_report.py > diagnostic.txt

# Include in bug report:
# - diagnostic.txt
# - Relevant logs from logs/
# - Steps to reproduce
```

---

## 7. Maintenance & Updates

### 7.1 Update Dependencies
```bash
# Activate virtual environment
source .venv/bin/activate

# Update all packages
pip install --upgrade pip
pip install --upgrade -e ".[dev]"

# Verify updates didn't break anything
pytest tests/
```

### 7.2 Database Maintenance

**Cleanup Old Data:**
```bash
# Delete data older than 30 days
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  DELETE FROM lob_data
  WHERE time < NOW() - INTERVAL '30 days';
"

# Vacuum database
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  VACUUM ANALYZE lob_data;
"
```

**Backup Database:**
```bash
# Backup to file
docker exec -it lob-timescaledb pg_dump -U postgres lob_prediction > backup_$(date +%Y%m%d).sql

# Restore from backup
docker exec -i lob-timescaledb psql -U postgres lob_prediction < backup_20241109.sql
```

**Database Statistics:**
```bash
# Check table size
docker exec -it lob-timescaledb psql -U postgres -d lob_prediction -c "
  SELECT
    pg_size_pretty(pg_total_relation_size('lob_data')) as total_size,
    pg_size_pretty(pg_relation_size('lob_data')) as table_size,
    pg_size_pretty(pg_indexes_size('lob_data')) as indexes_size;
"

# Example output:
#  total_size | table_size | indexes_size
# ------------+------------+--------------
#  2048 MB    | 1536 MB    | 512 MB
```

### 7.3 Model Retraining

**When to Retrain:**

- After collecting 2-4 weeks of new data
- If prediction accuracy degrades (market regime change)
- After significant market events (e.g., Bitcoin halving)

**Incremental Retraining:**
```bash
# Train on most recent data only
python -m src.models.train \
  --symbol BTCUSDT \
  --days 30 \
  --epochs 50 \
  --resume data/models/BTCUSDT/best_model.pth  # Start from existing model

# This performs "fine-tuning" on new data
```

### 7.4 Git Workflow

**Before Making Changes:**
```bash
# Pull latest changes
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name
```

**After Making Changes:**
```bash
# Run tests
pytest tests/

# Format code
black src/ tests/
isort src/ tests/

# Commit
git add .
git commit -m "feat: your feature description"

# Push
git push origin feature/your-feature-name
```

**Create Pull Request:**

- Go to GitHub
- Click "New Pull Request"
- Select `develop` as base branch
- Fill in description
- Wait for CI to pass
- Merge when ready

---

## 8. Quick Reference

### 8.1 Common Commands
```bash
# Start system
docker-compose up -d
source .venv/bin/activate

# Collect data
python -m src.data.live_pipeline

# Train model
python -m src.models.train --symbol BTCUSDT --days 7 --epochs 100

# Evaluate model
python -m src.models.evaluate --model data/models/BTCUSDT/best_model.pth

# Stop system
docker-compose down
deactivate
```

### 8.2 File Locations
```
lob-prediction-system/
├── config/              # Configuration files
├── data/
│   ├── models/          # Trained model checkpoints
│   ├── raw/             # Raw collected data (CSV)
│   └── processed/       # Processed datasets
├── docs/                # Documentation
├── logs/                # Log files
├── src/
│   ├── data/            # Data collection & processing
│   ├── models/          # Model definitions & training
│   └── api/             # API server (future)
├── tests/               # Unit tests
├── .env                 # Environment variables
└── docker-compose.yml   # Docker services config
```

### 8.3 Important URLs

- **GitHub Repository:** https://github.com/yourusername/lob-prediction-system
- **Binance API Docs:** https://binance-docs.github.io/apidocs/spot/en/
- **TimescaleDB Docs:** https://docs.timescale.com/
- **PyTorch Docs:** https://pytorch.org/docs/

---

## 9. FAQ

**Q: How much data do I need to train a good model?**

A: Minimum 1 week (600K snapshots). Optimal: 1 month (2.5M snapshots).

**Q: Can I train on multiple symbols simultaneously?**

A: Not yet. Currently one symbol per training run. Future: Multi-symbol support.

**Q: What if Binance API changes?**

A: The WebSocket API is stable. If changes occur, update `src/data/binance_stream.py`.

**Q: Can I use this for live trading?**

A: **No.** This is a research system for educational purposes. Not production-ready for real money trading.

**Q: How often should I retrain?**

A: Every 2-4 weeks, or when accuracy degrades by >5%.

**Q: What's the expected prediction accuracy?**

A: 60-70% directional accuracy for 1-10s horizons. 55-60% for longer horizons.

**Q: Can I run this on a cloud server?**

A: Yes. Use a GPU instance (AWS p3, GCP n1-highmem) for faster training.

**Q: How do I cite this in my dissertation?**

A: See `CITATION.md` for proper citation format.

---

## Appendix: System Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  Command Line Tools | API Server (Future) | Dashboard (Future)  │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                      Application Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Data         │  │ Model        │  │ Prediction   │         │
│  │ Collection   │  │ Training     │  │ Engine       │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                      Infrastructure                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ TimescaleDB  │  │ Redis        │  │ PyTorch      │         │
│  │ (PostgreSQL) │  │ (Cache)      │  │ (ML Engine)  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                      External Services                           │
│                   Binance WebSocket API                          │
└──────────────────────────────────────────────────────────────────┘
```

---

**End of User Guide**
