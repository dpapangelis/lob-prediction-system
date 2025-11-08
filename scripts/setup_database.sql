-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- LOB RAW DATA (Level 1 - Current MVP)
-- =============================================================================
CREATE TABLE IF NOT EXISTS lob_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- Order book levels (extensible to 5-10 levels)
    bid_price_1 DECIMAL(20, 8),
    bid_volume_1 DECIMAL(20, 8),
    ask_price_1 DECIMAL(20, 8),
    ask_volume_1 DECIMAL(20, 8),
    bid_price_2 DECIMAL(20, 8),
    bid_volume_2 DECIMAL(20, 8),
    ask_price_2 DECIMAL(20, 8),
    ask_volume_2 DECIMAL(20, 8),
    bid_price_3 DECIMAL(20, 8),
    bid_volume_3 DECIMAL(20, 8),
    ask_price_3 DECIMAL(20, 8),
    ask_volume_3 DECIMAL(20, 8),
    
    -- Derived features
    mid_price DECIMAL(20, 8),
    spread DECIMAL(20, 8),
    imbalance DECIMAL(10, 4),
    
    -- Metadata
    raw_data JSONB  -- Store full raw message for debugging/future features
);

SELECT create_hypertable('lob_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_lob_symbol_time ON lob_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_lob_raw_data ON lob_data USING GIN (raw_data);

-- =============================================================================
-- TCN PREDICTIONS (Short-term - MVP)
-- =============================================================================
CREATE TABLE IF NOT EXISTS tcn_predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_version VARCHAR(50),
    
    -- Predictions at different horizons
    predicted_price_1s DECIMAL(20, 8),
    predicted_price_5s DECIMAL(20, 8),
    predicted_price_10s DECIMAL(20, 8),
    predicted_price_30s DECIMAL(20, 8),
    predicted_price_60s DECIMAL(20, 8),
    
    -- Actual values (filled in later for evaluation)
    actual_price_1s DECIMAL(20, 8),
    actual_price_5s DECIMAL(20, 8),
    actual_price_10s DECIMAL(20, 8),
    actual_price_30s DECIMAL(20, 8),
    actual_price_60s DECIMAL(20, 8),
    
    -- Confidence scores
    confidence DECIMAL(5, 4),
    
    -- Inference metadata
    inference_time_ms DECIMAL(10, 2),
    input_features JSONB  -- Store features used for this prediction
);

SELECT create_hypertable('tcn_predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_tcn_symbol_time ON tcn_predictions (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_tcn_model_version ON tcn_predictions (model_version, time DESC);

-- =============================================================================
-- TST PREDICTIONS (Long-term - Future Extension)
-- =============================================================================
CREATE TABLE IF NOT EXISTS tst_predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_version VARCHAR(50),
    
    -- Long-term predictions (5-60 minutes)
    predicted_price_5min DECIMAL(20, 8),
    predicted_price_15min DECIMAL(20, 8),
    predicted_price_30min DECIMAL(20, 8),
    predicted_price_60min DECIMAL(20, 8),
    
    -- Actual values
    actual_price_5min DECIMAL(20, 8),
    actual_price_15min DECIMAL(20, 8),
    actual_price_30min DECIMAL(20, 8),
    actual_price_60min DECIMAL(20, 8),
    
    -- Inputs used (for analysis)
    tcn_predictions_used JSONB,  -- Array of TCN predictions fed to TST
    sentiment_score DECIMAL(5, 4),  -- Future: from FinBERT
    
    -- Metadata
    confidence DECIMAL(5, 4),
    inference_time_ms DECIMAL(10, 2)
);

SELECT create_hypertable('tst_predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_tst_symbol_time ON tst_predictions (symbol, time DESC);

-- =============================================================================
-- SHAP EXPLANATIONS (TCN & TST)
-- =============================================================================
CREATE TABLE IF NOT EXISTS shap_values (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    model_type VARCHAR(20),  -- 'tcn' or 'tst'
    model_version VARCHAR(50),
    prediction_horizon VARCHAR(20),  -- e.g., '1s', '5min'
    
    -- Feature importance
    feature_name VARCHAR(100),
    shap_value DECIMAL(20, 8),
    feature_value DECIMAL(20, 8),
    
    -- Grouping for aggregations
    batch_id UUID  -- Group all features from one prediction
);

SELECT create_hypertable('shap_values', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_shap_model_time ON shap_values (model_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_shap_batch ON shap_values (batch_id);

-- =============================================================================
-- SENTIMENT DATA (Future: FinBERT)
-- =============================================================================
CREATE TABLE IF NOT EXISTS sentiment_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- News/Tweet source
    source VARCHAR(50),  -- 'twitter', 'news', 'reddit'
    source_url TEXT,
    text_snippet TEXT,
    
    -- FinBERT outputs
    sentiment_positive DECIMAL(5, 4),
    sentiment_negative DECIMAL(5, 4),
    sentiment_neutral DECIMAL(5, 4),
    sentiment_score DECIMAL(5, 4),  -- Aggregated score
    
    -- Metadata
    raw_data JSONB
);

SELECT create_hypertable('sentiment_data', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time ON sentiment_data (symbol, time DESC);

-- =============================================================================
-- MODEL METADATA & VERSIONING
-- =============================================================================
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(20),  -- 'tcn', 'tst'
    version VARCHAR(50) UNIQUE NOT NULL,
    architecture JSONB,  -- Store hyperparameters
    training_dataset_period TSTZRANGE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    metrics JSONB,  -- Store training metrics (loss, accuracy, etc.)
    file_path TEXT,
    is_active BOOLEAN DEFAULT FALSE
);

-- =============================================================================
-- CONTINUOUS AGGREGATES (Performance optimization)
-- =============================================================================

-- 1-minute OHLCV aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS lob_1min_agg
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    FIRST(mid_price, time) AS open,
    MAX(mid_price) AS high,
    MIN(mid_price) AS low,
    LAST(mid_price, time) AS close,
    AVG(bid_volume_1 + ask_volume_1) AS avg_volume,
    AVG(spread) AS avg_spread,
    AVG(imbalance) AS avg_imbalance
FROM lob_data
GROUP BY bucket, symbol
WITH NO DATA;

SELECT add_continuous_aggregate_policy('lob_1min_agg',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- TCN prediction accuracy aggregates (for dashboard)
CREATE MATERIALIZED VIEW IF NOT EXISTS tcn_accuracy_1min
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    symbol,
    model_version,
    COUNT(*) AS prediction_count,
    AVG(ABS(predicted_price_1s - actual_price_1s)) AS mae_1s,
    AVG(ABS(predicted_price_5s - actual_price_5s)) AS mae_5s,
    AVG(ABS(predicted_price_60s - actual_price_60s)) AS mae_60s,
    AVG(confidence) AS avg_confidence
FROM tcn_predictions
WHERE actual_price_1s IS NOT NULL
GROUP BY bucket, symbol, model_version
WITH NO DATA;

SELECT add_continuous_aggregate_policy('tcn_accuracy_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- =============================================================================
-- RETENTION POLICIES (Manage data growth)
-- =============================================================================

-- Keep raw LOB data for 7 days (high frequency)
SELECT add_retention_policy('lob_data', INTERVAL '7 days', if_not_exists => TRUE);

-- Keep predictions for 30 days
SELECT add_retention_policy('tcn_predictions', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('tst_predictions', INTERVAL '30 days', if_not_exists => TRUE);

-- Keep aggregates for 90 days
SELECT add_retention_policy('lob_1min_agg', INTERVAL '90 days', if_not_exists => TRUE);

-- Keep SHAP values for 14 days (can be expensive to store)
SELECT add_retention_policy('shap_values', INTERVAL '14 days', if_not_exists => TRUE);

-- Keep sentiment data for 30 days
SELECT add_retention_policy('sentiment_data', INTERVAL '30 days', if_not_exists => TRUE);