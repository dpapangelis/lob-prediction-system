-- LOB Prediction System - Database Schema
-- Initialize TimescaleDB with all required tables

\c lob_prediction;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =============================================================================
-- LOB Data Table (Historical limit order book snapshots)
-- =============================================================================

CREATE TABLE IF NOT EXISTS lob_data (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,

    -- Price data
    mid_price DOUBLE PRECISION NOT NULL,
    spread DOUBLE PRECISION NOT NULL,
    spread_bps DOUBLE PRECISION NOT NULL,

    -- Level 1 (best bid/ask)
    bid_price_1 DOUBLE PRECISION NOT NULL,
    bid_volume_1 DOUBLE PRECISION NOT NULL,
    ask_price_1 DOUBLE PRECISION NOT NULL,
    ask_volume_1 DOUBLE PRECISION NOT NULL,

    -- Level 2
    bid_price_2 DOUBLE PRECISION NOT NULL,
    bid_volume_2 DOUBLE PRECISION NOT NULL,
    ask_price_2 DOUBLE PRECISION NOT NULL,
    ask_volume_2 DOUBLE PRECISION NOT NULL,

    -- Level 3
    bid_price_3 DOUBLE PRECISION NOT NULL,
    bid_volume_3 DOUBLE PRECISION NOT NULL,
    ask_price_3 DOUBLE PRECISION NOT NULL,
    ask_volume_3 DOUBLE PRECISION NOT NULL,

    -- Level 4
    bid_price_4 DOUBLE PRECISION NOT NULL,
    bid_volume_4 DOUBLE PRECISION NOT NULL,
    ask_price_4 DOUBLE PRECISION NOT NULL,
    ask_volume_4 DOUBLE PRECISION NOT NULL,

    -- Level 5
    bid_price_5 DOUBLE PRECISION NOT NULL,
    bid_volume_5 DOUBLE PRECISION NOT NULL,
    ask_price_5 DOUBLE PRECISION NOT NULL,
    ask_volume_5 DOUBLE PRECISION NOT NULL,

    -- Aggregated features
    total_bid_volume DOUBLE PRECISION NOT NULL,
    total_ask_volume DOUBLE PRECISION NOT NULL,
    volume_imbalance DOUBLE PRECISION NOT NULL,
    weighted_mid_price DOUBLE PRECISION NOT NULL,
    price_range DOUBLE PRECISION NOT NULL,
    depth_imbalance DOUBLE PRECISION NOT NULL,

    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable
SELECT create_hypertable('lob_data', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_lob_data_symbol_time ON lob_data (symbol, time DESC);

-- Retention policy (keep 90 days)
SELECT add_retention_policy('lob_data', INTERVAL '90 days', if_not_exists => TRUE);

COMMENT ON TABLE lob_data IS 'Historical limit order book snapshots with computed features';

-- =============================================================================
-- Predictions Table (Model predictions)
-- =============================================================================

CREATE TABLE IF NOT EXISTS predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,

    -- Predictions for each horizon
    pred_1s DOUBLE PRECISION NOT NULL,
    pred_5s DOUBLE PRECISION NOT NULL,
    pred_10s DOUBLE PRECISION NOT NULL,
    pred_30s DOUBLE PRECISION NOT NULL,
    pred_60s DOUBLE PRECISION NOT NULL,

    -- Context at prediction time
    mid_price DOUBLE PRECISION NOT NULL,
    spread_bps DOUBLE PRECISION,
    volume_imbalance DOUBLE PRECISION,

    -- Metadata
    model_confidence DOUBLE PRECISION,
    inference_time_ms DOUBLE PRECISION,

    PRIMARY KEY (time, symbol, model_version)
);

-- Convert to hypertable
SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time
ON predictions (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_model_version
ON predictions (model_version, time DESC);

-- Retention policy
SELECT add_retention_policy('predictions', INTERVAL '90 days', if_not_exists => TRUE);

COMMENT ON TABLE predictions IS 'Real-time model predictions for price movements';

-- =============================================================================
-- Prediction Outcomes Table (Predictions matched with actuals)
-- =============================================================================

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    prediction_time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon TEXT NOT NULL,  -- '1s', '5s', '10s', '30s', '60s'

    -- Prediction
    predicted_return DOUBLE PRECISION NOT NULL,

    -- Actual outcome
    actual_return DOUBLE PRECISION NOT NULL,

    -- Evaluation metrics
    error DOUBLE PRECISION NOT NULL,
    squared_error DOUBLE PRECISION NOT NULL,
    absolute_error DOUBLE PRECISION NOT NULL,
    direction_correct BOOLEAN NOT NULL,

    -- Timing
    outcome_time TIMESTAMPTZ NOT NULL,
    evaluated_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (prediction_time, symbol, model_version, horizon)
);

-- Convert to hypertable
SELECT create_hypertable('prediction_outcomes', 'prediction_time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_outcomes_symbol_horizon
ON prediction_outcomes (symbol, horizon, prediction_time DESC);

CREATE INDEX IF NOT EXISTS idx_outcomes_model_version
ON prediction_outcomes (model_version, evaluated_at DESC);

CREATE INDEX IF NOT EXISTS idx_outcomes_direction
ON prediction_outcomes (symbol, horizon, direction_correct);

-- Retention policy
SELECT add_retention_policy('prediction_outcomes', INTERVAL '180 days', if_not_exists => TRUE);

COMMENT ON TABLE prediction_outcomes IS 'Predictions matched with actual outcomes for evaluation';

-- =============================================================================
-- Model Performance Table (Aggregated metrics over time)
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_performance (
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon TEXT NOT NULL,

    -- Sample counts
    num_predictions INTEGER NOT NULL,

    -- Regression metrics
    mse DOUBLE PRECISION NOT NULL,
    rmse DOUBLE PRECISION NOT NULL,
    mae DOUBLE PRECISION NOT NULL,
    r_squared DOUBLE PRECISION,

    -- Classification metrics
    directional_accuracy DOUBLE PRECISION NOT NULL,
    precision_score DOUBLE PRECISION,
    recall_score DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,

    -- Confusion matrix
    true_positives INTEGER,
    true_negatives INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,

    -- Financial metrics
    sharpe_ratio DOUBLE PRECISION,

    -- Computed at
    computed_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (time, symbol, model_version, horizon)
);

-- Convert to hypertable
SELECT create_hypertable('model_performance', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_performance_model_time
ON model_performance (model_version, time DESC);

COMMENT ON TABLE model_performance IS 'Aggregated model performance metrics over time windows';

-- =============================================================================
-- SHAP Values Table (Feature importance explanations)
-- =============================================================================

CREATE TABLE IF NOT EXISTS shap_values (
    prediction_time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horizon TEXT NOT NULL,
    feature_name TEXT NOT NULL,

    -- SHAP value for this feature
    shap_value DOUBLE PRECISION NOT NULL,

    -- Feature value at prediction time
    feature_value DOUBLE PRECISION NOT NULL,

    -- Base value (expected model output)
    base_value DOUBLE PRECISION,

    PRIMARY KEY (prediction_time, symbol, model_version, horizon, feature_name)
);

-- Convert to hypertable
SELECT create_hypertable('shap_values', 'prediction_time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_shap_feature
ON shap_values (feature_name, prediction_time DESC);

CREATE INDEX IF NOT EXISTS idx_shap_model_time
ON shap_values (model_version, prediction_time DESC);

-- Retention policy (SHAP values are large, keep for shorter period)
SELECT add_retention_policy('shap_values', INTERVAL '30 days', if_not_exists => TRUE);

COMMENT ON TABLE shap_values IS 'SHAP feature importance values for model explainability';

-- =============================================================================
-- Views for Easy Querying
-- =============================================================================

-- Recent prediction performance view
CREATE OR REPLACE VIEW recent_performance AS
SELECT
    symbol,
    model_version,
    horizon,
    COUNT(*) as predictions_count,
    AVG(squared_error) as mse,
    SQRT(AVG(squared_error)) as rmse,
    AVG(absolute_error) as mae,
    AVG(CASE WHEN direction_correct THEN 1 ELSE 0 END) * 100 as directional_accuracy,
    MAX(evaluated_at) as last_evaluated
FROM prediction_outcomes
WHERE evaluated_at > NOW() - INTERVAL '24 hours'
GROUP BY symbol, model_version, horizon
ORDER BY symbol, horizon;

COMMENT ON VIEW recent_performance IS 'Performance metrics for last 24 hours';

-- Top important features view (from SHAP)
CREATE OR REPLACE VIEW top_features AS
SELECT
    symbol,
    model_version,
    horizon,
    feature_name,
    AVG(ABS(shap_value)) as avg_abs_shap,
    COUNT(*) as sample_count
FROM shap_values
WHERE prediction_time > NOW() - INTERVAL '7 days'
GROUP BY symbol, model_version, horizon, feature_name
ORDER BY avg_abs_shap DESC;

COMMENT ON VIEW top_features IS 'Most important features based on SHAP values (last 7 days)';

-- =============================================================================
-- Grants
-- =============================================================================

-- Grant permissions to postgres user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- =============================================================================
-- Summary
-- =============================================================================

\echo '==================================================================='
\echo 'LOB Prediction Database Initialized'
\echo '==================================================================='
\echo 'Tables created:'
\echo '  - lob_data: Historical LOB snapshots'
\echo '  - predictions: Real-time model predictions'
\echo '  - prediction_outcomes: Predictions vs actuals'
\echo '  - model_performance: Aggregated metrics'
\echo '  - shap_values: Feature importance'
\echo ''
\echo 'Views created:'
\echo '  - recent_performance: Last 24h metrics'
\echo '  - top_features: Most important features'
\echo '==================================================================='
