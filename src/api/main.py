"""
FastAPI application for LOB prediction system dashboard.

Provides REST and WebSocket APIs for:
- Real-time predictions and metrics
- Training monitoring and control
- Model management
- Data analytics
- SHAP explanations

Usage:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import asyncpg
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="LOB Prediction System API",
    description="Real-time prediction, training, and analytics dashboard",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database pool
db_pool: Optional[asyncpg.Pool] = None


# =============================================================================
# Database Connection
# =============================================================================


@app.on_event("startup")
async def startup():
    """Initialize database connection pool."""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=5,
            max_size=20,
        )
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Close database connection pool."""
    # global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


# =============================================================================
# Pydantic Models
# =============================================================================


class Prediction(BaseModel):
    """Single prediction response."""

    time: datetime
    symbol: str
    model_version: str
    pred_1s: float
    pred_5s: float
    pred_10s: float
    pred_30s: float
    pred_60s: float
    mid_price: float
    spread_bps: float
    volume_imbalance: float
    inference_time_ms: float


class PredictionOutcome(BaseModel):
    """Prediction with actual outcome."""

    prediction_time: datetime
    horizon: str
    predicted_return: float
    actual_return: float
    error: float
    absolute_error: float
    squared_error: float
    direction_correct: bool
    mid_price: float


class AccuracyMetrics(BaseModel):
    """Accuracy metrics for a time window."""

    time_window: str
    horizon: str
    num_predictions: int
    mse: float
    rmse: float
    mae: float
    r_squared: float
    directional_accuracy: float
    mean_predicted: float
    mean_actual: float


class SHAPExplanation(BaseModel):
    """SHAP explanation for a prediction."""

    prediction_time: datetime
    horizon: str
    feature_name: str
    shap_value: float
    feature_value: float
    base_value: float


class DataStatistics(BaseModel):
    """Dataset statistics."""

    symbol: str
    total_samples: int
    time_range_start: datetime
    time_range_end: datetime
    duration_hours: float
    avg_spread_bps: float
    avg_volume_imbalance: float
    samples_per_hour: float


class ModelInfo(BaseModel):
    """Trained model information."""

    model_path: str
    symbol: str
    model_version: str
    created_at: datetime
    num_parameters: int
    train_samples: int
    val_samples: int
    best_val_loss: float
    best_epoch: int


class TrainingStatus(BaseModel):
    """Current training status."""

    is_training: bool
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    train_loss: Optional[float]
    val_loss: Optional[float]
    elapsed_time: Optional[float]
    estimated_time_remaining: Optional[float]


# =============================================================================
# API Endpoints - Live Predictions
# =============================================================================


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "LOB Prediction System API", "version": "1.0.0"}


@app.get("/api/predictions/latest", response_model=List[Prediction])
async def get_latest_predictions(
    symbol: str = "BTCUSDT", model_version: str = "v1.0", limit: int = Query(100, ge=1, le=1000)
):
    """Get latest predictions."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *
            FROM predictions
            WHERE symbol = $1 AND model_version = $2
            ORDER BY time DESC
            LIMIT $3
            """,
            symbol,
            model_version,
            limit,
        )

        return [
            Prediction(
                time=row["time"],
                symbol=row["symbol"],
                model_version=row["model_version"],
                pred_1s=row["pred_1s"],
                pred_5s=row["pred_5s"],
                pred_10s=row["pred_10s"],
                pred_30s=row["pred_30s"],
                pred_60s=row["pred_60s"],
                mid_price=row["mid_price"],
                spread_bps=row["spread_bps"],
                volume_imbalance=row["volume_imbalance"],
                inference_time_ms=row["inference_time_ms"],
            )
            for row in rows
        ]


@app.get("/api/predictions/history", response_model=List[Prediction])
async def get_prediction_history(
    symbol: str = "BTCUSDT",
    model_version: str = "v1.0",
    hours: int = Query(24, ge=1, le=168),  # Max 1 week
    limit: int = Query(1000, ge=1, le=10000),
):
    """Get prediction history for a time window."""
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *
            FROM predictions
            WHERE symbol = $1
              AND model_version = $2
              AND time >= $3
            ORDER BY time DESC
            LIMIT $4
            """,
            symbol,
            model_version,
            start_time,
            limit,
        )

        return [
            Prediction(
                time=row["time"],
                symbol=row["symbol"],
                model_version=row["model_version"],
                pred_1s=row["pred_1s"],
                pred_5s=row["pred_5s"],
                pred_10s=row["pred_10s"],
                pred_30s=row["pred_30s"],
                pred_60s=row["pred_60s"],
                mid_price=row["mid_price"],
                spread_bps=row["spread_bps"],
                volume_imbalance=row["volume_imbalance"],
                inference_time_ms=row["inference_time_ms"],
            )
            for row in rows
        ]


@app.get("/api/lob/latest")
async def get_latest_lob(symbol: str = "BTCUSDT"):
    """Get latest LOB snapshot with all 5 levels."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                time,
                mid_price,
                spread,
                spread_bps,
                bid_price_1, bid_volume_1,
                bid_price_2, bid_volume_2,
                bid_price_3, bid_volume_3,
                bid_price_4, bid_volume_4,
                bid_price_5, bid_volume_5,
                ask_price_1, ask_volume_1,
                ask_price_2, ask_volume_2,
                ask_price_3, ask_volume_3,
                ask_price_4, ask_volume_4,
                ask_price_5, ask_volume_5,
                total_bid_volume,
                total_ask_volume,
                volume_imbalance
            FROM lob_data
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT 1
            """,
            symbol,
        )

        if not row:
            raise HTTPException(status_code=404, detail="No LOB data found")

        return {
            "time": row["time"].isoformat(),
            "mid_price": float(row["mid_price"]),
            "spread": float(row["spread"]),
            "spread_bps": float(row["spread_bps"]),
            "bids": [
                {"price": float(row[f"bid_price_{i}"]), "volume": float(row[f"bid_volume_{i}"])}
                for i in range(1, 6)
            ],
            "asks": [
                {"price": float(row[f"ask_price_{i}"]), "volume": float(row[f"ask_volume_{i}"])}
                for i in range(1, 6)
            ],
            "total_bid_volume": float(row["total_bid_volume"]),
            "total_ask_volume": float(row["total_ask_volume"]),
            "volume_imbalance": float(row["volume_imbalance"]),
        }


# =============================================================================
# API Endpoints - Accuracy Metrics
# =============================================================================


@app.get("/api/metrics/accuracy", response_model=List[AccuracyMetrics])
async def get_accuracy_metrics(
    symbol: str = "BTCUSDT",
    model_version: str = "v1.0",
    time_window: str = Query("1h", regex="^(1m|5m|1h|24h|all)$"),
):
    """
    Get accuracy metrics for different time windows.

    time_window options: 1m, 5m, 1h, 24h, all
    """
    # Determine time cutoff
    cutoff_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "all": timedelta(days=365),  # Arbitrary large value
    }

    cutoff_time = datetime.now(timezone.utc) - cutoff_map[time_window]

    async with db_pool.acquire() as conn:
        # First compute the mean actual return in a CTE
        rows = await conn.fetch(
            """
            WITH stats AS (
                SELECT
                    horizon,
                    AVG(actual_return) as mean_actual
                FROM prediction_outcomes
                WHERE symbol = $1
                  AND model_version = $2
                  AND prediction_time >= $3
                GROUP BY horizon
            )
            SELECT
                po.horizon,
                COUNT(*) as num_predictions,
                AVG((po.predicted_return - po.actual_return)^2) as mse,
                SQRT(AVG((po.predicted_return - po.actual_return)^2)) as rmse,
                AVG(ABS(po.predicted_return - po.actual_return)) as mae,
                1 - (
                    SUM((po.predicted_return - po.actual_return)^2) /
                    NULLIF(SUM((po.actual_return - s.mean_actual)^2), 0)
                ) as r_squared,
                AVG(
                    CASE WHEN po.direction_correct THEN 1.0 ELSE 0.0 END
                ) * 100 as directional_accuracy,
                AVG(po.predicted_return) as mean_predicted,
                s.mean_actual
            FROM prediction_outcomes po
            JOIN stats s ON po.horizon = s.horizon
            WHERE po.symbol = $1
              AND po.model_version = $2
              AND po.prediction_time >= $3
            GROUP BY po.horizon, s.mean_actual
            ORDER BY
                CASE po.horizon
                    WHEN '1s' THEN 1
                    WHEN '5s' THEN 2
                    WHEN '10s' THEN 3
                    WHEN '30s' THEN 4
                    WHEN '60s' THEN 5
                END
            """,
            symbol,
            model_version,
            cutoff_time,
        )

        return [
            AccuracyMetrics(
                time_window=time_window,
                horizon=row["horizon"],
                num_predictions=row["num_predictions"],
                mse=float(row["mse"]) if row["mse"] else 0.0,
                rmse=float(row["rmse"]) if row["rmse"] else 0.0,
                mae=float(row["mae"]) if row["mae"] else 0.0,
                r_squared=float(row["r_squared"]) if row["r_squared"] else 0.0,
                directional_accuracy=(
                    float(row["directional_accuracy"]) if row["directional_accuracy"] else 0.0
                ),
                mean_predicted=float(row["mean_predicted"]) if row["mean_predicted"] else 0.0,
                mean_actual=float(row["mean_actual"]) if row["mean_actual"] else 0.0,
            )
            for row in rows
        ]


@app.get("/api/metrics/accuracy/timeseries")
async def get_accuracy_timeseries(
    symbol: str = "BTCUSDT",
    model_version: str = "v1.0",
    horizon: str = Query("10s", regex="^(1s|5s|10s|30s|60s)$"),
    hours: int = Query(24, ge=1, le=168),
    bucket_minutes: int = Query(5, ge=1, le=60),
):
    """Get accuracy metrics as time series (bucketed)."""
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                time_bucket('{bucket_minutes} minutes', prediction_time) as time_bucket,
                COUNT(*) as num_predictions,
                AVG(ABS(predicted_return - actual_return)) as mae,
                AVG(CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END) * 100 as directional_accuracy
            FROM prediction_outcomes
            WHERE symbol = $1
              AND model_version = $2
              AND horizon = $3
              AND prediction_time >= $4
            GROUP BY time_bucket
            ORDER BY time_bucket DESC
            """,
            symbol,
            model_version,
            horizon,
            start_time,
        )

        return [
            {
                "time": row["time_bucket"].isoformat(),
                "num_predictions": row["num_predictions"],
                "mae": float(row["mae"]) if row["mae"] else 0.0,
                "directional_accuracy": (
                    float(row["directional_accuracy"]) if row["directional_accuracy"] else 0.0
                ),
            }
            for row in rows
        ]


# =============================================================================
# API Endpoints - Outcomes
# =============================================================================


@app.get("/api/outcomes/recent", response_model=List[PredictionOutcome])
async def get_recent_outcomes(
    symbol: str = "BTCUSDT",
    model_version: str = "v1.0",
    horizon: str = Query("10s", regex="^(1s|5s|10s|30s|60s)$"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get recent prediction outcomes."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                po.prediction_time,
                po.horizon,
                po.predicted_return,
                po.actual_return,
                po.error,
                po.absolute_error,
                po.squared_error,
                po.direction_correct,
                p.mid_price
            FROM prediction_outcomes po
            LEFT JOIN predictions p
                ON po.prediction_time = p.time
                AND po.symbol = p.symbol
                AND po.model_version = p.model_version
            WHERE po.symbol = $1
              AND po.model_version = $2
              AND po.horizon = $3
            ORDER BY po.prediction_time DESC
            LIMIT $4
            """,
            symbol,
            model_version,
            horizon,
            limit,
        )

        return [
            PredictionOutcome(
                prediction_time=row["prediction_time"],
                horizon=row["horizon"],
                predicted_return=row["predicted_return"],
                actual_return=row["actual_return"],
                error=row["error"],
                absolute_error=row["absolute_error"],
                squared_error=row["squared_error"],
                direction_correct=row["direction_correct"],
                mid_price=float(row["mid_price"]) if row["mid_price"] else 0.0,
            )
            for row in rows
        ]


# =============================================================================
# API Endpoints - SHAP Explanations
# =============================================================================


@app.get("/api/shap/feature-importance")
async def get_feature_importance(
    symbol: str = "BTCUSDT",
    model_version: str = "v1.0",
    horizon: str = Query("10s", regex="^(1s|5s|10s|30s|60s)$"),
    top_k: int = Query(10, ge=1, le=43),
):
    """Get top-k most important features for a horizon."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                feature_name,
                AVG(ABS(shap_value)) as avg_importance,
                AVG(shap_value) as avg_direction,
                STDDEV(shap_value) as variability,
                COUNT(*) as sample_count
            FROM shap_values
            WHERE symbol = $1
              AND model_version = $2
              AND horizon = $3
            GROUP BY feature_name
            ORDER BY avg_importance DESC
            LIMIT $4
            """,
            symbol,
            model_version,
            horizon,
            top_k,
        )

        return [
            {
                "feature_name": row["feature_name"],
                "importance": float(row["avg_importance"]),
                "direction": float(row["avg_direction"]),
                "variability": float(row["variability"]) if row["variability"] else 0.0,
                "sample_count": row["sample_count"],
            }
            for row in rows
        ]


@app.get("/api/shap/explanation", response_model=List[SHAPExplanation])
async def get_shap_explanation(
    prediction_time: datetime,
    symbol: str = "BTCUSDT",
    model_version: str = "v1.0",
    horizon: str = Query("10s", regex="^(1s|5s|10s|30s|60s)$"),
):
    """Get SHAP explanation for a specific prediction."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT *
            FROM shap_values
            WHERE prediction_time = $1
              AND symbol = $2
              AND model_version = $3
              AND horizon = $4
            ORDER BY ABS(shap_value) DESC
            """,
            prediction_time,
            symbol,
            model_version,
            horizon,
        )

        if not rows:
            raise HTTPException(status_code=404, detail="SHAP explanation not found")

        return [
            SHAPExplanation(
                prediction_time=row["prediction_time"],
                horizon=row["horizon"],
                feature_name=row["feature_name"],
                shap_value=row["shap_value"],
                feature_value=row["feature_value"],
                base_value=row["base_value"],
            )
            for row in rows
        ]


# =============================================================================
# API Endpoints - Data Statistics
# =============================================================================


@app.get("/api/data/statistics", response_model=DataStatistics)
async def get_data_statistics(symbol: str = "BTCUSDT"):
    """Get dataset statistics."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                symbol,
                COUNT(*) as total_samples,
                MIN(time) as time_range_start,
                MAX(time) as time_range_end,
                EXTRACT(EPOCH FROM (MAX(time) - MIN(time))) / 3600 as duration_hours,
                AVG(spread_bps) as avg_spread_bps,
                AVG(volume_imbalance) as avg_volume_imbalance
            FROM lob_data
            WHERE symbol = $1
            GROUP BY symbol
            """,
            symbol,
        )

        if not row:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        samples_per_hour = (
            row["total_samples"] / row["duration_hours"] if row["duration_hours"] > 0 else 0
        )

        return DataStatistics(
            symbol=row["symbol"],
            total_samples=row["total_samples"],
            time_range_start=row["time_range_start"],
            time_range_end=row["time_range_end"],
            duration_hours=row["duration_hours"],
            avg_spread_bps=float(row["avg_spread_bps"]),
            avg_volume_imbalance=float(row["avg_volume_imbalance"]),
            samples_per_hour=samples_per_hour,
        )


@app.get("/api/data/collection-status")
async def get_collection_status(symbol: str = "BTCUSDT"):
    """Check if data collection is active."""
    async with db_pool.acquire() as conn:
        # Check if we have data from the last minute
        row = await conn.fetchrow(
            """
            SELECT MAX(time) as last_update
            FROM lob_data
            WHERE symbol = $1
            """,
            symbol,
        )

        if not row or not row["last_update"]:
            return {
                "is_collecting": False,
                "last_update": None,
                "seconds_since_update": None,
            }

        last_update = row["last_update"]
        seconds_since = (datetime.now(timezone.utc) - last_update).total_seconds()
        is_collecting = seconds_since < 60  # Consider active if data within last minute

        return {
            "is_collecting": is_collecting,
            "last_update": last_update.isoformat(),
            "seconds_since_update": seconds_since,
        }


# =============================================================================
# API Endpoints - Model Management
# =============================================================================


@app.get("/api/models/list")
async def list_models():
    """List all available trained models."""
    # This would scan the models directory
    # For now, return mock data
    # TODO: Implement actual model scanning
    return [
        {
            "model_path": "data/models/BTCUSDT/best_model.pth",
            "symbol": "BTCUSDT",
            "model_version": "v1.0",
            "created_at": "2024-11-10T12:00:00Z",
            "num_parameters": 895109,
            "architecture": "TCN",
        }
    ]


# =============================================================================
# WebSocket Endpoints
# =============================================================================


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")


manager = ConnectionManager()


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """
    WebSocket endpoint for real-time predictions.

    Streams new predictions as they are generated.
    """
    await manager.connect(websocket)

    try:
        # Send initial data
        await websocket.send_json(
            {"type": "connected", "message": "Connected to predictions stream"}
        )

        # Keep connection alive and stream predictions
        while True:
            # Query latest prediction
            async with db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT *
                    FROM predictions
                    WHERE symbol = 'BTCUSDT'
                    ORDER BY time DESC
                    LIMIT 1
                    """
                )

                if row:
                    prediction = {
                        "type": "prediction",
                        "data": {
                            "time": row["time"].isoformat(),
                            "symbol": row["symbol"],
                            "pred_1s": float(row["pred_1s"]),
                            "pred_5s": float(row["pred_5s"]),
                            "pred_10s": float(row["pred_10s"]),
                            "pred_30s": float(row["pred_30s"]),
                            "pred_60s": float(row["pred_60s"]),
                            "mid_price": float(row["mid_price"]),
                            "spread_bps": float(row["spread_bps"]),
                            "volume_imbalance": float(row["volume_imbalance"]),
                        },
                    }
                    await websocket.send_json(prediction)

            # Wait before next update (1 second)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics.

    Streams accuracy metrics as they are updated.
    """
    await manager.connect(websocket)

    try:
        await websocket.send_json({"type": "connected", "message": "Connected to metrics stream"})

        while True:
            # Query latest metrics (1min window)
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=1)

            async with db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        horizon,
                        COUNT(*) as num_predictions,
                        AVG(ABS(predicted_return - actual_return)) as mae,
                        AVG(
                            CASE WHEN direction_correct THEN 1.0 ELSE 0.0 END
                        ) * 100 as directional_accuracy
                    FROM prediction_outcomes
                    WHERE symbol = 'BTCUSDT'
                    AND prediction_time >= $1
                    GROUP BY horizon
                    """,
                    cutoff_time,
                )

                if rows:
                    metrics = {
                        "type": "metrics",
                        "data": {
                            "time_window": "1m",
                            "horizons": [
                                {
                                    "horizon": row["horizon"],
                                    "num_predictions": row["num_predictions"],
                                    "mae": float(row["mae"]) if row["mae"] else 0.0,
                                    "directional_accuracy": (
                                        float(row["directional_accuracy"])
                                        if row["directional_accuracy"]
                                        else 0.0
                                    ),
                                }
                                for row in rows
                            ],
                        },
                    }
                    await websocket.send_json(metrics)

            await asyncio.sleep(5)  # Update every 5 seconds

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
