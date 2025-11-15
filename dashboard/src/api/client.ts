import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface Prediction {
  time: string;
  symbol: string;
  model_version: string;
  pred_1s: number;
  pred_5s: number;
  pred_10s: number;
  pred_30s: number;
  pred_60s: number;
  mid_price: number;
  spread_bps: number;
  volume_imbalance: number;
  inference_time_ms: number;
}

export interface AccuracyMetrics {
  time_window: string;
  horizon: string;
  num_predictions: number;
  mse: number;
  rmse: number;
  mae: number;
  r_squared: number;
  directional_accuracy: number;
  mean_predicted: number;
  mean_actual: number;
}

export interface PredictionOutcome {
  prediction_time: string;
  horizon: string;
  predicted_return: number;
  actual_return: number;
  error: number;
  absolute_error: number;
  squared_error: number;
  direction_correct: boolean;
  mid_price: number;
}

export interface FeatureImportance {
  feature_name: string;
  importance: number;
  direction: number;
  variability: number;
  sample_count: number;
}

export interface DataStatistics {
  symbol: string;
  total_samples: number;
  time_range_start: string;
  time_range_end: string;
  duration_hours: number;
  avg_spread_bps: number;
  avg_volume_imbalance: number;
  samples_per_hour: number;
}

export interface CollectionStatus {
  is_collecting: boolean;
  last_update: string | null;
  seconds_since_update: number | null;
}

export interface LOBSnapshot {
  time: string;
  mid_price: number;
  spread: number;
  spread_bps: number;
  bids: Array<{ price: number; volume: number }>;
  asks: Array<{ price: number; volume: number }>;
  total_bid_volume: number;
  total_ask_volume: number;
  volume_imbalance: number;
}

// API Functions
export const apiClient = {
  // Predictions
  getLatestPredictions: (limit: number = 100) =>
    api.get<Prediction[]>(`/api/predictions/latest?limit=${limit}`),

  getPredictionHistory: (hours: number = 24, limit: number = 1000) =>
    api.get<Prediction[]>(`/api/predictions/history?hours=${hours}&limit=${limit}`),

    // LOB
  getLatestLOB: () =>
    api.get<LOBSnapshot>('/api/lob/latest'),

  // Metrics
  getAccuracyMetrics: (timeWindow: '1m' | '5m' | '1h' | '24h' | 'all' = '1h') =>
    api.get<AccuracyMetrics[]>(`/api/metrics/accuracy?time_window=${timeWindow}`),

  getAccuracyTimeseries: (horizon: string, hours: number = 24, bucketMinutes: number = 5) =>
    api.get(`/api/metrics/accuracy/timeseries?horizon=${horizon}&hours=${hours}&bucket_minutes=${bucketMinutes}`),

  // Outcomes
  getRecentOutcomes: (horizon: string = '10s', limit: number = 100) =>
    api.get<PredictionOutcome[]>(`/api/outcomes/recent?horizon=${horizon}&limit=${limit}`),

  // SHAP
  getFeatureImportance: (horizon: string = '10s', topK: number = 10) =>
    api.get<FeatureImportance[]>(`/api/shap/feature-importance?horizon=${horizon}&top_k=${topK}`),

  // Data
  getDataStatistics: () =>
    api.get<DataStatistics>('/api/data/statistics'),

  getCollectionStatus: () =>
    api.get<CollectionStatus>('/api/data/collection-status'),

  // Health
  healthCheck: () =>
    api.get('/'),
};

// WebSocket connection
export const createWebSocket = (endpoint: string) => {
  return new WebSocket(`ws://localhost:8000${endpoint}`);
};
