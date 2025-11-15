import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  ResponsiveContainer, Legend, RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar, ScatterChart, Scatter, ReferenceLine
} from 'recharts';
import { Info, TrendingUp, TrendingDown, Target, Zap } from 'lucide-react';
import { useState } from 'react';
import { format } from 'date-fns';

const Tooltip = ({ children, content }: { children: React.ReactNode; content: string }) => {
  const [show, setShow] = useState(false);

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="cursor-help"
      >
        {children}
      </div>
      {show && (
        <div className="absolute z-10 w-96 p-4 text-sm bg-slate-700 text-slate-200 rounded-lg shadow-lg -top-2 left-8 border border-slate-600 max-h-80 overflow-y-auto">
          {content}
          <div className="absolute w-2 h-2 bg-slate-700 border-l border-b border-slate-600 transform rotate-45 -left-1 top-4"></div>
        </div>
      )}
    </div>
  );
};

// Calculate linear regression
const calculateRegression = (data: Array<{ predicted: number; actual: number }>) => {
  const n = data.length;
  if (n === 0) return { slope: 0, intercept: 0, r2: 0 };

  const sumX = data.reduce((sum, d) => sum + d.predicted, 0);
  const sumY = data.reduce((sum, d) => sum + d.actual, 0);
  const sumXY = data.reduce((sum, d) => sum + d.predicted * d.actual, 0);
  const sumX2 = data.reduce((sum, d) => sum + d.predicted * d.predicted, 0);
  const sumY2 = data.reduce((sum, d) => sum + d.actual * d.actual, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Calculate R²
  const meanY = sumY / n;
  const ssTotal = data.reduce((sum, d) => sum + Math.pow(d.actual - meanY, 2), 0);
  const ssResidual = data.reduce((sum, d) => {
    const predicted = slope * d.predicted + intercept;
    return sum + Math.pow(d.actual - predicted, 2);
  }, 0);
  const r2 = 1 - (ssResidual / ssTotal);

  return { slope, intercept, r2 };
};

const COLORS = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'];

export default function AdvancedAnalyticsDashboard() {
  const [selectedHorizon, setSelectedHorizon] = useState('10s');

  // Fetch data for selected horizon
  const { data: metricsAll } = useQuery({
    queryKey: ['accuracy', 'all'],
    queryFn: () => apiClient.getAccuracyMetrics('all'),
  });

  const { data: outcomes } = useQuery({
    queryKey: ['outcomes', 'recent', selectedHorizon],
    queryFn: () => apiClient.getRecentOutcomes(selectedHorizon, 100),
  });

  const { data: featureImportance } = useQuery({
    queryKey: ['shap', 'feature-importance', selectedHorizon],
    queryFn: () => apiClient.getFeatureImportance(selectedHorizon, 10),
  });

  const { data: timeseriesData } = useQuery({
    queryKey: ['accuracy', 'timeseries', selectedHorizon],
    queryFn: () => apiClient.getAccuracyTimeseries(selectedHorizon, 24, 15),
  });

  // Process data
  const horizonComparisonData = metricsAll?.data.map(m => ({
    horizon: m.horizon,
    accuracy: m.directional_accuracy,
    mae: m.mae,
    rmse: m.rmse,
    r_squared: m.r_squared * 100,
  })) || [];

  const errorDistributionData = outcomes?.data.slice(0, 50).map(o => ({
    predicted: o.predicted_return,
    actual: o.actual_return,
    error: o.absolute_error,
  })) || [];

  const directionPieData = outcomes?.data ? [
    { name: 'Correct', value: outcomes.data.filter(o => o.direction_correct).length, color: '#10b981' },
    { name: 'Wrong', value: outcomes.data.filter(o => !o.direction_correct).length, color: '#ef4444' },
  ] : [];

  const radarData = featureImportance?.data.slice(0, 6).map(f => ({
    feature: f.feature_name.slice(0, 15),
    importance: (f.importance * 1000).toFixed(2),
  })) || [];

  // Calculate regression for scatter plot
  const regression = calculateRegression(errorDistributionData);
  const minPredicted = Math.min(...errorDistributionData.map(d => d.predicted));
  const maxPredicted = Math.max(...errorDistributionData.map(d => d.predicted));
  const regressionLine = [
    { predicted: minPredicted, actual: regression.slope * minPredicted + regression.intercept },
    { predicted: maxPredicted, actual: regression.slope * maxPredicted + regression.intercept },
  ];

  const currentMetric = metricsAll?.data.find(m => m.horizon === selectedHorizon);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">Advanced Analytics</h1>
            <p className="text-sm text-slate-400">
              Comprehensive performance analysis, error distributions, and feature impacts
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-400 mb-2 text-right">Select Prediction Horizon:</p>
            <div className="flex gap-2">
              {['1s', '5s', '10s', '30s', '60s'].map((horizon) => (
                <button
                  key={horizon}
                  onClick={() => setSelectedHorizon(horizon)}
                  className={`px-4 py-2 rounded-lg transition-colors ${
                    selectedHorizon === horizon
                      ? 'bg-blue-500 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {horizon}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Key Metrics Cards with Tooltips */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {currentMetric && (
          <>
            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-2">
                <div className="h-10 w-10 rounded-full bg-green-500/20 flex items-center justify-center">
                  <Target className="h-5 w-5 text-green-400" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-1">
                    <p className="text-xs text-slate-400">Directional Accuracy</p>
                    <Tooltip content="Percentage of predictions where the direction (up/down) was correctly predicted. 50% = random chance, 100% = perfect. This is the most important metric for trading as getting the direction right matters more than the exact magnitude.">
                      <Info className="h-3 w-3 text-slate-500" />
                    </Tooltip>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {currentMetric.directional_accuracy.toFixed(1)}%
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-500">
                {currentMetric.num_predictions} predictions
              </p>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-2">
                <div className="h-10 w-10 rounded-full bg-blue-500/20 flex items-center justify-center">
                  <Zap className="h-5 w-5 text-blue-400" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-1">
                    <p className="text-xs text-slate-400">MAE</p>
                    <Tooltip content="Mean Absolute Error: Average absolute difference between predicted and actual returns in percentage points. Lower is better. Example: MAE of 2% means predictions are off by 2% on average. Good: <5%, Acceptable: <10%, Poor: >20%">
                      <Info className="h-3 w-3 text-slate-500" />
                    </Tooltip>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {currentMetric.mae.toFixed(2)}%
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-500">Mean Absolute Error</p>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-2">
                <div className="h-10 w-10 rounded-full bg-purple-500/20 flex items-center justify-center">
                  <TrendingUp className="h-5 w-5 text-purple-400" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-1">
                    <p className="text-xs text-slate-400">RMSE</p>
                    <Tooltip content="Root Mean Squared Error: Similar to MAE but penalizes large errors more heavily. Square root of average squared errors. Always ≥ MAE. If RMSE >> MAE, model has some very bad predictions (outliers). Good: Close to MAE, Poor: Much larger than MAE">
                      <Info className="h-3 w-3 text-slate-500" />
                    </Tooltip>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {currentMetric.rmse.toFixed(2)}%
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-500">Root Mean Squared Error</p>
            </div>

            <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
              <div className="flex items-center gap-3 mb-2">
                <div className="h-10 w-10 rounded-full bg-orange-500/20 flex items-center justify-center">
                  <TrendingDown className="h-5 w-5 text-orange-400" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-1">
                    <p className="text-xs text-slate-400">R²</p>
                    <Tooltip content="R-squared (Coefficient of Determination): Proportion of variance in actual returns explained by predictions. Range: -∞ to 1. 1.0 = perfect predictions, 0 = predictions no better than average, <0 = worse than predicting the mean. Good: >0.3, Acceptable: 0.1-0.3, Poor: <0.1">
                      <Info className="h-3 w-3 text-slate-500" />
                    </Tooltip>
                  </div>
                  <p className="text-2xl font-bold text-white">
                    {(currentMetric.r_squared * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-500">Coefficient of Determination</p>
            </div>
          </>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Horizon Comparison */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-lg font-semibold text-white">Performance Across All Horizons</h2>
            <Tooltip content="Compare directional accuracy across all prediction horizons (1s to 60s). Shows which timeframe the model predicts best. Generally, longer horizons are easier to predict as short-term noise averages out.">
              <Info className="h-4 w-4 text-slate-500" />
            </Tooltip>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={horizonComparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="horizon" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: '#94a3b8' }} />
              <RechartsTooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
              />
              <Legend />
              <Bar dataKey="accuracy" fill="#10b981" name="Accuracy (%)" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Directional Accuracy Pie */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-lg font-semibold text-white">
              Direction Prediction ({selectedHorizon})
            </h2>
            <Tooltip content="Breakdown of correct vs incorrect direction predictions for the selected horizon. Green = predicted direction matched actual direction. Red = predicted opposite direction. For trading, getting direction right is crucial.">
              <Info className="h-4 w-4 text-slate-500" />
            </Tooltip>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={directionPieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {directionPieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <RechartsTooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Accuracy Over Time */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-lg font-semibold text-white">
              Accuracy Over Time ({selectedHorizon})
            </h2>
            <Tooltip content="How prediction accuracy and error metrics change over the last 24 hours in 15-minute buckets. Helps identify if model performance is stable or degrading. Sudden drops may indicate market regime changes or data quality issues.">
              <Info className="h-4 w-4 text-slate-500" />
            </Tooltip>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeseriesData?.data || []}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis
                dataKey="time"
                stroke="#94a3b8"
                tick={{ fontSize: 10 }}
                tickFormatter={(time) => format(new Date(time), 'HH:mm')}
              />
              <YAxis stroke="#94a3b8" />
              <RechartsTooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                labelFormatter={(time) => format(new Date(time), 'HH:mm:ss')}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="directional_accuracy"
                stroke="#10b981"
                name="Accuracy (%)"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="mae"
                stroke="#ef4444"
                name="MAE (%)"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Feature Importance Radar */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-lg font-semibold text-white">
              Top Features Radar ({selectedHorizon})
            </h2>
            <Tooltip content="Radar chart showing relative importance of top 6 features using SHAP values (scaled ×1000 for visibility). Larger polygon = model relies more on these features. Switch to Analytics tab to see detailed feature explanations.">
              <Info className="h-4 w-4 text-slate-500" />
            </Tooltip>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#475569" />
              <PolarAngleAxis dataKey="feature" stroke="#94a3b8" tick={{ fontSize: 10 }} />
              <PolarRadiusAxis stroke="#94a3b8" />
              <Radar
                name="Importance (×1000)"
                dataKey="importance"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.6}
              />
              <RechartsTooltip />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Predicted vs Actual Scatter with Regression */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-lg font-semibold text-white">
              Predicted vs Actual Returns ({selectedHorizon})
            </h2>
            <Tooltip content={`Scatter plot: Each dot = one prediction. Perfect predictions lie on the diagonal (red dashed line). Blue line = linear regression fit (R²=${regression.r2.toFixed(3)}). Slope=${regression.slope.toFixed(3)} (ideal=1.0). Intercept=${regression.intercept.toFixed(3)} (ideal=0). Tight clustering around diagonal = good calibration. Wide scatter = high uncertainty.`}>
              <Info className="h-4 w-4 text-slate-500" />
            </Tooltip>
          </div>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis
                type="number"
                dataKey="predicted"
                name="Predicted"
                stroke="#94a3b8"
                label={{ value: 'Predicted Return (%)', position: 'bottom', fill: '#94a3b8' }}
              />
              <YAxis
                type="number"
                dataKey="actual"
                name="Actual"
                stroke="#94a3b8"
                label={{ value: 'Actual Return (%)', angle: -90, position: 'left', fill: '#94a3b8' }}
              />
              <RechartsTooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                formatter={(value: any) => `${value.toFixed(3)}%`}
              />
              <Scatter
                name="Predictions"
                data={errorDistributionData}
                fill="#3b82f6"
                fillOpacity={0.6}
              />
              {/* Perfect prediction line (y=x) */}
              <ReferenceLine
                segment={[
                  { x: minPredicted, y: minPredicted },
                  { x: maxPredicted, y: maxPredicted }
                ]}
                stroke="#ef4444"
                strokeDasharray="5 5"
                strokeWidth={2}
                label={{ value: 'Perfect', position: 'top', fill: '#ef4444' }}
              />
              {/* Regression line */}
              <ReferenceLine
                segment={regressionLine.map(p => ({ x: p.predicted, y: p.actual }))}
                stroke="#10b981"
                strokeWidth={2}
                label={{
                  value: `Fit: R²=${regression.r2.toFixed(3)}`,
                  position: 'bottom',
                  fill: '#10b981'
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Error Distribution */}
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <h2 className="text-lg font-semibold text-white">
              Error Distribution ({selectedHorizon})
            </h2>
            <Tooltip content="Distribution of absolute prediction errors. Each bar = one prediction, sorted by error magnitude. Green (<1%) = excellent, Orange (1-5%) = acceptable, Red (>5%) = poor. Ideal distribution: mostly green bars (low errors). Many red bars indicate poor calibration or high uncertainty predictions.">
              <Info className="h-4 w-4 text-slate-500" />
            </Tooltip>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={errorDistributionData.slice(0, 30)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis
                dataKey="error"
                stroke="#94a3b8"
                tickFormatter={(val) => val.toFixed(1)}
                label={{ value: 'Absolute Error (%)', position: 'bottom', fill: '#94a3b8' }}
              />
              <YAxis stroke="#94a3b8" hide />
              <RechartsTooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                formatter={(value: any) => `${value.toFixed(3)}%`}
              />
              <Bar dataKey="error" fill="#8b5cf6">
                {errorDistributionData.slice(0, 30).map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.error < 1 ? '#10b981' : entry.error < 5 ? '#f59e0b' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Statistical Summary with Tooltips */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <h2 className="text-lg font-semibold text-white">Statistical Summary ({selectedHorizon})</h2>
          <Tooltip content="Comprehensive statistical breakdown of model performance for the selected horizon. Shows prediction counts, success rates, average returns, and bias metrics.">
            <Info className="h-4 w-4 text-slate-500" />
          </Tooltip>
        </div>
        {currentMetric && (() => {
          const correctPredictions = outcomes?.data.filter(o => o.direction_correct).length || 0;
          const totalPredictions = outcomes?.data.length || 1;
          const avgPredicted = currentMetric.mean_predicted;
          const avgActual = currentMetric.mean_actual;
          const bias = avgPredicted - avgActual;

          return (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">Total Predictions</p>
                  <Tooltip content="Total number of predictions evaluated for this horizon. More predictions = more reliable statistics.">
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className="text-xl font-bold text-white">{currentMetric.num_predictions}</p>
              </div>
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">Correct Directions</p>
                  <Tooltip content="Number of predictions where direction (up/down) was correct. This is what matters for profitable trading.">
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className="text-xl font-bold text-green-400">{correctPredictions}</p>
              </div>
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">Wrong Directions</p>
                  <Tooltip content="Number of predictions where direction was wrong. These would lead to losing trades.">
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className="text-xl font-bold text-red-400">{totalPredictions - correctPredictions}</p>
              </div>
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">Success Rate</p>
                  <Tooltip content="Percentage of correct directional predictions. Same as Directional Accuracy. >60% = good, >70% = excellent.">
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className="text-xl font-bold text-white">
                  {((correctPredictions / totalPredictions) * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">Avg Predicted Return</p>
                  <Tooltip content="Average of all predicted returns. If consistently positive/negative, model may have a directional bias. Ideally should be close to 0 unless there's a market trend.">
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className={`text-xl font-bold ${avgPredicted > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {avgPredicted > 0 ? '+' : ''}{avgPredicted.toFixed(3)}%
                </p>
              </div>
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">Avg Actual Return</p>
                  <Tooltip content="Average of all actual returns observed. Shows if market was trending up/down during evaluation period.">
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className={`text-xl font-bold ${avgActual > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {avgActual > 0 ? '+' : ''}{avgActual.toFixed(3)}%
                </p>
              </div>
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">Prediction Bias</p>
                  <Tooltip content={`Difference between avg predicted and avg actual returns. Bias = ${bias.toFixed(3)}%. Positive = model over-predicts (too optimistic). Negative = under-predicts (too pessimistic). Ideally should be near 0.`}>
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className={`text-xl font-bold ${bias > 0 ? 'text-yellow-400' : 'text-blue-400'}`}>
                  {(avgPredicted - avgActual).toFixed(3)}%
                </p>
              </div>
              <div className="bg-slate-700/30 p-4 rounded">
                <div className="flex items-center gap-1 mb-1">
                  <p className="text-xs text-slate-400">R² Score</p>
                  <Tooltip content="How well predictions explain variance in actual returns. 100% = perfect, 0% = no better than mean, <0% = worse than predicting mean. For price prediction, >30% is good.">
                    <Info className="h-3 w-3 text-slate-500" />
                  </Tooltip>
                </div>
                <p className="text-xl font-bold text-white">
                  {(currentMetric.r_squared * 100).toFixed(2)}%
                </p>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
