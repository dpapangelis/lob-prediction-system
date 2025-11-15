import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { Activity, TrendingUp, TrendingDown, Clock, Info, AlertCircle, CheckCircle, XCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { format } from 'date-fns';
import { useState } from 'react';

// Tooltip component
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
        <div className="absolute z-10 w-64 p-3 text-sm bg-slate-700 text-slate-200 rounded-lg shadow-lg -top-2 left-8 border border-slate-600">
          {content}
          <div className="absolute w-2 h-2 bg-slate-700 border-l border-b border-slate-600 transform rotate-45 -left-1 top-4"></div>
        </div>
      )}
    </div>
  );
};

// Collapsible Section component
const CollapsibleSection = ({
  title,
  children,
  defaultOpen = true,
  tooltip
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  tooltip?: string;
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-700/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          {tooltip && (
            <Tooltip content={tooltip}>
              <Info className="h-4 w-4 text-slate-500" />
            </Tooltip>
          )}
        </div>
        {isOpen ? (
          <ChevronUp className="h-5 w-5 text-slate-400" />
        ) : (
          <ChevronDown className="h-5 w-5 text-slate-400" />
        )}
      </button>
      {isOpen && <div className="px-6 pb-6">{children}</div>}
    </div>
  );
};

// Helper functions
const getPredictedPrice = (currentPrice: number, returnPercent: number) => {
  return currentPrice * (1 + returnPercent / 100);
};

const formatPreciseTime = (dateString: string) => {
  const date = new Date(dateString);
  return format(date, 'HH:mm:ss.SSS');
};

const getMarketTendency = (volumeImbalance: number, spread: number) => {
  if (volumeImbalance > 0.5) {
    return { label: 'Strong Buy Pressure', color: 'text-green-400', icon: TrendingUp };
  } else if (volumeImbalance < -0.5) {
    return { label: 'Strong Sell Pressure', color: 'text-red-400', icon: TrendingDown };
  } else if (spread < 0.001) {
    return { label: 'High Liquidity', color: 'text-blue-400', icon: Activity };
  } else if (spread > 0.005) {
    return { label: 'Low Liquidity', color: 'text-yellow-400', icon: AlertCircle };
  } else {
    return { label: 'Balanced Market', color: 'text-slate-400', icon: Activity };
  }
};

const generateInsights = (accuracyMetrics: any[]) => {
  const insights: { type: 'success' | 'warning' | 'info'; message: string }[] = [];

  if (!accuracyMetrics || accuracyMetrics.length === 0) {
    return [{ type: 'info' as const, message: 'Collecting metrics... Check back in a minute!' }];
  }

  const sortedByAccuracy = [...accuracyMetrics].sort((a, b) => b.directional_accuracy - a.directional_accuracy);
  const best = sortedByAccuracy[0];
  const worst = sortedByAccuracy[sortedByAccuracy.length - 1];

  if (best.directional_accuracy > 70) {
    insights.push({
      type: 'success',
      message: `🎯 ${best.horizon} predictions are highly accurate (${best.directional_accuracy.toFixed(1)}%)! Use this timeframe for trading decisions.`
    });
  } else if (best.directional_accuracy > 50) {
    insights.push({
      type: 'info',
      message: `✓ ${best.horizon} predictions show the best performance (${best.directional_accuracy.toFixed(1)}%), better than random chance.`
    });
  }

  if (worst.directional_accuracy < 30) {
    insights.push({
      type: 'warning',
      message: `⚠️ ${worst.horizon} predictions are struggling (${worst.directional_accuracy.toFixed(1)}%). Avoid using this timeframe.`
    });
  }

  const highMAE = accuracyMetrics.filter(m => m.mae > 100);
  if (highMAE.length > 0) {
    insights.push({
      type: 'warning',
      message: `⚠️ Model is overconfident - predictions show large magnitude errors (MAE > 100%). Trust the direction, not the exact percentage.`
    });
  }

  const lowSamples = accuracyMetrics.filter(m => m.num_predictions < 10);
  if (lowSamples.length > 0) {
    insights.push({
      type: 'info',
      message: `ℹ️ Some horizons have few predictions (< 10 samples). Metrics will stabilize with more data.`
    });
  }

  return insights;
};

export default function LiveDashboard() {
  const [tendencyWindow, setTendencyWindow] = useState<'live' | '1m' | '1h'>('live');

  // Fetch latest predictions
  const { data: predictions, isLoading: predictionsLoading } = useQuery({
    queryKey: ['predictions', 'latest'],
    queryFn: () => apiClient.getLatestPredictions(100),
  });

  // Fetch accuracy metrics
  const { data: metrics1m } = useQuery({
    queryKey: ['accuracy', '1m'],
    queryFn: () => apiClient.getAccuracyMetrics('1m'),
  });

  const { data: metrics1h } = useQuery({
    queryKey: ['accuracy', '1h'],
    queryFn: () => apiClient.getAccuracyMetrics('1h'),
  });

  const { data: metrics24h } = useQuery({
    queryKey: ['accuracy', '24h'],
    queryFn: () => apiClient.getAccuracyMetrics('24h'),
  });

  const { data: metricsAll } = useQuery({
    queryKey: ['accuracy', 'all'],
    queryFn: () => apiClient.getAccuracyMetrics('all'),
  });

  const { data: collectionStatus } = useQuery({
    queryKey: ['collection-status'],
    queryFn: () => apiClient.getCollectionStatus(),
    refetchInterval: 5000,
  });

  const { data: outcomes } = useQuery({
    queryKey: ['outcomes', 'recent', '10s'],
    queryFn: () => apiClient.getRecentOutcomes('10s', 10),
  });

  const latestPrediction = predictions?.data[0];
  const currentMetrics = metrics1m?.data || [];
  const insights = generateInsights(currentMetrics);

  // Calculate market tendency based on selected window
  const getTendencyStats = () => {
    if (!predictions?.data) return null;

    let relevantPredictions;
    if (tendencyWindow === 'live') {
      relevantPredictions = predictions.data.slice(0, 1);
    } else if (tendencyWindow === '1m') {
      relevantPredictions = predictions.data.slice(0, 60);
    } else {
      relevantPredictions = predictions.data;
    }

    const avgVolImb = relevantPredictions.reduce((sum, p) => sum + p.volume_imbalance, 0) / relevantPredictions.length;
    const avgSpread = relevantPredictions.reduce((sum, p) => sum + p.spread_bps, 0) / relevantPredictions.length;

    return {
      volumeImbalance: avgVolImb,
      spread: avgSpread,
      tendency: getMarketTendency(avgVolImb, avgSpread)
    };
  };

  const tendencyStats = getTendencyStats();

  return (
    <div className="space-y-6">
      {/* Status Bar */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <p className="text-sm text-slate-400">Collection Status</p>
                <Tooltip content="Shows whether the system is actively collecting live market data from Binance. Data is collected every 1-2 seconds.">
                  <Info className="h-4 w-4 text-slate-500" />
                </Tooltip>
              </div>
              <p className="text-2xl font-bold text-white mt-1">
                {collectionStatus?.data.is_collecting ? 'Active' : 'Inactive'}
              </p>
              {collectionStatus?.data.last_update && (
                <p className="text-xs text-slate-500 mt-1">
                  {formatPreciseTime(collectionStatus.data.last_update)}
                </p>
              )}
            </div>
            <div
              className={`h-12 w-12 rounded-full flex items-center justify-center ${
                collectionStatus?.data.is_collecting
                  ? 'bg-green-500/20 text-green-500'
                  : 'bg-red-500/20 text-red-500'
              }`}
            >
              <Activity className="h-6 w-6" />
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <p className="text-sm text-slate-400">Latest Prediction (10s)</p>
                <Tooltip content="Model's prediction for price movement in the next 10 seconds. Positive = price will rise, Negative = price will fall.">
                  <Info className="h-4 w-4 text-slate-500" />
                </Tooltip>
              </div>
              <p className="text-2xl font-bold text-white mt-1">
                {latestPrediction
                  ? `${latestPrediction.pred_10s > 0 ? '+' : ''}${latestPrediction.pred_10s.toFixed(3)}%`
                  : '--'}
              </p>
              {latestPrediction && (
                <p className="text-xs text-slate-500 mt-1">
                  ${latestPrediction.mid_price.toFixed(2)} → $
                  {getPredictedPrice(latestPrediction.mid_price, latestPrediction.pred_10s).toFixed(2)}
                </p>
              )}
            </div>
            <div
              className={`h-12 w-12 rounded-full flex items-center justify-center ${
                latestPrediction && latestPrediction.pred_10s > 0
                  ? 'bg-green-500/20 text-green-500'
                  : 'bg-red-500/20 text-red-500'
              }`}
            >
              {latestPrediction && latestPrediction.pred_10s > 0 ? (
                <TrendingUp className="h-6 w-6" />
              ) : (
                <TrendingDown className="h-6 w-6" />
              )}
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <p className="text-sm text-slate-400">Inference Time</p>
                <Tooltip content="How long it takes the model to generate a prediction. Lower is better for real-time trading.">
                  <Info className="h-4 w-4 text-slate-500" />
                </Tooltip>
              </div>
              <p className="text-2xl font-bold text-white mt-1">
                {latestPrediction ? `${latestPrediction.inference_time_ms.toFixed(1)}ms` : '--'}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                {latestPrediction && latestPrediction.inference_time_ms < 20 ? 'Excellent' : 'Good'}
              </p>
            </div>
            <div className="h-12 w-12 rounded-full bg-blue-500/20 text-blue-500 flex items-center justify-center">
              <Clock className="h-6 w-6" />
            </div>
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div className="w-full">
              <div className="flex items-center gap-2 mb-2">
                <p className="text-sm text-slate-400">Market Tendency</p>
                <Tooltip content="Current market state based on order book volume imbalance and spread. Indicates buying/selling pressure.">
                  <Info className="h-4 w-4 text-slate-500" />
                </Tooltip>
              </div>

              {/* Time window selector */}
              <div className="flex gap-1 mb-2">
                {['live', '1m', '1h'].map((window) => (
                  <button
                    key={window}
                    onClick={() => setTendencyWindow(window as any)}
                    className={`px-2 py-1 text-xs rounded transition-colors ${
                      tendencyWindow === window
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    }`}
                  >
                    {window === 'live' ? 'Live' : window}
                  </button>
                ))}
              </div>

              {tendencyStats && (
                <div className="flex items-center gap-3">
                  <div className={`h-10 w-10 rounded-full bg-slate-700/50 flex items-center justify-center ${tendencyStats.tendency.color}`}>
                    <tendencyStats.tendency.icon className="h-5 w-5" />
                  </div>
                  <div>
                    <p className={`text-base font-bold ${tendencyStats.tendency.color}`}>
                      {tendencyStats.tendency.label}
                    </p>
                    <p className="text-xs text-slate-500">
                      Vol Imb: {(tendencyStats.volumeImbalance * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Live LOB Data */}
      <CollapsibleSection
        title="Live Order Book Data"
        defaultOpen={true}
        tooltip="Real-time limit order book data from Binance. Shows best bid/ask prices and volumes at 5 price levels."
      >
        {(() => {
          const { data: lobData } = useQuery({
            queryKey: ['lob', 'latest'],
            queryFn: () => apiClient.getLatestLOB(),
            refetchInterval: 1000, // Update every second
          });

          if (!lobData?.data) {
            return <div className="text-center py-4 text-slate-400">Loading LOB data...</div>;
          }

          const lob = lobData.data;
          const maxVolume = Math.max(
            ...lob.bids.map(b => b.volume),
            ...lob.asks.map(a => a.volume)
          );

          return (
            <div>
              {/* Summary Stats */}
              <div className="grid grid-cols-4 gap-4 mb-6">
                <div className="bg-slate-700/30 p-3 rounded">
                  <p className="text-xs text-slate-400 mb-1">Mid Price</p>
                  <p className="text-lg font-mono font-bold text-white">
                    ${lob.mid_price.toFixed(2)}
                  </p>
                </div>
                <div className="bg-slate-700/30 p-3 rounded">
                  <p className="text-xs text-slate-400 mb-1">Spread</p>
                  <p className="text-lg font-mono font-bold text-white">
                    {lob.spread_bps.toFixed(4)} bps
                  </p>
                </div>
                <div className="bg-green-500/10 p-3 rounded border border-green-500/20">
                  <p className="text-xs text-green-400 mb-1">Total Bids</p>
                  <p className="text-lg font-mono font-bold text-green-400">
                    {lob.total_bid_volume.toFixed(2)}
                  </p>
                </div>
                <div className="bg-red-500/10 p-3 rounded border border-red-500/20">
                  <p className="text-xs text-red-400 mb-1">Total Asks</p>
                  <p className="text-lg font-mono font-bold text-red-400">
                    {lob.total_ask_volume.toFixed(2)}
                  </p>
                </div>
              </div>

              {/* Order Book Visualization */}
              <div className="grid grid-cols-2 gap-6">
                {/* Bids (Buy Orders) */}
                <div>
                  <h3 className="text-sm font-semibold text-green-400 mb-3 flex items-center gap-2">
                    <TrendingUp className="h-4 w-4" />
                    Bids (Buy Orders)
                  </h3>
                  <div className="space-y-2">
                    {lob.bids.map((bid, idx) => {
                      const volumePercent = (bid.volume / maxVolume) * 100;
                      return (
                        <div key={idx} className="relative">
                          {/* Volume bar */}
                          <div
                            className="absolute inset-0 bg-green-500/10 rounded"
                            style={{ width: `${volumePercent}%` }}
                          />
                          {/* Content */}
                          <div className="relative flex justify-between items-center px-3 py-2 text-sm font-mono">
                            <span className="text-green-400 font-semibold">
                              ${bid.price.toFixed(2)}
                            </span>
                            <span className="text-slate-300">
                              {bid.volume.toFixed(4)}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Asks (Sell Orders) */}
                <div>
                  <h3 className="text-sm font-semibold text-red-400 mb-3 flex items-center gap-2">
                    <TrendingDown className="h-4 w-4" />
                    Asks (Sell Orders)
                  </h3>
                  <div className="space-y-2">
                    {lob.asks.map((ask, idx) => {
                      const volumePercent = (ask.volume / maxVolume) * 100;
                      return (
                        <div key={idx} className="relative">
                          {/* Volume bar */}
                          <div
                            className="absolute inset-0 bg-red-500/10 rounded"
                            style={{ width: `${volumePercent}%` }}
                          />
                          {/* Content */}
                          <div className="relative flex justify-between items-center px-3 py-2 text-sm font-mono">
                            <span className="text-red-400 font-semibold">
                              ${ask.price.toFixed(2)}
                            </span>
                            <span className="text-slate-300">
                              {ask.volume.toFixed(4)}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Volume Imbalance Gauge */}
              <div className="mt-6 bg-slate-700/30 p-4 rounded">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-slate-400">Volume Imbalance</span>
                  <span className={`text-sm font-bold ${
                    lob.volume_imbalance > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {(lob.volume_imbalance * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="relative h-4 bg-slate-600 rounded-full overflow-hidden">
                  <div
                    className={`absolute top-0 h-full transition-all ${
                      lob.volume_imbalance > 0 ? 'bg-green-500' : 'bg-red-500'
                    }`}
                    style={{
                      left: lob.volume_imbalance > 0 ? '50%' : `${50 + lob.volume_imbalance * 50}%`,
                      width: `${Math.abs(lob.volume_imbalance) * 50}%`
                    }}
                  />
                  <div className="absolute top-0 left-1/2 w-0.5 h-full bg-slate-400" />
                </div>
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>← More Asks</span>
                  <span>More Bids →</span>
                </div>
              </div>
            </div>
          );
        })()}
      </CollapsibleSection>

      {/* Insights Panel */}
      <CollapsibleSection
        title="Insights & Tips"
        defaultOpen={true}
        tooltip="Automatic insights generated from model performance. Helps you understand which predictions to trust."
      >
        <div className="space-y-3">
          {insights.map((insight, idx) => (
            <div
              key={idx}
              className={`flex items-start gap-3 p-3 rounded-lg ${
                insight.type === 'success'
                  ? 'bg-green-500/10 border border-green-500/20'
                  : insight.type === 'warning'
                  ? 'bg-yellow-500/10 border border-yellow-500/20'
                  : 'bg-blue-500/10 border border-blue-500/20'
              }`}
            >
              {insight.type === 'success' && <CheckCircle className="h-5 w-5 text-green-400 flex-shrink-0 mt-0.5" />}
              {insight.type === 'warning' && <AlertCircle className="h-5 w-5 text-yellow-400 flex-shrink-0 mt-0.5" />}
              {insight.type === 'info' && <Info className="h-5 w-5 text-blue-400 flex-shrink-0 mt-0.5" />}
              <p className="text-sm text-slate-200">{insight.message}</p>
            </div>
          ))}
        </div>
      </CollapsibleSection>

      {/* Cumulative Accuracy Metrics */}
      <CollapsibleSection
        title="Cumulative Accuracy Metrics"
        defaultOpen={true}
        tooltip="Directional Accuracy: % of predictions where direction (up/down) was correct. MAE: Mean Absolute Error in percentage points - lower is better."
      >
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {[
            { label: 'Last Minute', data: metrics1m?.data, color: 'border-red-500/30' },
            { label: 'Last Hour', data: metrics1h?.data, color: 'border-orange-500/30' },
            { label: 'Last 24 Hours', data: metrics24h?.data, color: 'border-blue-500/30' },
            { label: 'All Time', data: metricsAll?.data, color: 'border-green-500/30' },
          ].map((timeWindow) => (
            <div key={timeWindow.label} className={`bg-slate-700/30 rounded-lg p-4 border-2 ${timeWindow.color}`}>
              <h3 className="text-sm font-semibold text-slate-300 mb-3">{timeWindow.label}</h3>
              {timeWindow.data && timeWindow.data.length > 0 ? (
                <div className="space-y-2">
                  {timeWindow.data.map((metric) => (
                    <div key={metric.horizon} className="flex justify-between items-center">
                      <span className="text-xs text-slate-400">{metric.horizon}</span>
                      <div className="text-right">
                        <div className={`text-sm font-bold ${
                          metric.directional_accuracy > 60 ? 'text-green-400' :
                          metric.directional_accuracy > 40 ? 'text-yellow-400' :
                          'text-red-400'
                        }`}>
                          {metric.directional_accuracy.toFixed(1)}%
                        </div>
                        <div className="text-xs text-slate-500">
                          MAE: {metric.mae.toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-xs text-slate-500">
                  No data yet
                </div>
              )}
            </div>
          ))}
        </div>
      </CollapsibleSection>

      {/* Predictions vs Actuals */}
      {outcomes?.data && outcomes.data.length > 0 && (
        <CollapsibleSection
          title="Recent Predictions vs Actuals (10s Horizon)"
          defaultOpen={true}
          tooltip="Comparison of what the model predicted vs what actually happened. Green = correct direction, Red = wrong direction."
        >
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-slate-400 border-b border-slate-700">
                  <th className="pb-3">Time</th>
                  <th className="pb-3">Predicted</th>
                  <th className="pb-3">Actual</th>
                  <th className="pb-3">Error</th>
                  <th className="pb-3">Direction</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                {outcomes.data.map((outcome, idx) => (
                  <tr
                    key={idx}
                    className="border-b border-slate-700/50 hover:bg-slate-700/30"
                  >
                    <td className="py-3 text-slate-300 font-mono text-xs">
                      {formatPreciseTime(outcome.prediction_time)}
                    </td>
                    <td className={`py-3 font-mono ${
                      outcome.predicted_return > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {outcome.predicted_return > 0 ? '+' : ''}
                      {outcome.predicted_return.toFixed(3)}%
                    </td>
                    <td className={`py-3 font-mono ${
                      outcome.actual_return > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {outcome.actual_return > 0 ? '+' : ''}
                      {outcome.actual_return.toFixed(3)}%
                    </td>
                    <td className="py-3 text-slate-300 font-mono">
                      {outcome.absolute_error.toFixed(3)}%
                    </td>
                    <td className="py-3">
                      {outcome.direction_correct ? (
                        <CheckCircle className="h-5 w-5 text-green-400" />
                      ) : (
                        <XCircle className="h-5 w-5 text-red-400" />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CollapsibleSection>
      )}

      {/* Recent Predictions */}
      <CollapsibleSection
        title="Recent Predictions (Live Feed)"
        defaultOpen={false}
        tooltip="Real-time predictions as they are generated. Shows all 5 prediction horizons plus market context."
      >
        {predictionsLoading ? (
          <div className="text-center py-8 text-slate-400">Loading predictions...</div>
        ) : predictions?.data && predictions.data.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-slate-400 border-b border-slate-700">
                  <th className="pb-3">Time (HH:MM:SS.ms)</th>
                  <th className="pb-3">Mid Price</th>
                  <th className="pb-3">1s</th>
                  <th className="pb-3">5s</th>
                  <th className="pb-3">10s</th>
                  <th className="pb-3">30s</th>
                  <th className="pb-3">60s</th>
                  <th className="pb-3">Spread (bps)</th>
                  <th className="pb-3">Vol Imb</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                {predictions.data.slice(0, 20).map((pred, idx) => (
                  <tr
                    key={idx}
                    className="border-b border-slate-700/50 hover:bg-slate-700/30"
                  >
                    <td className="py-3 text-slate-300 font-mono text-xs">
                      {formatPreciseTime(pred.time)}
                    </td>
                    <td className="py-3 text-white font-mono">
                      ${pred.mid_price.toFixed(2)}
                    </td>
                    <td
                      className={`py-3 font-mono ${
                        pred.pred_1s > 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {pred.pred_1s > 0 ? '+' : ''}
                      {pred.pred_1s.toFixed(3)}%
                    </td>
                    <td
                      className={`py-3 font-mono ${
                        pred.pred_5s > 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {pred.pred_5s > 0 ? '+' : ''}
                      {pred.pred_5s.toFixed(3)}%
                    </td>
                    <td
                      className={`py-3 font-mono ${
                        pred.pred_10s > 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {pred.pred_10s > 0 ? '+' : ''}
                      {pred.pred_10s.toFixed(3)}%
                    </td>
                    <td
                      className={`py-3 font-mono ${
                        pred.pred_30s > 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {pred.pred_30s > 0 ? '+' : ''}
                      {pred.pred_30s.toFixed(3)}%
                    </td>
                    <td
                      className={`py-3 font-mono ${
                        pred.pred_60s > 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {pred.pred_60s > 0 ? '+' : ''}
                      {pred.pred_60s.toFixed(3)}%
                    </td>
                    <td className="py-3 text-slate-300 font-mono text-xs">
                      {pred.spread_bps.toFixed(4)}
                    </td>
                    <td
                      className={`py-3 font-mono text-xs ${
                        pred.volume_imbalance > 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {(pred.volume_imbalance * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-slate-400">No predictions available</div>
        )}
      </CollapsibleSection>
    </div>
  );
}
