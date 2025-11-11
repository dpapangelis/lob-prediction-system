import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { Activity, TrendingUp, TrendingDown, Clock } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

export default function LiveDashboard() {
  // Fetch latest predictions
  const { data: predictions, isLoading: predictionsLoading } = useQuery({
    queryKey: ['predictions', 'latest'],
    queryFn: () => apiClient.getLatestPredictions(10),
  });

  // Fetch accuracy metrics
  const { data: accuracyMetrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['accuracy', '1m'],
    queryFn: () => apiClient.getAccuracyMetrics('1m'),
  });

  // Fetch collection status
  const { data: collectionStatus } = useQuery({
    queryKey: ['collection-status'],
    queryFn: () => apiClient.getCollectionStatus(),
    refetchInterval: 5000,
  });

  const latestPrediction = predictions?.data[0];

  return (
    <div className="space-y-6">
      {/* Status Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Collection Status</p>
              <p className="text-2xl font-bold text-white mt-1">
                {collectionStatus?.data.is_collecting ? 'Active' : 'Inactive'}
              </p>
              {collectionStatus?.data.last_update && (
                <p className="text-xs text-slate-500 mt-1">
                  {formatDistanceToNow(new Date(collectionStatus.data.last_update), {
                    addSuffix: true,
                  })}
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
              <p className="text-sm text-slate-400">Latest Prediction (10s)</p>
              <p className="text-2xl font-bold text-white mt-1">
                {latestPrediction
                  ? `${latestPrediction.pred_10s > 0 ? '+' : ''}${latestPrediction.pred_10s.toFixed(3)}%`
                  : '--'}
              </p>
              {latestPrediction && (
                <p className="text-xs text-slate-500 mt-1">
                  ${latestPrediction.mid_price.toFixed(2)}
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
              <p className="text-sm text-slate-400">Inference Time</p>
              <p className="text-2xl font-bold text-white mt-1">
                {latestPrediction ? `${latestPrediction.inference_time_ms.toFixed(1)}ms` : '--'}
              </p>
              <p className="text-xs text-slate-500 mt-1">Average latency</p>
            </div>
            <div className="h-12 w-12 rounded-full bg-blue-500/20 text-blue-500 flex items-center justify-center">
              <Clock className="h-6 w-6" />
            </div>
          </div>
        </div>
      </div>

      {/* Accuracy Metrics */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h2 className="text-lg font-semibold text-white mb-4">
          Accuracy Metrics (Last Minute)
        </h2>
        {metricsLoading ? (
          <div className="text-center py-8 text-slate-400">Loading metrics...</div>
        ) : accuracyMetrics?.data && accuracyMetrics.data.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {accuracyMetrics.data.map((metric) => (
              <div
                key={metric.horizon}
                className="bg-slate-700/50 rounded-lg p-4 border border-slate-600"
              >
                <p className="text-sm text-slate-400 mb-1">{metric.horizon}</p>
                <p className="text-2xl font-bold text-white">
                  {metric.directional_accuracy.toFixed(1)}%
                </p>
                <p className="text-xs text-slate-500 mt-1">
                  {metric.num_predictions} predictions
                </p>
                <p className="text-xs text-slate-500">
                  MAE: {metric.mae.toFixed(3)}%
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-slate-400">
            No metrics available for the last minute
          </div>
        )}
      </div>

      {/* Recent Predictions */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h2 className="text-lg font-semibold text-white mb-4">Recent Predictions</h2>
        {predictionsLoading ? (
          <div className="text-center py-8 text-slate-400">Loading predictions...</div>
        ) : predictions?.data && predictions.data.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-left text-sm text-slate-400 border-b border-slate-700">
                  <th className="pb-3">Time</th>
                  <th className="pb-3">Mid Price</th>
                  <th className="pb-3">1s</th>
                  <th className="pb-3">5s</th>
                  <th className="pb-3">10s</th>
                  <th className="pb-3">30s</th>
                  <th className="pb-3">60s</th>
                  <th className="pb-3">Spread</th>
                  <th className="pb-3">Vol Imb</th>
                </tr>
              </thead>
              <tbody className="text-sm">
                {predictions.data.map((pred, idx) => (
                  <tr
                    key={idx}
                    className="border-b border-slate-700/50 hover:bg-slate-700/30"
                  >
                    <td className="py-3 text-slate-300">
                      {formatDistanceToNow(new Date(pred.time), { addSuffix: true })}
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
                    <td className="py-3 text-slate-300">
                      {pred.spread_bps.toFixed(4)}
                    </td>
                    <td
                      className={`py-3 ${
                        pred.volume_imbalance > 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {pred.volume_imbalance.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-slate-400">No predictions available</div>
        )}
      </div>
    </div>
  );
}
