import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { Database, Clock, TrendingUp } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

export default function DataDashboard() {
  const { data: stats, isLoading } = useQuery({
    queryKey: ['data-statistics'],
    queryFn: () => apiClient.getDataStatistics(),
  });

  if (isLoading) {
    return <div className="text-center py-8 text-slate-400">Loading...</div>;
  }

  const statistics = stats?.data;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Total Samples</p>
              <p className="text-2xl font-bold text-white mt-1">
                {statistics?.total_samples.toLocaleString()}
              </p>
            </div>
            <Database className="h-8 w-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Duration</p>
              <p className="text-2xl font-bold text-white mt-1">
                {statistics?.duration_hours.toFixed(1)} hrs
              </p>
            </div>
            <Clock className="h-8 w-8 text-green-500" />
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-400">Sample Rate</p>
              <p className="text-2xl font-bold text-white mt-1">
                {statistics?.samples_per_hour.toFixed(1)}/hr
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-purple-500" />
          </div>
        </div>
      </div>

      {statistics && (
        <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
          <h2 className="text-lg font-semibold text-white mb-4">Dataset Details</h2>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-slate-400">Symbol</p>
              <p className="text-lg text-white">{statistics.symbol}</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Start Time</p>
              <p className="text-lg text-white">
                {formatDistanceToNow(new Date(statistics.time_range_start), {
                  addSuffix: true,
                })}
              </p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Avg Spread (bps)</p>
              <p className="text-lg text-white">{statistics.avg_spread_bps.toFixed(4)}</p>
            </div>
            <div>
              <p className="text-sm text-slate-400">Avg Volume Imbalance</p>
              <p className="text-lg text-white">
                {statistics.avg_volume_imbalance.toFixed(3)}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
