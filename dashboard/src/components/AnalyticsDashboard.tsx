import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function AnalyticsDashboard() {
  const { data: featureImportance, isLoading } = useQuery({
    queryKey: ['shap', 'feature-importance', '10s'],
    queryFn: () => apiClient.getFeatureImportance('10s', 15),
  });

  const chartData = featureImportance?.data.map(f => ({
    name: f.feature_name,
    importance: f.importance,
  })) || [];

  return (
    <div className="space-y-6">
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <h2 className="text-lg font-semibold text-white mb-4">
          Feature Importance (10s Horizon)
        </h2>
        {isLoading ? (
          <div className="text-center py-8 text-slate-400">Loading...</div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis type="number" stroke="#94a3b8" />
              <YAxis type="category" dataKey="name" width={150} stroke="#94a3b8" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Bar dataKey="importance" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center py-8 text-slate-400">No SHAP data available</div>
        )}
      </div>
    </div>
  );
}
