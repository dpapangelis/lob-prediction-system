import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../api/client';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';
import { Info, TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { useState } from 'react';

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
        <div className="absolute z-10 w-80 p-4 text-sm bg-slate-700 text-slate-200 rounded-lg shadow-lg -top-2 left-8 border border-slate-600">
          {content}
          <div className="absolute w-2 h-2 bg-slate-700 border-l border-b border-slate-600 transform rotate-45 -left-1 top-4"></div>
        </div>
      )}
    </div>
  );
};

// Feature explanations
const FEATURE_EXPLANATIONS: Record<string, {
  name: string;
  description: string;
  impact: string;
  icon: any;
}> = {
  'total_ask_volume': {
    name: 'Total Ask Volume',
    description: 'Sum of all sell orders in the order book across all price levels.',
    impact: 'High ask volume indicates strong selling pressure. When this feature has high importance, it means sell-side liquidity is a key predictor of price movements.',
    icon: TrendingDown,
  },
  'total_bid_volume': {
    name: 'Total Bid Volume',
    description: 'Sum of all buy orders in the order book across all price levels.',
    impact: 'High bid volume indicates strong buying pressure. Important for predicting upward price movements.',
    icon: TrendingUp,
  },
  'volume_imbalance': {
    name: 'Volume Imbalance',
    description: 'Ratio of (bid volume - ask volume) / (bid volume + ask volume). Ranges from -1 (all asks) to +1 (all bids).',
    impact: 'Strong predictor of price direction. Positive imbalance (more bids) suggests upward pressure, negative (more asks) suggests downward pressure.',
    icon: Activity,
  },
  'spread_bps': {
    name: 'Spread (Basis Points)',
    description: 'Difference between best ask and best bid as a percentage of mid price, in basis points (1 bps = 0.01%).',
    impact: 'Tight spread = high liquidity = more predictable prices. Wide spread = low liquidity = higher uncertainty in predictions.',
    icon: Activity,
  },
  'bid_volume_1': {
    name: 'Best Bid Volume',
    description: 'Volume available at the best (highest) bid price.',
    impact: 'Large volume at best bid provides immediate support against price drops.',
    icon: TrendingUp,
  },
  'ask_volume_1': {
    name: 'Best Ask Volume',
    description: 'Volume available at the best (lowest) ask price.',
    impact: 'Large volume at best ask provides immediate resistance against price rises.',
    icon: TrendingDown,
  },
  'mid_price': {
    name: 'Mid Price',
    description: 'Average of best bid and best ask: (bid + ask) / 2',
    impact: 'Should have LOW importance - model should predict % returns, not absolute price levels.',
    icon: Activity,
  },
  'spread': {
    name: 'Spread (Absolute)',
    description: 'Absolute dollar difference between best ask and best bid.',
    impact: 'Similar to spread_bps but in absolute terms. Indicates market liquidity.',
    icon: Activity,
  },
  'weighted_mid_price': {
    name: 'Volume-Weighted Mid Price',
    description: 'Mid price weighted by volumes at each level, giving more weight to levels with higher volume.',
    impact: 'More accurate representation of "true" market price than simple mid price.',
    icon: Activity,
  },
  'price_range': {
    name: 'Price Range',
    description: 'Difference between 5th level ask and 5th level bid (depth of order book).',
    impact: 'Wider range = more price levels = better depth. Important for large orders.',
    icon: Activity,
  },
  'depth_imbalance': {
    name: 'Depth Imbalance',
    description: 'Weighted difference between bid and ask sides across all levels, giving more weight to closer levels.',
    impact: 'Captures the overall balance of buy/sell pressure across the entire visible order book.',
    icon: Activity,
  },
};

// Add generic explanation for features not in the map
const getFeatureExplanation = (featureName: string) => {
  if (FEATURE_EXPLANATIONS[featureName]) {
    return FEATURE_EXPLANATIONS[featureName];
  }

  // Handle price_level_X
  if (featureName.startsWith('price_level_')) {
    const level = featureName.split('_')[2];
    return {
      name: `Price Level ${level}`,
      description: `Distance of level ${level} bid price from mid price, as a percentage.`,
      impact: `Shows how far level ${level} is from current price. Important for understanding order book structure.`,
      icon: Activity,
    };
  }

  // Handle volume_ratio_X
  if (featureName.startsWith('volume_ratio_')) {
    const level = featureName.split('_')[2];
    return {
      name: `Volume Ratio ${level}`,
      description: `Proportion of total volume at level ${level}.`,
      impact: `Indicates concentration of liquidity. High ratio means liquidity is concentrated at this level.`,
      icon: Activity,
    };
  }

  // Handle bid_price_X / ask_price_X / bid_volume_X / ask_volume_X
  if (featureName.match(/^(bid|ask)_(price|volume)_\d+$/)) {
    const [side, type, level] = featureName.split('_');
    return {
      name: `${side.charAt(0).toUpperCase() + side.slice(1)} ${type.charAt(0).toUpperCase() + type.slice(1)} Level ${level}`,
      description: `${type === 'price' ? 'Price' : 'Volume'} at level ${level} on the ${side} side.`,
      impact: `Level ${level} ${type} provides ${side === 'bid' ? 'support' : 'resistance'}. Deeper levels matter for larger orders.`,
      icon: side === 'bid' ? TrendingUp : TrendingDown,
    };
  }

  return {
    name: featureName,
    description: 'Custom LOB feature',
    impact: 'Contributes to prediction accuracy',
    icon: Activity,
  };
};

export default function AnalyticsDashboard() {
  const [selectedHorizon, setSelectedHorizon] = useState('10s');
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);

  const { data: featureImportance, isLoading } = useQuery({
    queryKey: ['shap', 'feature-importance', selectedHorizon],
    queryFn: () => apiClient.getFeatureImportance(selectedHorizon, 15),
  });

  const chartData = featureImportance?.data.map(f => ({
    name: f.feature_name,
    importance: f.importance,
    direction: f.direction,
  })) || [];

  const selectedFeatureData = selectedFeature
    ? featureImportance?.data.find(f => f.feature_name === selectedFeature)
    : null;

  const explanation = selectedFeature ? getFeatureExplanation(selectedFeature) : null;
  const Icon = explanation?.icon || Activity;

  return (
    <div className="space-y-6">
      {/* Header with explanation */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <div className="flex items-start gap-3">
          <Info className="h-6 w-6 text-blue-400 flex-shrink-0 mt-1" />
          <div>
            <h2 className="text-lg font-semibold text-white mb-2">
              Understanding Feature Importance (SHAP Values)
            </h2>
            <p className="text-sm text-slate-300 mb-3">
              This dashboard shows which Order Book features the model uses to make predictions.
              SHAP (SHapley Additive exPlanations) values measure each feature's contribution to predictions.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-slate-700/50 p-3 rounded">
                <p className="text-slate-400 mb-1">📊 <strong>Importance Score</strong></p>
                <p className="text-slate-300">
                  Higher score = feature matters more for predictions.
                  Measures average absolute impact across all predictions.
                </p>
              </div>
              <div className="bg-slate-700/50 p-3 rounded">
                <p className="text-slate-400 mb-1">➡️ <strong>Direction</strong></p>
                <p className="text-slate-300">
                  Positive direction = feature increases predicted return.
                  Negative direction = feature decreases predicted return.
                </p>
              </div>
              <div className="bg-slate-700/50 p-3 rounded">
                <p className="text-slate-400 mb-1">🎯 <strong>How to Use</strong></p>
                <p className="text-slate-300">
                  Click on any bar to see detailed explanation of that feature
                  and how it affects price predictions.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Horizon selector */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <div className="flex items-center gap-3 mb-4">
          <h3 className="text-sm font-semibold text-white">Select Prediction Horizon:</h3>
          <div className="flex gap-2">
            {['1s', '5s', '10s', '30s', '60s'].map((horizon) => (
              <button
                key={horizon}
                onClick={() => {
                  setSelectedHorizon(horizon);
                  setSelectedFeature(null);
                }}
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

      {/* Feature importance chart */}
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <h2 className="text-lg font-semibold text-white">
            Top 15 Most Important Features ({selectedHorizon} Horizon)
          </h2>
          <Tooltip content="Click on any bar to see a detailed explanation of that feature and how it influences predictions.">
            <Info className="h-4 w-4 text-slate-500" />
          </Tooltip>
        </div>
        {isLoading ? (
          <div className="text-center py-8 text-slate-400">Loading...</div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={500}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 150 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis type="number" stroke="#94a3b8" />
              <YAxis
                type="category"
                dataKey="name"
                width={140}
                stroke="#94a3b8"
                tick={{ fontSize: 12 }}
              />
              <RechartsTooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569' }}
                labelStyle={{ color: '#e2e8f0' }}
                formatter={(value: any, name: string) => {
                  if (name === 'importance') {
                    return [value.toFixed(4), 'Importance'];
                  }
                  return [value, name];
                }}
              />
              <Bar
                dataKey="importance"
                onClick={(data) => setSelectedFeature(data.name)}
                cursor="pointer"
              >
                {chartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.name === selectedFeature ? '#10b981' : '#3b82f6'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center py-8 text-slate-400">No SHAP data available</div>
        )}
      </div>

      {/* Feature detail panel */}
      {selectedFeature && explanation && selectedFeatureData && (
        <div className="bg-slate-800 rounded-lg p-6 border border-2 border-green-500/30">
          <div className="flex items-start gap-4">
            <div className="h-12 w-12 rounded-full bg-green-500/20 flex items-center justify-center flex-shrink-0">
              <Icon className="h-6 w-6 text-green-400" />
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-white mb-2">{explanation.name}</h3>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div className="bg-slate-700/50 p-3 rounded">
                  <p className="text-xs text-slate-400 mb-1">Importance Score</p>
                  <p className="text-2xl font-bold text-green-400">
                    {selectedFeatureData.importance.toFixed(4)}
                  </p>
                </div>
                <div className="bg-slate-700/50 p-3 rounded">
                  <p className="text-xs text-slate-400 mb-1">Average Direction</p>
                  <p className={`text-2xl font-bold ${
                    selectedFeatureData.direction > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {selectedFeatureData.direction > 0 ? '+' : ''}
                    {selectedFeatureData.direction.toFixed(4)}
                  </p>
                </div>
                <div className="bg-slate-700/50 p-3 rounded">
                  <p className="text-xs text-slate-400 mb-1">Sample Count</p>
                  <p className="text-2xl font-bold text-blue-400">
                    {selectedFeatureData.sample_count}
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <div>
                  <p className="text-sm font-semibold text-slate-300 mb-1">📖 What is this feature?</p>
                  <p className="text-sm text-slate-400">{explanation.description}</p>
                </div>

                <div>
                  <p className="text-sm font-semibold text-slate-300 mb-1">💡 How does it affect predictions?</p>
                  <p className="text-sm text-slate-400">{explanation.impact}</p>
                </div>

                <div>
                  <p className="text-sm font-semibold text-slate-300 mb-1">
                    🎯 Interpretation for {selectedHorizon} predictions:
                  </p>
                  <p className="text-sm text-slate-400">
                    {selectedFeatureData.direction > 0
                      ? `Higher values of ${explanation.name} lead to higher predicted returns (upward price movement).`
                      : `Higher values of ${explanation.name} lead to lower predicted returns (downward price movement).`
                    }
                    {' '}
                    This feature has an average absolute impact of {(selectedFeatureData.importance * 100).toFixed(2)}%
                    on predictions.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
