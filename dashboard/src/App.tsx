import { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Activity, BarChart3, Brain, Database, TrendingUp } from 'lucide-react';
import LiveDashboard from './components/LiveDashboard';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import DataDashboard from './components/DataDashboard';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 5000, // Refetch every 5 seconds
      staleTime: 3000,
    },
  },
});

type Tab = 'live' | 'analytics' | 'data' | 'training' | 'models';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('live');

  const tabs = [
    { id: 'live' as Tab, name: 'Live Inference', icon: Activity },
    { id: 'analytics' as Tab, name: 'Analytics', icon: BarChart3 },
    { id: 'data' as Tab, name: 'Data', icon: Database },
    { id: 'training' as Tab, name: 'Training', icon: TrendingUp },
    { id: 'models' as Tab, name: 'Models', icon: Brain },
  ];

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-slate-900">
        {/* Header */}
        <header className="bg-slate-800 border-b border-slate-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div className="flex items-center">
                <TrendingUp className="h-8 w-8 text-blue-500 mr-3" />
                <div>
                  <h1 className="text-2xl font-bold text-white">
                    LOB Prediction System
                  </h1>
                  <p className="text-sm text-slate-400">
                    Real-time cryptocurrency price prediction
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="flex items-center">
                  <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse mr-2"></div>
                  <span className="text-sm text-slate-300">Live</span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Navigation Tabs */}
        <nav className="bg-slate-800 border-b border-slate-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm
                      transition-colors
                      ${
                        activeTab === tab.id
                          ? 'border-blue-500 text-blue-500'
                          : 'border-transparent text-slate-400 hover:text-slate-300 hover:border-slate-300'
                      }
                    `}
                  >
                    <Icon className="h-5 w-5" />
                    <span>{tab.name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {activeTab === 'live' && <LiveDashboard />}
          {activeTab === 'analytics' && <AnalyticsDashboard />}
          {activeTab === 'data' && <DataDashboard />}
          {activeTab === 'training' && (
            <div className="text-center py-12">
              <TrendingUp className="h-16 w-16 text-slate-600 mx-auto mb-4" />
              <h2 className="text-xl font-semibold text-slate-400">
                Training Dashboard Coming Soon
              </h2>
            </div>
          )}
          {activeTab === 'models' && (
            <div className="text-center py-12">
              <Brain className="h-16 w-16 text-slate-600 mx-auto mb-4" />
              <h2 className="text-xl font-semibold text-slate-400">
                Model Registry Coming Soon
              </h2>
            </div>
          )}
        </main>
      </div>
    </QueryClientProvider>
  );
}

export default App;
