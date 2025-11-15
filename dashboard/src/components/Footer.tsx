import { Github, BookOpen, Mail, ExternalLink } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-slate-800 border-t border-slate-700 mt-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">
              About This Project
            </h3>
            <p className="text-sm text-slate-400 mb-3">
              LOB-based cryptocurrency price prediction system using Temporal Convolutional Networks (TCN).
              Part of a Computer Science dissertation at CITY College, University of York.
            </p>
            <p className="text-xs text-slate-500">
              © 2024 LOB Prediction System. All rights reserved.
            </p>
          </div>

          {/* Documentation & Resources */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">
              Documentation & Resources
            </h3>
            <ul className="space-y-2">
              <li>
                <a
                  href="/docs/technical/01_system_architecture.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-blue-400 flex items-center gap-2 transition-colors"
                >
                  <BookOpen className="h-4 w-4" />
                  System Architecture
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a
                  href="/docs/technical/02_data_collection.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-blue-400 flex items-center gap-2 transition-colors"
                >
                  <BookOpen className="h-4 w-4" />
                  Data Collection
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a
                  href="/docs/technical/05_model_architecture.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-blue-400 flex items-center gap-2 transition-colors"
                >
                  <BookOpen className="h-4 w-4" />
                  Model Architecture (TCN)
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a
                  href="/docs/technical/08_shap_explainability.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-blue-400 flex items-center gap-2 transition-colors"
                >
                  <BookOpen className="h-4 w-4" />
                  SHAP Explainability
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
            </ul>
          </div>

          {/* Contact & Links */}
          <div>
            <h3 className="text-sm font-semibold text-white mb-3">
              Contact & Links
            </h3>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://github.com/yourusername/lob-prediction-system"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-blue-400 flex items-center gap-2 transition-colors"
                >
                  <Github className="h-4 w-4" />
                  View on GitHub
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
              <li>
                <a
                  href="mailto:your.email@york.ac.uk"
                  className="text-sm text-slate-400 hover:text-blue-400 flex items-center gap-2 transition-colors"
                >
                  <Mail className="h-4 w-4" />
                  Contact Developer
                </a>
              </li>
              <li>
                <a
                  href="http://localhost:8000/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-400 hover:text-blue-400 flex items-center gap-2 transition-colors"
                >
                  <BookOpen className="h-4 w-4" />
                  API Documentation
                  <ExternalLink className="h-3 w-3" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-8 pt-6 border-t border-slate-700">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-xs text-slate-500">
              Built with React, TypeScript, FastAPI, TimescaleDB, and PyTorch
            </p>
            <div className="flex gap-6 text-xs text-slate-500">
              <span>API Status: <span className="text-green-400">●</span> Online</span>
              <span>Database: <span className="text-green-400">●</span> Connected</span>
              <span>Model: <span className="text-green-400">●</span> Active</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
