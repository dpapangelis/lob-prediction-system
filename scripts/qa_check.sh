#!/bin/bash
# Quick QA check script - runs all quality checks

set -e  # Exit on first error

echo "🔍 Running Quality Assurance Checks..."
echo ""

echo "✨ Step 1: Formatting code with Black..."
black src/ tests/ config/
echo "✅ Black formatting complete"
echo ""

echo "📦 Step 2: Sorting imports with isort..."
isort src/ tests/ config/
echo "✅ Import sorting complete"
echo ""

echo "🧪 Step 3: Running tests with coverage..."
pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
echo "✅ Tests passed"
echo ""

echo "🔎 Step 4: Linting with flake8..."
flake8 src/ tests/ config/ --max-line-length=100 --extend-ignore=E203,W503 --statistics
echo "✅ Linting passed"
echo ""

echo "🔐 Step 5: Type checking with mypy..."
mypy src/ --ignore-missing-imports || echo "⚠️  Type hints need attention (non-blocking)"
echo ""

echo "🛡️  Step 6: Security check with bandit..."
bandit -r src/ -f screen -ll || echo "⚠️  Security warnings found (review recommended)"
echo ""

echo "✅ All QA checks complete!"
echo "📊 Coverage report available at: htmlcov/index.html"
