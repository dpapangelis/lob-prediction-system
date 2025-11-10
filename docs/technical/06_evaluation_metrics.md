# Evaluation Metrics: Performance Measurement & Interpretation

## Executive Summary

This document provides a comprehensive guide to evaluating LOB-based price prediction models, including metric definitions, interpretation guidelines, and statistical significance testing. Understanding these metrics is critical for assessing model performance, comparing architectures, and determining deployment readiness.

**Key Metrics Covered:**
- Regression Metrics (MSE, RMSE, MAE, R²)
- Classification Metrics (Directional Accuracy, Confusion Matrix)
- Financial Metrics (Sharpe Ratio, Information Ratio)
- Statistical Significance Tests

**Purpose:** Enable rigorous, reproducible evaluation that aligns with academic standards and industry best practices.

---

## 1. Evaluation Philosophy

### 1.1 Why Multiple Metrics?

**Single metrics are insufficient for financial prediction:**

| Metric | What It Measures | What It Misses |
|--------|-----------------|----------------|
| MSE only | Average squared error | Direction correctness |
| Accuracy only | Direction correctness | Magnitude errors |
| R² only | Variance explained | Practical profitability |

**Our Approach:** Multi-faceted evaluation
- **Regression metrics:** How close are predictions?
- **Classification metrics:** Are directions correct?
- **Financial metrics:** Would this be profitable?
- **Statistical tests:** Is performance significant?

### 1.2 Evaluation Principles

**1. Separate Test Set (Never Seen During Training)**
```python
# CORRECT: Chronological split
train: Jan 1 - Jul 31 (70%)
val:   Aug 1 - Sep 15 (15%)
test:  Sep 16 - Oct 31 (15%)  ← Evaluate ONLY on this

# Model has NEVER seen test data during:
# - Training
# - Validation
# - Hyperparameter tuning
```

**Why Critical:**
- Prevents overfitting evaluation
- Simulates real deployment
- Ensures unbiased performance estimate

**2. Consistent Baselines**

Always compare against:
- **Random baseline:** 50% directional accuracy
- **Naive baseline:** Yesterday's return predicts today
- **Simple baseline:** Linear regression on basic features

**3. Statistical Significance**

Performance must be:
- **Statistically significant:** p < 0.05 (not due to chance)
- **Economically significant:** Improvements large enough to matter
- **Robust:** Consistent across different time periods

---

## 2. Regression Metrics

### 2.1 Mean Squared Error (MSE)

**Definition:**
```python
MSE = (1/N) * sum((y_pred - y_true)^2 for all samples)

Where:
- N = number of test samples
- y_pred = predicted return (%)
- y_true = actual return (%)
```

**Mathematical Properties:**

1. **Non-negative:** MSE ≥ 0 (perfect predictions → MSE = 0)
2. **Quadratic penalty:** Large errors penalized heavily
   - Error = 2% → Contribution = 4
   - Error = 4% → Contribution = 16 (4x worse, not 2x)
3. **Units:** Squared percentage points

**Interpretation:**

| MSE Value | Interpretation | Quality |
|-----------|---------------|---------|
| 0.0001 | Excellent (0.01% RMSE) | ⭐⭐⭐⭐⭐ |
| 0.01 | Very Good (0.1% RMSE) | ⭐⭐⭐⭐ |
| 0.05 | Good (0.22% RMSE) | ⭐⭐⭐ |
| 0.10 | Moderate (0.32% RMSE) | ⭐⭐ |
| 0.50 | Poor (0.71% RMSE) | ⭐ |
| >1.0 | Very Poor (>1% RMSE) | ❌ |

**Context: BTC Price Prediction**
```python
# BTC at $100,000
MSE = 0.01 (RMSE = 0.1%)
Average prediction error: $100 per BTC

# Is this good?
# - Typical BTC minute return: ±0.05%
# - Our error: 0.1% (2x typical move)
# - Conclusion: Moderate performance
```

**Advantages:**
- Mathematically convenient (differentiable)
- Heavily penalizes outliers (desirable in finance)
- Standard in ML literature

**Disadvantages:**
- Sensitive to outliers (can be too harsh)
- Units (squared %) hard to interpret
- Doesn't indicate direction errors

**When to Use:**
- Model optimization (loss function)
- Comparing model architectures
- Academic benchmarking

### 2.2 Root Mean Squared Error (RMSE)

**Definition:**
```python
RMSE = sqrt(MSE)
```

**Why Use RMSE over MSE?**

**Same units as target:**
```python
Target: Returns in %
MSE: 0.01 (%²) ← Hard to interpret
RMSE: 0.1 (%) ← "Average error is 0.1%"
```

**Interpretation:**
```python
RMSE = 0.1% means:
"On average, predictions are off by ±0.1%"

# For $100,000 BTC:
Price error ≈ $100 on average
```

**Comparison to Actual Volatility:**
```python
# BTC minute returns
Actual std = 0.08%
RMSE = 0.10%

# RMSE > actual volatility → Poor model
# RMSE < actual volatility → Good model
```

**Advantages:**
- Interpretable (same units as target)
- Still penalizes large errors
- Easy to communicate to non-technical stakeholders

**Disadvantages:**
- Same sensitivity to outliers as MSE
- Doesn't distinguish direction errors

### 2.3 Mean Absolute Error (MAE)

**Definition:**
```python
MAE = (1/N) * sum(|y_pred - y_true| for all samples)
```

**Difference from RMSE:**

| Metric | Error Penalty | Outlier Sensitivity |
|--------|--------------|---------------------|
| RMSE | Quadratic (squared) | High |
| MAE | Linear (absolute) | Low |

**Example:**
```python
# Two predictions:
Errors: [0.1%, 5.0%]

# RMSE calculation:
MSE = (0.1² + 5.0²) / 2 = (0.01 + 25) / 2 = 12.505
RMSE = √12.505 = 3.54%

# MAE calculation:
MAE = (0.1 + 5.0) / 2 = 2.55%

# RMSE > MAE because large error (5%) dominates
```

**When RMSE >> MAE:**

Indicates presence of large outlier errors
```python
RMSE = 0.50%, MAE = 0.10%
→ Most predictions accurate (MAE low)
→ But some very bad predictions (RMSE high)
→ Model unstable
```

**When RMSE ≈ MAE:**

Errors uniformly distributed
```python
RMSE = 0.12%, MAE = 0.10%
→ Consistent performance
→ No extreme outliers
→ Model stable
```

**Advantages:**
- Robust to outliers
- Easy to interpret
- More realistic for financial applications

**Disadvantages:**
- Doesn't heavily penalize large errors (may be undesirable)
- Less common in ML literature

**Recommended Use:**
- Primary metric for financial applications
- Report alongside RMSE for completeness

### 2.4 R² (Coefficient of Determination)

**Definition:**
```python
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = sum((y_true - y_pred)²)  # Residual sum of squares
SS_tot = sum((y_true - mean(y_true))²)  # Total sum of squares
```

**Intuitive Interpretation:**

"Proportion of variance in target explained by model"
```python
R² = 0.7 means:
"Model explains 70% of price movement variance"
"30% remains unexplained (noise, other factors)"
```

**Range and Meaning:**

| R² Value | Interpretation | Model Quality |
|----------|---------------|---------------|
| 1.0 | Perfect predictions | Impossible (overfitting) |
| 0.7-0.9 | Excellent | ⭐⭐⭐⭐⭐ |
| 0.5-0.7 | Very Good | ⭐⭐⭐⭐ |
| 0.3-0.5 | Good | ⭐⭐⭐ |
| 0.1-0.3 | Moderate | ⭐⭐ |
| 0-0.1 | Weak | ⭐ |
| <0 | Worse than mean | ❌ |

**Negative R²:**
```python
R² = -0.2 means:
"Model performs WORSE than just predicting the mean"
"Better to ignore model and use average"
```

**Context for Financial Time Series:**
```python
# Typical R² for LOB prediction:
1s horizon: R² = 0.3-0.5 (Good)
5s horizon: R² = 0.2-0.4 (Moderate)
60s horizon: R² = 0.05-0.15 (Weak but acceptable)

# Why lower than other ML tasks?
# - Financial markets highly noisy
# - Random walk component dominates
# - R² = 0.3 is actually impressive!
```

**Comparison to Literature:**

| Paper | Task | R² |
|-------|------|-----|
| Zhang et al. (2019) | LOB 10-step | Not reported (used accuracy) |
| Ntakaris et al. (2018) | Mid-price | 0.15-0.35 |
| **Our Target** | Multi-horizon | 0.25-0.40 (short horizons) |

**Advantages:**
- Bounded [0, 1] for good models
- Interpretable (% variance explained)
- Standard in statistics

**Disadvantages:**
- Can be negative (confusing)
- Sensitive to outliers
- Doesn't indicate direction correctness

**When to Use:**
- Academic reporting
- Model comparison
- Understanding predictive power

---

## 3. Classification Metrics (Directional Accuracy)

### 3.1 Directional Accuracy

**Definition:**
```python
Directional_Accuracy = (1/N) * sum(sign(y_pred) == sign(y_true))

Where:
sign(x) = +1 if x > 0 (price up)
          -1 if x < 0 (price down)
           0 if x = 0 (no change)
```

**Why It Matters:**
```python
# Scenario 1: Low MSE, Low Accuracy
Actual: [+0.5%, +0.3%, +0.1%]
Predicted: [-0.1%, -0.2%, -0.05%]
MSE: 0.04 (very low)
Direction: 0% correct! ← Would lose money trading

# Scenario 2: High MSE, High Accuracy
Actual: [+0.5%, +0.3%, +0.1%]
Predicted: [+2.0%, +1.5%, +0.8%]
MSE: 1.5 (high)
Direction: 100% correct ← Would make money!
```

**Financial Relevance:**

In trading, **direction matters more than magnitude**:
- Correct direction + wrong magnitude = Profit (smaller)
- Wrong direction + correct magnitude = Loss

**Interpretation:**

| Accuracy | Interpretation | Trading Viability |
|----------|---------------|-------------------|
| 50% | Random guessing | ❌ No edge |
| 52-55% | Weak signal | ⚠️ Marginal (high costs) |
| 55-60% | Moderate signal | ✅ Viable (low costs) |
| 60-70% | Strong signal | ✅✅ Very profitable |
| >70% | Exceptional | ⭐⭐⭐ Extraordinary |

**Statistical Significance:**
```python
# Is 55% accuracy significant?
N = 10,000 predictions
Accuracy = 55% = 5,500 correct

# Binomial test:
p-value = binom_test(5500, 10000, 0.5)
# p < 0.001 → Highly significant!

# Rule of thumb:
# Need accuracy > 50% + (1.96 / sqrt(N))
# N = 10,000: Need > 50.98% for p < 0.05
# N = 1,000: Need > 53.1%
```

**By Horizon:**
```python
Expected directional accuracy (our target):

Horizon | Target | Reason
--------|--------|--------
1s      | 68-72% | Strong microstructure signal
5s      | 64-68% | Moderate signal
10s     | 60-65% | Signal decaying
30s     | 56-60% | Weak signal
60s     | 52-56% | Very weak signal

# Why decreasing?
# - More time = more new information arrives
# - LOB features become stale
# - Random walk dominates
```

### 3.2 Confusion Matrix Analysis

**2x2 Matrix:**
```
                    Predicted
                 Up      Down
Actual  Up      TP       FN      (True Positives, False Negatives)
        Down    FP       TN      (False Positives, True Negatives)

Where:
TP = Correctly predicted up
TN = Correctly predicted down
FP = Predicted up, actually down (costly error!)
FN = Predicted down, actually up (missed opportunity)
```

**Example:**
```python
# 1000 predictions
TP = 350  # Predicted up, was up ✓
TN = 300  # Predicted down, was down ✓
FP = 200  # Predicted up, was down ✗ (BAD)
FN = 150  # Predicted down, was up ✗ (MISSED)

# Metrics:
Accuracy = (TP + TN) / Total = 650 / 1000 = 65%
Precision = TP / (TP + FP) = 350 / 550 = 63.6%
Recall = TP / (TP + FN) = 350 / 500 = 70%
```

**Interpretation:**

**Precision:** "When model predicts up, how often is it correct?"
```python
Precision = 63.6%
"If we trade on 'up' signals, 63.6% will be profitable"
```

**Recall:** "Of all actual up moves, how many did model catch?"
```python
Recall = 70%
"Model catches 70% of profitable opportunities"
"Misses 30% (conservative)"
```

**Trading Implications:**

| Strategy | Optimize For | Why |
|----------|-------------|-----|
| Aggressive | Recall | Catch all opportunities |
| Conservative | Precision | Avoid false signals |
| Balanced | F1-Score | Harmonic mean |

**F1-Score:**
```python
F1 = 2 * (Precision * Recall) / (Precision + Recall)
   = 2 * (0.636 * 0.70) / (0.636 + 0.70)
   = 0.667

# Balances precision and recall
# Useful for imbalanced datasets
```

### 3.3 Asymmetric Costs

**Not All Errors Are Equal:**
```python
# FP (False Positive): Predict up, market goes down
# - Enter long position
# - Market moves against us
# - Loss: Trade size × price drop

# FN (False Negative): Predict down, market goes up
# - Don't trade
# - Opportunity cost: Missed profit
# - Loss: $0 (no position)

# FP typically more costly than FN
```

**Cost-Weighted Accuracy:**
```python
# Standard accuracy:
Accuracy = (TP + TN) / N

# Cost-weighted:
Cost = (FP × cost_FP) + (FN × cost_FN)

# Example:
cost_FP = 1.0  # Losing trade
cost_FN = 0.3  # Missed opportunity
Total_Cost = (200 × 1.0) + (150 × 0.3) = 245
```

---

## 4. Financial Performance Metrics

### 4.1 Sharpe Ratio

**Definition:**
```python
Sharpe_Ratio = (Mean_Return - Risk_Free_Rate) / Std_Return

Where:
Mean_Return = Average return from trading strategy
Risk_Free_Rate ≈ 0 (for crypto, no treasury equivalent)
Std_Return = Standard deviation of returns
```

**Interpretation:**

"Risk-adjusted return"
- Sharpe = 1.0: For every 1% risk, earn 1% return
- Sharpe = 2.0: For every 1% risk, earn 2% return (excellent)

**Typical Values:**

| Sharpe Ratio | Interpretation | Investment Quality |
|--------------|---------------|-------------------|
| <0 | Losing money | ❌ Avoid |
| 0-0.5 | Barely profitable | ⚠️ Weak |
| 0.5-1.0 | Decent | ✅ Acceptable |
| 1.0-2.0 | Very Good | ✅✅ Strong |
| >2.0 | Exceptional | ⭐⭐⭐ Rare |

**Context:**
```python
# Traditional markets:
S&P 500 (long-term): Sharpe ≈ 0.4-0.5
Good hedge fund: Sharpe ≈ 1.0-1.5

# Crypto trading:
Buy-and-hold BTC: Sharpe ≈ 0.5-1.0 (volatile)
Our target: Sharpe ≈ 0.8-1.5
```

**Limitations:**

1. **Assumes normal distribution** (returns are fat-tailed)
2. **Doesn't distinguish upside/downside volatility**
3. **Sensitive to outliers**

### 4.2 Information Ratio

**Definition:**
```python
Information_Ratio = (Model_Return - Benchmark_Return) / Tracking_Error

Where:
Tracking_Error = Std(Model_Return - Benchmark_Return)
```

**Why Use This?**

"How much excess return per unit of excess risk?"
```python
# Example:
Benchmark (Buy & Hold): 10% annual return, 20% volatility
Model Strategy: 15% annual return, 22% volatility

Excess Return = 15% - 10% = 5%
Tracking Error = std(daily_model_returns - daily_benchmark_returns)
Information Ratio = 5% / Tracking_Error

# IR = 0.5: Moderate skill
# IR = 1.0: High skill
# IR = 2.0: Exceptional skill (top 5% of managers)
```

**Interpretation:**

| Information Ratio | Interpretation |
|------------------|----------------|
| <0 | Underperforming benchmark |
| 0-0.5 | Marginal improvement |
| 0.5-1.0 | Good active management |
| >1.0 | Exceptional skill |

### 4.3 Maximum Drawdown

**Definition:**
```python
Drawdown_t = (Equity_t - Peak_Equity) / Peak_Equity

Max_Drawdown = min(Drawdown_t for all t)
```

**Interpretation:**

"Largest peak-to-trough decline"
```python
# Portfolio value over time:
$10,000 → $15,000 (peak) → $9,000 (trough) → $12,000

Max_Drawdown = (9,000 - 15,000) / 15,000 = -40%

# Interpretation:
# "At worst point, down 40% from peak"
# Psychological tolerance: Can you stomach -40%?
```

**Risk Management:**

| Max Drawdown | Psychological Impact | Risk Level |
|--------------|---------------------|------------|
| <10% | Comfortable | Low |
| 10-20% | Moderate stress | Medium |
| 20-30% | High stress | High |
| >30% | Severe stress | Very High |

**Target for Trading Strategy:**
```python
# Typical targets:
Conservative: Max DD < 15%
Moderate: Max DD < 25%
Aggressive: Max DD < 40%

# Our target: < 20% (moderate risk)
```

---

## 5. Evaluation by Prediction Horizon

### 5.1 Short Horizon (1-5 seconds)

**Expected Performance:**
```python
Directional Accuracy: 65-72%
RMSE: 0.08-0.15%
R²: 0.30-0.50
```

**Why Better Performance?**

1. **Strong microstructure signal:**
   - Volume imbalance persists 1-5s
   - Order book depth relevant
   - LOB features still informative

2. **Less new information:**
   - Minimal news in 5 seconds
   - No macro events
   - Primarily technical factors

**Evaluation Focus:**

- Directional accuracy (most important)
- Precision (avoid false positives)
- Latency (must be <100ms for actionability)

### 5.2 Medium Horizon (10-30 seconds)

**Expected Performance:**
```python
Directional Accuracy: 58-65%
RMSE: 0.15-0.25%
R²: 0.15-0.35
```

**Why Degrading?**

1. **Signal decay:**
   - LOB features less predictive
   - Momentum becomes more important
   - Technical indicators gain relevance

2. **More noise:**
   - Random traders enter
   - Small news items can appear
   - Greater uncertainty

**Evaluation Focus:**

- R² (variance explained still meaningful)
- MAE (magnitude errors more important)
- Sharpe ratio (risk-adjusted returns)

### 5.3 Long Horizon (60 seconds+)

**Expected Performance:**
```python
Directional Accuracy: 52-58%
RMSE: 0.25-0.40%
R²: 0.05-0.20
```

**Why Difficult?**

1. **Random walk dominance:**
   - Efficient market hypothesis applies
   - LOB features nearly irrelevant
   - Mostly unpredictable

2. **External factors:**
   - News can break
   - Macro sentiment shifts
   - Our model can't see these

**Evaluation Focus:**

- Statistical significance (even 52% is good if significant)
- Consistency (same accuracy across weeks?)
- Robustness (performance in different market regimes)

---

## 6. Statistical Significance Testing

### 6.1 Binomial Test (Directional Accuracy)

**Null Hypothesis:** Model is no better than random (50% accuracy)

**Test:**
```python
from scipy.stats import binom_test

# Our results:
n_correct = 6500
n_total = 10000
p_value = binom_test(n_correct, n_total, 0.5, alternative='greater')

# p_value = 1.2e-89 ← Extremely significant!
# Conclusion: Model is NOT random guessing
```

**Interpretation:**

| p-value | Interpretation |
|---------|---------------|
| <0.001 | Extremely significant (***) |
| <0.01 | Very significant (**) |
| <0.05 | Significant (*) |
| >0.05 | Not significant |

### 6.2 Diebold-Mariano Test (Forecasting)

**Compares two forecasting models:**
```python
from scipy.stats import ttest_rel

# Model A errors vs Model B errors
errors_A = predictions_A - actual
errors_B = predictions_B - actual

# Squared errors
se_A = errors_A ** 2
se_B = errors_B ** 2

# Paired t-test
t_stat, p_value = ttest_rel(se_A, se_B)

# If p < 0.05: Models are significantly different
```

**Use Case:**
```python
# Compare:
Model A: Our TCN
Model B: Simple linear regression baseline

# If p < 0.05 and TCN has lower MSE:
# → TCN significantly better than baseline
```

### 6.3 Walk-Forward Validation

**Temporal Cross-Validation:**
```python
# Instead of single train/test split:
for window in rolling_windows:
    train_data = window[0:80%]
    test_data = window[80:100%]

    train_model(train_data)
    evaluate(test_data)

    # Collect metrics

# Analyze:
mean_accuracy = mean(accuracies)
std_accuracy = std(accuracies)

# If std is low → Robust
# If std is high → Unstable (overfitting or regime-dependent)
```

**Benefits:**

- Tests robustness over time
- Detects regime-dependent performance
- More realistic evaluation

---

## 7. Benchmarking Against Baselines

### 7.1 Random Baseline

**Performance:**
```python
Directional Accuracy: 50% (by definition)
RMSE: sqrt(variance_of_returns)
R²: 0.0 (explains nothing)
```

**Why Important:**

Any model must beat this; otherwise useless.

### 7.2 Naive Baseline (Persistence)

**Prediction:** Tomorrow's return = today's return
```python
y_pred[t] = y_true[t-1]

# For BTC:
# - Short-term momentum exists
# - This baseline can achieve 52-54% accuracy!
# - Our model must beat this
```

**Expected Performance:**
```python
Directional Accuracy: 52-54% (weak momentum)
R²: 0.05-0.15
```

### 7.3 Linear Regression Baseline

**Model:** Simple linear model on LOB features
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Expected:
Directional Accuracy: 55-58%
R²: 0.15-0.25
```

**Why Use:**

- Tests if deep learning is necessary
- If linear performs similarly → Use linear (simpler)
- If TCN much better → Justifies complexity

### 7.4 Literature Comparison

**FI-2010 Benchmark (Standard in LOB Prediction):**

| Model | Horizon | Accuracy |
|-------|---------|----------|
| Logistic Regression | 10 ticks | 58% |
| Random Forest | 10 ticks | 62% |
| LSTM | 10 ticks | 68% |
| **DeepLOB (CNN)** | **10 ticks** | **79%** |
| **Our TCN (Target)** | **10 steps** | **60-70%** |

**Note:** Direct comparison difficult due to:
- Different datasets (Finnish stocks vs BTC)
- Different markets (equities vs crypto)
- Different time periods

---

## 8. Evaluation Workflow

### 8.1 Complete Evaluation Checklist

**Step 1: Load Model and Data**
```python
☐ Load trained model checkpoint
☐ Load test set (unseen during training/validation)
☐ Verify data quality (no NaNs, outliers flagged)
☐ Apply same normalization as training
```

**Step 2: Generate Predictions**
```python
☐ model.eval() mode (disable dropout)
☐ torch.no_grad() (no gradient computation)
☐ Batch size appropriate for memory
☐ Save predictions to disk
```

**Step 3: Compute Regression Metrics**
```python
☐ MSE (optimization metric)
☐ RMSE (interpretable error)
☐ MAE (robust error)
☐ R² (variance explained)
☐ Per-horizon breakdown
```

**Step 4: Compute Classification Metrics**
```python
☐ Directional accuracy (overall and per-horizon)
☐ Confusion matrix
☐ Precision, Recall, F1
☐ Statistical significance test (binomial)
```

**Step 5: Compute Financial Metrics (Optional)**
```python
☐ Sharpe ratio (if trading strategy defined)
☐ Information ratio (vs benchmark)
☐ Maximum drawdown
```

**Step 6: Compare to Baselines**
```python
☐ Random baseline
☐ Naive (persistence) baseline
☐ Linear regression baseline
☐ Literature benchmarks
```

**Step 7: Visualize Results**
```python
☐ Prediction vs actual scatter plots
☐ Error distribution histograms
☐ Directional accuracy bar charts
☐ Time series of predictions (sample period)
```

**Step 8: Statistical Tests**
```python
☐ Binomial test (accuracy > 50%?)
☐ Diebold-Mariano (better than baseline?)
☐ Walk-forward validation (robust over time?)
```

**Step 9: Report Writing**
```python
☐ Summary table (all metrics)
☐ Best/worst case examples
☐ Limitations and caveats
☐ Practical implications
```

### 8.2 Red Flags

**Warning Signs of Issues:**

❌ **Train accuracy >> Test accuracy**
```python
Train: 85% directional accuracy
Test: 52% directional accuracy
→ Severe overfitting
```

❌ **Negative R²**
```python
R² = -0.5
→ Model worse than predicting mean
→ Fundamental problem with model or data
```

❌ **Extremely high accuracy**
```python
Directional accuracy: 95%+
→ Data leakage likely
→ Check for future information in features
```

❌ **Predictions all same sign**
```python
99% of predictions positive
→ Model collapsed to trivial solution
→ Check loss function and data balance
```

❌ **Unstable across time**
```python
Week 1: 70% accuracy
Week 2: 50% accuracy
Week 3: 75% accuracy
→ Regime-dependent, not robust
```

---

## 9. Reporting Results

### 9.1 Academic Format (Dissertation)

**Table: Model Performance**

| Metric | 1s | 5s | 10s | 30s | 60s |
|--------|----|----|-----|-----|-----|
| MSE | 0.0234 | 0.0312 | 0.0389 | 0.0512 | 0.0678 |
| RMSE (%) | 0.153 | 0.177 | 0.197 | 0.226 | 0.260 |
| MAE (%) | 0.112 | 0.139 | 0.157 | 0.183 | 0.212 |
| R² | 0.452*** | 0.381*** | 0.325*** | 0.246** | 0.179* |
| Dir. Acc. (%) | 68.4*** | 64.2*** | 61.8*** | 58.3** | 55.7* |

*p < 0.05, **p < 0.01, ***p < 0.001

**Narrative:**

"The TCN model achieved statistically significant directional accuracy across all prediction horizons (p < 0.05). Performance degraded with increasing horizon, as expected, from 68.4% at 1-second to 55.7% at 60-seconds. The model explained 45.2% of price variance at the 1-second horizon (R² = 0.452, p < 0.001), substantially outperforming the naive baseline (R² = 0.08)."

### 9.2 Visual Reporting

**Essential Plots:**

1. **Scatter: Predicted vs Actual**
   - Shows correlation visually
   - Identify systematic bias
   - Check for heteroskedasticity

2. **Histogram: Error Distribution**
   - Should be centered at 0
   - Check for fat tails
   - Identify outliers

3. **Bar Chart: Accuracy by Horizon**
   - Clear trend visible
   - Easy for non-technical audience
   - Compare to baselines

4. **Time Series: Sample Predictions**
   - Show model tracking actual prices
   - Identify failure cases
   - Build intuition

### 9.3 Executive Summary

**Template:**

"We evaluated a Temporal Convolutional Network for predicting Bitcoin price movements from Limit Order Book data. The model was trained on 600,000 snapshots and tested on 90,000 unseen samples. Key findings:

- **Accuracy:** 68% directional accuracy at 1-second horizon, declining to 56% at 60-seconds
- **Significance:** All results statistically significant (p < 0.001)
- **Comparison:** Outperforms linear baseline by 12 percentage points
- **Robustness:** Consistent performance across 4-week test period

The model demonstrates predictive power at short time horizons suitable for high-frequency trading applications, though performance degrades at longer horizons as expected from market microstructure theory."

---

## 10. Limitations and Caveats

### 10.1 Evaluation Limitations

**1. Historical Data Bias**
```python
# Backtest on 2024 Q1 (bull market)
Accuracy: 70% ✓

# Deploy in 2024 Q2 (bear market)
Accuracy: 52% ✗

# Market regime changed!
```

**Mitigation:** Evaluate on multiple market conditions

**2. Survivorship Bias**

Only evaluating on "normal" periods, excluding:
- Flash crashes
- Exchange outages
- Extreme volatility events

**Mitigation:** Stress testing on historical crises

**3. Overfitting to Test Set**
```python
# Implicit overfitting:
Try architecture A → Test → 60% accuracy
Try architecture B → Test → 62% accuracy
Try architecture C → Test → 65% accuracy ← Choose this

# Problem: "Test set" is now validation set!
```

**Mitigation:** Hold out final test set until very end

### 10.2 Practical Deployment Gaps

**Evaluation Doesn't Include:**

1. **Transaction costs:** 0.1% per trade reduces profitability
2. **Slippage:** Execution price ≠ predicted price
3. **Latency:** 44ms inference + 100ms network = 144ms delay
4. **Market impact:** Large orders move price against you
5. **Liquidity:** Not enough volume at predicted price

**Reality Check:**
```python
# Backtest: 65% accuracy → 15% annual return
# Live trading: 65% accuracy → 8% annual return

# Why?
# - Transaction costs: -3%
# - Slippage: -2%
# - Partial fills: -1.5%
# - Bad fills during volatility: -0.5%
```

---

## 11. References

**Evaluation Metrics:**

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: data mining, inference, and prediction*. Springer. (Chapter 7: Model Assessment and Selection)

2. James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An introduction to statistical learning*. Springer. (Chapter 5: Resampling Methods)

**Financial Metrics:**

3. Sharpe, W. F. (1994). The sharpe ratio. *Journal of portfolio management*, 21(1), 49-58.

4. Sortino, F. A., & Price, L. N. (1994). Performance measurement in a downside risk framework. *The Journal of Investing*, 3(3), 59-64.

**Statistical Testing:**

5. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & economic statistics*, 20(1), 134-144.

6. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of forecasting*, 13(2), 281-291.

**LOB Prediction Benchmarks:**

7. Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing*, 67(11), 3001-3012.

8. Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods. *Journal of Forecasting*, 37(8), 852-866.

**Market Microstructure:**

9. Hasbrouck, J. (2007). *Empirical market microstructure: The institutions, economics, and econometrics of securities trading*. Oxford University Press.

**Walk-Forward Testing:**

10. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

---

## Appendix A: Metric Quick Reference

| Metric | Formula | Range | Higher is Better? | Use Case |
|--------|---------|-------|------------------|----------|
| **MSE** | mean((y-ŷ)²) | [0, ∞) | No | Optimization |
| **RMSE** | √MSE | [0, ∞) | No | Interpretability |
| **MAE** | mean(\|y-ŷ\|) | [0, ∞) | No | Robustness |
| **R²** | 1 - SS_res/SS_tot | (-∞, 1] | Yes | Variance explained |
| **Dir. Acc.** | mean(sign(y)==sign(ŷ)) | [0, 1] | Yes | Trading |
| **Sharpe** | μ/σ | (-∞, ∞) | Yes | Risk-adjusted return |
| **Info Ratio** | (μ-μ_b)/σ_e | (-∞, ∞) | Yes | Active management |

---

## Appendix B: Sample Evaluation Report
```
================================================================
LOB PRICE PREDICTION MODEL - EVALUATION REPORT
================================================================

Model: TCN (895K parameters)
Dataset: BTCUSDT (Binance)
Training Period: 2024-10-01 to 2024-10-25 (600K samples)
Test Period: 2024-10-26 to 2024-10-31 (90K samples)
Date: 2024-11-09

================================================================
REGRESSION METRICS
================================================================

Horizon | MSE    | RMSE   | MAE    | R²     | p-value
--------|--------|--------|--------|--------|----------
1s      | 0.0234 | 0.153% | 0.112% | 0.452  | <0.001 ***
5s      | 0.0312 | 0.177% | 0.139% | 0.381  | <0.001 ***
10s     | 0.0389 | 0.197% | 0.157% | 0.325  | <0.001 ***
30s     | 0.0512 | 0.226% | 0.183% | 0.246  | <0.01  **
60s     | 0.0678 | 0.260% | 0.212% | 0.179  | <0.05  *

================================================================
CLASSIFICATION METRICS
================================================================

Horizon | Accuracy | Precision | Recall | F1-Score
--------|----------|-----------|--------|----------
1s      | 68.4%    | 67.8%     | 69.2%  | 0.685
5s      | 64.2%    | 63.5%     | 65.1%  | 0.643
10s     | 61.8%    | 61.0%     | 62.8%  | 0.619
30s     | 58.3%    | 57.4%     | 59.5%  | 0.584
60s     | 55.7%    | 54.9%     | 56.8%  | 0.558

Overall Directional Accuracy: 61.7%

================================================================
BASELINE COMPARISON
================================================================

Model          | 1s Accuracy | 10s Accuracy | 60s Accuracy
---------------|-------------|--------------|-------------
Random         | 50.0%       | 50.0%        | 50.0%
Naive          | 52.3%       | 51.8%        | 50.5%
Linear Reg.    | 56.1%       | 54.2%        | 52.1%
Our TCN        | 68.4% ✓     | 61.8% ✓      | 55.7% ✓

Improvement over Linear: +12.3 pp (1s), +7.6 pp (10s), +3.6 pp (60s)

================================================================
STATISTICAL SIGNIFICANCE
================================================================

Binomial Test (vs 50% random):
- All horizons: p < 0.001 (highly significant)
- Sample size: N = 90,000
- Confidence: 99.9%

Diebold-Mariano Test (vs Linear Regression):
- 1s: t = 23.4, p < 0.001 ***
- 10s: t = 18.7, p < 0.001 ***
- 60s: t = 8.2, p < 0.001 ***

Conclusion: TCN significantly outperforms baselines.

================================================================
KEY FINDINGS
================================================================

✓ Model demonstrates strong predictive power at short horizons
✓ Performance degrades with horizon (expected from theory)
✓ All results statistically significant
✓ Substantial improvement over simple baselines
✓ Consistent performance across test period

================================================================
LIMITATIONS
================================================================

- Evaluation on single month (Oct 2024)
- Bull market conditions (may not generalize to bear)
- No transaction costs included
- Assumes immediate execution
- Single symbol (BTCUSDT only)

================================================================
RECOMMENDED NEXT STEPS
================================================================

1. Evaluate on extended test period (6+ months)
2. Test in different market regimes (bull/bear/sideways)
3. Include transaction cost analysis
4. Walk-forward validation
5. Multi-symbol evaluation

================================================================
```

---

**End of Document**
