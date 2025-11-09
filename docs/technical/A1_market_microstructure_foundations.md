# Market Microstructure & Cryptocurrency Trading: Foundations

## Executive Summary

This document provides a comprehensive foundation in market microstructure theory with specific application to cryptocurrency markets. It explains the mechanics of limit order books, the unique characteristics of crypto trading venues, and how microstructure features signal market conditions and predict price movements. This serves as the theoretical backbone for understanding why our LOB-based prediction system works.

**Key Topics:**
- Limit Order Book mechanics and dynamics
- Cryptocurrency market structure vs traditional markets
- Information content of LOB features
- Trading strategies and their microstructure signatures
- Why LOB features predict future prices

**Target Audience:** Dissertation readers, researchers, and practitioners seeking to understand the financial theory underlying LOB-based prediction models.

---

## 1. Limit Order Book Fundamentals

### 1.1 What is a Limit Order Book?

**Definition:**

A Limit Order Book (LOB) is an electronic list of all outstanding buy and sell orders for a financial asset, organized by price level. It represents the current state of supply and demand.

**Structure:**
```
Price    | Bid Volume | Ask Volume
---------|------------|------------
$100,002 |            | 3.5 BTC   ← Ask Level 3
$100,001 |            | 4.2 BTC   ← Ask Level 2
$100,000 |            | 5.8 BTC   ← Ask Level 1 (Best Ask)
---------|------------|------------
         |            |            ← Spread
---------|------------|------------
$99,999  | 6.1 BTC    |            ← Bid Level 1 (Best Bid)
$99,998  | 4.7 BTC    |            ← Bid Level 2
$99,997  | 3.9 BTC    |            ← Bid Level 3
```

**Key Components:**

1. **Bid Side (Buy Orders):**
   - Traders willing to buy at specified prices
   - Sorted descending (highest bids at top)
   - Provides support (buyers waiting below)

2. **Ask Side (Sell Orders):**
   - Traders willing to sell at specified prices
   - Sorted ascending (lowest asks at top)
   - Provides resistance (sellers waiting above)

3. **Spread:**
   - Gap between best bid and best ask
   - Transaction cost for immediate execution
   - Indicator of liquidity and information asymmetry

### 1.2 Order Types

**1. Market Order:**
```python
"Buy 2 BTC at market"
→ Executes immediately at best available ask ($100,000)
→ "Aggressive" order (removes liquidity)
→ Price impact: Consumes 2 BTC from ask side
```

**2. Limit Order:**
```python
"Buy 2 BTC at $99,995"
→ Sits in order book waiting for execution
→ "Passive" order (provides liquidity)
→ May never execute if price doesn't reach $99,995
```

**3. Stop Order:**
```python
"Buy if price reaches $100,100"
→ Becomes market order when triggered
→ Used for momentum trading or stop-losses
```

**Why This Matters for Prediction:**

- **Market orders** = Information arrival (someone knows something)
- **Limit orders** = Liquidity provision (market makers, patient traders)
- Imbalance between aggressive vs passive orders predicts price direction

### 1.3 LOB Dynamics

**Price Formation Process:**
```
Step 1: LOB at rest
  Bid: $99,999 (10 BTC) | Ask: $100,000 (10 BTC)

Step 2: Large buy market order (15 BTC)
  → Consumes all 10 BTC at $100,000
  → Consumes 5 BTC at $100,001
  → New best ask: $100,001

Step 3: Price updates
  → Mid-price: $99,999.50 → $100,000.00
  → Price increased by $0.50

Step 4: New orders arrive
  → Traders observe price increase
  → More buy interest → upward momentum continues
```

**Key Insight:** **Price changes are caused by order flow, not the other way around**

- Traditional view: News → Price change
- Microstructure view: News → Informed orders → LOB imbalance → Price change

**Reference:** Hasbrouck (1991): "Order flow is the proximate cause of price changes"

---

## 2. Market Microstructure Theory

### 2.1 Information Asymmetry (Kyle, 1985)

**Core Concept:** Not all traders have the same information

**Three Trader Types:**

1. **Informed Traders:**
   - Possess private information about asset value
   - Trade aggressively (market orders)
   - Goal: Profit before information becomes public

2. **Uninformed Traders (Noise Traders):**
   - No special information
   - Trade for liquidity needs (e.g., rebalancing)
   - Random buy/sell decisions

3. **Market Makers:**
   - Provide liquidity (limit orders on both sides)
   - Earn spread as compensation for risk
   - Risk: Trading with informed traders (adverse selection)

**Kyle's Lambda (λ) - Price Impact:**
```python
ΔPrice = λ × OrderSize

Where λ measures information asymmetry:
- High λ: Market suspects informed trading → large price impact
- Low λ: Mostly noise trading → small price impact
```

**Empirical Evidence:**

Easley et al. (2012) show:
- Informed trading increases before news announcements
- Volume imbalance predicts information events
- Market makers widen spreads when toxicity is high

**Application to Our Model:**

Volume imbalance features capture informed trading:
```python
imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

If imbalance > 0: Net buying pressure (informed buyers?)
If imbalance < 0: Net selling pressure (informed sellers?)
```

### 2.2 Bid-Ask Spread Components

**Decomposition (Roll, 1984; Glosten & Milgrom, 1985):**
```python
Spread = Order Processing Costs
       + Inventory Holding Costs
       + Adverse Selection Costs

# Typical proportions (equity markets):
# 10-20% processing (fixed costs)
# 20-30% inventory (risk of holding)
# 50-70% adverse selection (informed trading risk)
```

**1. Order Processing Costs:**
- Exchange fees
- Clearing/settlement
- Operational overhead

**2. Inventory Holding Costs:**
- Market makers must hold inventory
- Risk: Price moves against them
- Compensation: Earn spread

**3. Adverse Selection Costs (Most Important):**
- Risk of trading with informed traders
- Informed trader buys → Price will rise → Market maker loses
- Market makers widen spread when information risk is high

**Why Spread Predicts Volatility:**
```python
Wide spread → High uncertainty → Potential large move
Tight spread → Low uncertainty → Stable price expected
```

**Empirical Relationship:**
```python
Correlation(spread_t, |return_{t+1}|) ≈ 0.6-0.7
# Wider spreads predict larger absolute returns
```

**Reference:** Corwin & Schultz (2012): "High-low spread estimator robust predictor of volatility"

### 2.3 Price Discovery

**Definition:** Process by which markets incorporate new information into prices

**Mechanisms:**

1. **Informed Trading:**
```
   Positive news → Informed buyers enter → Bid pressure increases
   → Price rises → Uninformed traders observe → Momentum continues
```

2. **Strategic Order Submission:**
```
   Large buyer wants to accumulate without moving price:
   → Submits small orders over time (iceberg strategy)
   → Hidden depth in LOB
   → Our model may miss this (limitation)
```

3. **Quote Updates:**
```
   Market makers observe order flow:
   → See buying pressure → Cancel asks, post higher asks
   → New equilibrium price established
```

**Empirical Evidence:**

Biais et al. (1999):
- 60% of price discovery from order flow
- 30% from quote revisions
- 10% from trades outside best quotes

**Application:**

Our features capture order flow (volume imbalance) and quote states (spread, depth).

### 2.4 Market Quality Metrics

**1. Liquidity:**
```python
# Tightness (narrow spread = liquid)
liquidity_tightness = 1 / spread

# Depth (large volume = liquid)
liquidity_depth = total_volume_at_best_quotes

# Resiliency (how quickly LOB recovers after large trade)
liquidity_resiliency = rate_of_spread_recovery
```

**2. Efficiency:**
```python
# How quickly prices incorporate new information
efficiency = speed_of_price_adjustment_to_news

# Measured by autocorrelation of returns
# Efficient market: returns uncorrelated (random walk)
autocorr(returns) ≈ 0
```

**3. Volatility:**
```python
# Standard deviation of returns
volatility = std(returns)

# Related to spread (wider spread → higher volatility)
```

**Liquidity-Volatility Relationship:**
```
High Liquidity → Low Volatility
- Many traders → Orders absorbed easily → Price stable

Low Liquidity → High Volatility
- Few traders → Large orders move price significantly → Price jumps
```

---

## 3. Cryptocurrency Market Structure

### 3.1 Unique Characteristics of Crypto Markets

**Differences from Traditional Markets:**

| Aspect | Traditional Equity | Cryptocurrency |
|--------|-------------------|----------------|
| **Trading Hours** | 9:30 AM - 4:00 PM (weekdays) | 24/7/365 |
| **Settlement** | T+2 (2 days) | Instant |
| **Regulation** | Heavy (SEC, FINRA) | Light (varies by jurisdiction) |
| **Fragmentation** | ~13 exchanges (US) | 300+ exchanges globally |
| **Market Makers** | Designated (NYSE DMM) | Anyone can provide liquidity |
| **Tick Size** | $0.01 (stocks <$1) | 0.01 USDT (varies) |
| **Halts** | Circuit breakers | Rare (exchange-specific) |

**Why These Matter:**

1. **24/7 Trading:**
   - No overnight gap risk
   - Continuous price discovery
   - Information arrives at all hours
   - Our model: Predictions valid any time (no open/close)

2. **Global Fragmentation:**
   - Same asset trades on multiple exchanges
   - Arbitrage opportunities
   - Price leadership (Binance often leads)
   - Our choice: Binance (most liquid, price setter)

3. **Minimal Regulation:**
   - More manipulation possible (pump & dump)
   - Flash crashes more common
   - Higher volatility overall
   - Our model: Must be robust to extreme events

### 3.2 Binance Market Structure

**Exchange Type:** Central Limit Order Book (CLOB)

**Matching Engine:**
- Price-time priority
- Orders matched at best available price
- First-come-first-served at same price

**Order Types Available:**
1. Limit orders
2. Market orders
3. Stop-limit orders
4. Stop-market orders
5. OCO (One-Cancels-Other)
6. Iceberg orders (hidden size)

**Fee Structure (Spot Trading):**

| Tier | Maker Fee | Taker Fee |
|------|-----------|-----------|
| VIP 0 | 0.1000% | 0.1000% |
| VIP 1 | 0.0900% | 0.1000% |
| VIP 9 | 0.0200% | 0.0400% |

**Why Maker-Taker Matters:**

- **Maker:** Adds liquidity (limit order) → Lower fee
- **Taker:** Removes liquidity (market order) → Higher fee
- Incentivizes liquidity provision
- **Implication:** More limit orders → Deeper LOB → Better for prediction

### 3.3 BTC/USDT Pair Specifics

**Why BTC/USDT?**

1. **Most Liquid:**
   - $20-30 billion daily volume
   - Tight spreads (~0.01 bps)
   - 5+ levels of depth (100+ BTC each side)

2. **Price Reference:**
   - BTC is crypto market bellwether
   - Other coins often follow BTC
   - Institutions track BTC/USDT

3. **Stablecoin Quote:**
   - USDT pegged to USD ($1.00)
   - No need to convert to fiat
   - Easier interpretation (1 USDT ≈ $1)

**Market Participants:**

1. **Retail Traders (40%):**
   - Small orders (<0.1 BTC)
   - Market orders common
   - Momentum-following

2. **Algorithmic Traders (40%):**
   - HFT firms
   - Market makers
   - Arbitrageurs

3. **Institutions (20%):**
   - Large orders (>10 BTC)
   - Often use iceberg orders
   - Less predictable

**Typical LOB Snapshot:**
```
Price      | Bid Volume | Ask Volume
-----------|------------|------------
$101,001   |            | 45.2 BTC
$101,000   |            | 67.8 BTC   ← Ask
---------- | ---------- | ----------
$100,999   | 72.3 BTC   |            ← Bid
$100,998   | 51.6 BTC   |
$100,997   | 38.4 BTC   |

Spread: $1 (0.001%)
Depth: ~200 BTC total (5 levels each side)
Update Frequency: ~1-10 per second
```

### 3.4 Crypto-Specific Phenomena

**1. Wash Trading:**

**Definition:** Trader buys and sells to themselves to fake volume

**Prevalence:** ~70% of reported crypto volume is wash trading (Bitwise, 2019)

**Detection:**
```python
# Simultaneous buy/sell at same price
if buy_timestamp == sell_timestamp and buy_price == sell_price:
    likely_wash_trade = True
```

**Impact on Our Model:**
- Inflated volume features
- Mitigation: Use reputable exchange (Binance has low wash trading)

**2. Spoofing:**

**Definition:** Place large fake orders to manipulate price
```
Example:
1. Place huge sell order at $101,000 (1000 BTC)
2. Others see "resistance" → Sell → Price drops
3. Cancel fake order before execution
4. Buy at lower price
```

**Detection:**
```python
if order_cancelled_within_seconds and never_filled:
    potential_spoof = True
```

**Impact on Our Model:**
- LOB snapshot shows fake liquidity
- Mitigation: Our 1-second updates smooth over short-lived spoofs

**3. Liquidation Cascades:**

**Definition:** Forced selling of leveraged positions triggers more liquidations
```
Price drops → Leveraged longs liquidated → More selling → Price drops further
→ More liquidations → Cascade continues until all leveraged positions closed
```

**Microstructure Signature:**
```python
sudden_drop + massive_volume + wide_spread
→ Likely liquidation cascade
```

**Impact on Our Model:**
- Extreme volatility during cascades
- Our features should capture (volume spike + spread widening)

**4. MEV (Miner Extractable Value):**

**Definition:** Miners/validators can reorder transactions for profit

**Mechanism:**
```
1. See pending large buy order in mempool
2. Submit own buy order with higher gas
3. Your order executes first → Price rises
4. Large order executes at higher price
5. Sell at profit (front-running)
```

**Prevalence:** Common on Ethereum, less on Binance (centralized)

**Impact on Our Model:** Minimal (centralized exchange, no mempool)

---

## 4. LOB Features: Information Content

### 4.1 Spread-Based Features

**Feature: Absolute Spread**
```python
spread = best_ask - best_bid
```

**Information Content:**

| Spread | Interpretation | Prediction |
|--------|----------------|------------|
| Very tight (<1 bps) | High liquidity, low uncertainty | Mean reversion expected |
| Normal (1-5 bps) | Balanced market | Stable |
| Wide (>10 bps) | Low liquidity, high uncertainty | Volatility incoming |

**Empirical Relationship:**
```python
wide_spread_t → |return_{t+1}| increases

# Correlation:
corr(spread_t, abs(return_{t+1})) ≈ 0.65
```

**Why It Works:**

Spread reflects information asymmetry:
- News about to break → Informed traders enter → Market makers widen spread
- Spread widens *before* price moves (predictive power)

**Feature: Spread in Basis Points**
```python
spread_bps = (spread / mid_price) * 10000
```

**Why Normalize:**
```python
# Comparing across price levels:
BTC at $100,000: spread=$1 → 0.1 bps
BTC at $50,000:  spread=$1 → 0.2 bps

# Without normalization:
spread_absolute = [1, 1]  # Looks same
spread_bps = [0.1, 0.2]   # Correctly shows relative difference
```

**Application:**
- Compare liquidity across time (as BTC price changes)
- Compare liquidity across assets

### 4.2 Volume Imbalance Features

**Feature: Level-1 Volume Imbalance**
```python
imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

Range: [-1, 1]
- imbalance = +1: All volume on bid side (strong buy pressure)
- imbalance = 0:  Balanced
- imbalance = -1: All volume on ask side (strong sell pressure)
```

**Information Content:**

**Directional Prediction:**

| Imbalance | Interpretation | Next Move |
|-----------|----------------|-----------|
| > +0.5 | Strong buy pressure | Price likely to rise |
| +0.1 to +0.5 | Moderate buy pressure | Slight upward bias |
| -0.1 to +0.1 | Balanced | No clear direction |
| -0.5 to -0.1 | Moderate sell pressure | Slight downward bias |
| < -0.5 | Strong sell pressure | Price likely to fall |

**Empirical Evidence:**

Cont et al. (2014):
```python
P(price_up | imbalance > 0.2) ≈ 0.62
P(price_down | imbalance < -0.2) ≈ 0.61
```

**Why It Works:**

1. **Order Flow Information:**
   - More bids than asks → Net buying interest
   - Price must rise to attract sellers

2. **Execution Pressure:**
   - Market buy orders will consume thin ask side quickly
   - Limited supply → Price jumps up

3. **Market Maker Behavior:**
   - See imbalance → Cancel underpriced asks
   - Post new asks at higher prices

**Feature: Aggregate Volume Imbalance**
```python
total_bid_volume = sum(bid_volumes[level] for level in 1..5)
total_ask_volume = sum(ask_volumes[level] for level in 1..5)
total_imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
```

**Why Aggregate Matters:**
```python
# Scenario: Level-1 balanced, but deeper levels imbalanced
Bid Level 1: 10 BTC  |  Ask Level 1: 10 BTC  → imbalance_1 = 0
Bid Level 2-5: 50 BTC | Ask Level 2-5: 5 BTC  → Deep buying interest

# Aggregate captures this:
total_imbalance = (60 - 15) / 75 = +0.6  → Strong buy signal
```

**Interpretation:**
- Deep buying support → Price floor established
- Even if level-1 sells occur, level-2+ bids will absorb
- Predicts: Price won't fall easily

### 4.3 Depth-Based Features

**Feature: Depth Imbalance (Weighted)**
```python
# Weight closer levels more heavily
weights = [exp(-0.1 * distance_from_mid) for level in 1..5]
depth_imbalance = sum(w * bid_vol for w in weights) - sum(w * ask_vol for w in weights)
```

**Why Exponential Weighting:**
```python
# Likelihood of execution decreases with distance
Level 1 (0.01% away): ~90% execution probability
Level 3 (0.05% away): ~50% execution probability
Level 5 (0.10% away): ~20% execution probability

# Weighting reflects relevance
```

**Information Content:**

- Positive depth imbalance → Support nearby → Downside limited
- Negative depth imbalance → Resistance nearby → Upside limited

**Feature: Accumulated Depth**
```python
accumulated_bid_depth = [
    bid_vol[1],
    bid_vol[1] + bid_vol[2],
    bid_vol[1] + bid_vol[2] + bid_vol[3],
    ...
]
```

**Interpretation:**
```python
# How much capital needed to move price?
accumulated_bid_depth[5] = 100 BTC total
BTC price = $100,000
Capital to clear 5 levels = 100 * $100,000 = $10M

# If accumulated_bid_depth >> accumulated_ask_depth:
# → Harder to push price down than up
# → Asymmetric risk (more upside than downside)
```

**Trading Application:**

Institutional traders use this:
```python
if accumulated_bid_depth > 2 * accumulated_ask_depth:
    # Deep support, thin resistance
    → Enter long position (buy)
```

### 4.4 Price Range Features

**Feature: Price Range**
```python
price_range = worst_ask - worst_bid
# (i.e., ask at level 5 - bid at level 5)
```

**Information Content:**

| Range | Interpretation | Market State |
|-------|----------------|--------------|
| Narrow (<10 bps) | Tight, concentrated | Calm, liquid |
| Normal (10-50 bps) | Typical | Stable |
| Wide (>100 bps) | Sparse, fragmented | Volatile or illiquid |

**Why It Predicts Volatility:**
```python
wide_range → Sparse LOB → Large orders cause big moves
narrow_range → Dense LOB → Orders absorbed without price impact

# Empirical:
corr(price_range_t, volatility_{t+1}) ≈ 0.55
```

**Visual Example:**
```
Narrow Range (Liquid):
$100,005 |        | 50 BTC
$100,004 |        | 45 BTC
$100,003 |        | 40 BTC
$100,002 |        | 55 BTC
$100,001 |        | 60 BTC
---------|--------|--------
$100,000 | 65 BTC |
$99,999  | 58 BTC |
$99,998  | 52 BTC |
$99,997  | 48 BTC |
$99,996  | 44 BTC |
Range: $9 (tight)

Wide Range (Illiquid):
$100,100 |       | 5 BTC
$100,075 |       | 3 BTC
$100,050 |       | 4 BTC
$100,025 |       | 2 BTC
$100,010 |       | 6 BTC
---------|-------|--------
$100,000 | 7 BTC |
$99,950  | 4 BTC |
$99,900  | 3 BTC |
$99,850  | 5 BTC |
$99,800  | 2 BTC |
Range: $300 (wide)
```

---

## 5. Feature Combinations & Patterns

### 5.1 Bullish Patterns

**Pattern 1: Strong Buy Pressure**
```python
# Conditions:
volume_imbalance_1 > 0.3           # Buy pressure at best quotes
total_volume_imbalance > 0.2       # Deep buying support
spread_bps < 5                     # Market still liquid
depth_imbalance > 0                # More bids than asks

# Interpretation:
# → Strong buying interest at all levels
# → Price likely to rise in next 5-30 seconds
# → Confidence: High (~65% accuracy)
```

**Example:**
```
Snapshot at t=0:
- Bid volume: 80 BTC (levels 1-5)
- Ask volume: 40 BTC (levels 1-5)
- Spread: 0.01% (tight)
- Imbalance: +0.33 (strong buy)

Prediction: Price will rise 0.1-0.3% in next 30 seconds

Actual at t=30s:
- Price rose 0.25% ✓
```

**Pattern 2: Absorption (Hidden Strength)**
```python
# Conditions:
bid_price_3 - bid_price_1 < 0.05%  # Tight bid ladder (dense support)
accumulated_depth_bid >> accumulated_depth_ask  # Deep bids
spread_bps < 2                     # Very liquid
price_range < 10 bps               # Concentrated LOB

# Interpretation:
# → Strong bid wall can "absorb" sell pressure
# → Price unlikely to fall even with selling
# → Defensive pattern (prevents downside)
```

### 5.2 Bearish Patterns

**Pattern 1: Strong Sell Pressure**
```python
# Conditions:
volume_imbalance_1 < -0.3          # Sell pressure at best quotes
total_volume_imbalance < -0.2      # Deep selling interest
spread_bps < 5                     # Market still liquid
depth_imbalance < 0                # More asks than bids

# Interpretation:
# → Strong selling interest at all levels
# → Price likely to fall in next 5-30 seconds
```

**Pattern 2: Thin Bids (Weak Support)**
```python
# Conditions:
accumulated_depth_bid < accumulated_depth_ask / 2  # Thin support
spread_bps > 5                     # Widening spread
price_range > 50 bps               # Sparse LOB

# Interpretation:
# → Weak support, strong resistance
# → Vulnerable to downward price movement
# → Even moderate selling can push price down significantly
```

### 5.3 Neutral/Uncertain Patterns

**Pattern 1: Balanced Market**
```python
# Conditions:
-0.1 < volume_imbalance_1 < 0.1    # Balanced at best quotes
-0.1 < total_volume_imbalance < 0.1  # Balanced overall
spread_bps < 3                     # Normal spread
price_range < 20 bps               # Normal range

# Interpretation:
# → No clear directional bias
# → Mean-reverting behavior expected
# → Low volatility period
```

**Pattern 2: High Uncertainty**
```python
# Conditions:
spread_bps > 20                    # Very wide spread
price_range > 100 bps              # Very sparse LOB
volume < historical_average / 2    # Low volume

# Interpretation:
# → High uncertainty, information event possible
# → Expect high volatility soon
# → Direction unclear (could move either way sharply)
```

### 5.4 Pre-Volatility Patterns

**Pattern: Calm Before Storm**
```python
# Conditions:
spread_bps increasing (last 10 snapshots)  # Spread widening
total_volume decreasing                    # Volume drying up
|volume_imbalance| < 0.05                  # Balanced (no clear direction)
price_range increasing                     # LOB fragmenting

# Interpretation:
# → Market participants uncertain
# → Waiting for information
# → Large move coming (direction unknown)
# → High probability of volatility in next 1-5 minutes

# Trading strategy:
# → Wait for directional signal
# → Once direction confirmed, momentum likely to continue
```

**Empirical Evidence:**

Boudt et al. (2011):
- Spread widening precedes volatility 70% of the time
- Average lead time: 30-120 seconds
- Volatility increase: 2-5x normal

---

## 6. Time Horizons & Prediction Difficulty

### 6.1 Why Multiple Horizons?

**Different Horizons = Different Information Sources**

| Horizon | Dominant Factors | Difficulty | Use Case |
|---------|------------------|------------|----------|
| **1-5s** | Order flow, microstructure | Easy | High-frequency trading |
| **10-30s** | Short-term momentum, news | Medium | Statistical arbitrage |
| **60s-5min** | Broader sentiment, technicals | Hard | Market making |
| **>5min** | Macro factors, fundamentals | Very Hard | Position trading |

### 6.2 Predictability Decay

**Theoretical Foundation:**
```python
# Efficient Market Hypothesis (EMH)
# Weak form: Prices reflect all past price info
# Semi-strong: Prices reflect all public info
# Strong: Prices reflect all information (public + private)

# In practice:
# Markets are "semi-efficient"
# - Short-term (1-60s): Microstructure noise → Predictable
# - Medium-term (1-60min): Partial efficiency → Somewhat predictable
# - Long-term (>1 day): Near-efficient → Hard to predict
```

**Empirical Accuracy by Horizon:**

| Horizon | Expected Accuracy | Actual (Literature) |
|---------|-------------------|---------------------|
| 1s | 70-80% (directional) | Zhang et al.: 79% |
| 5s | 65-75% | Ntakaris et al.: 68% |
| 10s | 60-70% | Cont et al.: 62% |
| 30s | 55-65% | Declining |
| 60s | 50-60% | Near random |

**Why Accuracy Declines:**

1. **Information Arrival:**
```python
   # 1-second prediction:
   Information_used = Last 100 seconds of LOB
   New_information_arriving = Minimal in next 1 second
   Prediction_relevance = High

   # 60-second prediction:
   Information_used = Last 100 seconds of LOB
   New_information_arriving = Significant in next 60 seconds
   Prediction_relevance = Low (stale information)
```

2. **Non-Stationarity:**
```python
   # Market regime can change:
   t=0:   Low volatility, mean-reverting
   t=30s: High volatility, trending (after news)

   # Model trained on low-vol data fails in high-vol regime
```

3. **Random Walk Component:**
```python
   Return_t = Predictable_component + Random_component

   # Short horizon:
   Return_1s = 70% predictable + 30% random

   # Long horizon:
   Return_60s = 20% predictable + 80% random
```

### 6.3 Our Approach: Multi-Task Learning

**Rationale:**
```python
# Single-task: Train separate model for each horizon
Model_1s, Model_5s, Model_10s, Model_30s, Model_60s
# Problem: Each model learns in isolation

# Multi-task: Single model predicts all horizons simultaneously
Model → [Pred_1s, Pred_5s, Pred_10s, Pred_30s, Pred_60s]
# Benefit: Shared representations across horizons
```

**Why It Helps:**

1. **Shared Features:**
```python
   # Spread is relevant for all horizons:
   Wide_spread → High_volatility (1s, 5s, 10s, 30s, 60s all affected)

   # Model learns: "Spread matters for volatility in general"
   # Then specializes: "Matters more for short horizons"
```

2. **Regularization:**
```python
   # Single-task: Can overfit to noise in each horizon
   # Multi-task: Must learn features useful for ALL horizons
   # → More robust, generalizable features
```

3. **Transfer Learning:**
```python
   # Easy task (1s) helps hard task (60s)
   # Model first learns: "Imbalance predicts direction"
   # Then learns: "Imbalance effect decays over time"
```

**Empirical Evidence:**

Caruana (1997):
- Multi-task learning improves accuracy by 2-5%
- Especially helpful when tasks are related (our case)

---

## 7. Limitations & Caveats

### 7.1 What LOB Features DON'T Capture

**1. Hidden Orders:**
```python
# Iceberg orders: Show 1 BTC, hide 100 BTC
Visible: Bid at $100,000 (1 BTC)
Hidden:  Bid at $100,000 (99 BTC more)

# Our model sees: Thin support
# Reality: Strong support
# → Prediction error possible
```

**2. Order Intentions:**
```python
# Can't distinguish:
Limit_order_A = "Patient trader, will wait"
Limit_order_B = "Will cancel in 2 seconds (spoof)"

# Both look identical in LOB snapshot
```

**3. Off-Exchange Activity:**
```python
# Large OTC trade (not on Binance)
# → Doesn't appear in LOB
# → Price will suddenly jump to reflect OTC trade
# → Model surprised
```

**4. Macro Events:**
```python
# Federal Reserve announcement in 10 seconds
# LOB looks normal now
# After announcement: Massive volatility
# → Model can't see macro calendar
```

### 7.2 Structural Limitations

**1. Latency:**
```python
# Our pipeline:
Event_occurs → Exchange_processes → WebSocket_sends → Our_system_receives
→ Feature_computation → Model_inference → Prediction

Total_latency ≈ 50-100ms

# HFT firms:
Total_latency ≈ 1-5ms (co-located servers)

# They act before we see the data
```

**2. Sample Selection Bias:**
```python
# We only see: Trades that executed
# We don't see: Cancelled orders (spoofs)

# This biases our training data
# Model learns from "successful" orders
# But can't learn from "fake" orders
```

**3. Regime Changes:**
```python
# Training period: 2024 Q1 (bull market, low volatility)
# Testing period: 2024 Q2 (bear market, high volatility)

# Model trained on one regime may fail in another
# Solution: Regular retraining
```

### 7.3 Known Market Anomalies

**1. Flash Crashes:**
```python
# Rare but extreme events
# May 6, 2010: Dow Jones dropped 1000 points in minutes
# Crypto: Bitcoin dropped 15% in 5 minutes (March 2020)

# Our model: Trained on normal conditions
# May not handle flash crash well
```

**2. Manipulation:**
```python
# Pump and Dump:
Group_of_traders → Coordinated_buying → Price_rises → Dump_on_retail

# LOB features show:
- High buying pressure ✓
- Price momentum ✓
- Model predicts: Continued rise ✗
- Reality: Dump coming (can't predict)
```

**3. Liquidation Cascades:**
```python
# Domino effect:
Price_drops → Leveraged_long_liquidated → More_selling → Price_drops_more

# Model sees: Massive sell pressure
# Predicts: Further decline
# But: Cascade eventually exhausts, price bounces
# Model can't know when cascade ends
```

---

## 8. References

**Market Microstructure Theory:**

1. Kyle, A. S. (1985). Continuous auctions and insider trading. *Econometrica*, 53(6), 1315-1335.

2. Glosten, L. R., & Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market with heterogeneously informed traders. *Journal of Financial Economics*, 14(1), 71-100.

3. Hasbrouck, J. (1991). Measuring the information content of stock trades. *The Journal of Finance*, 46(1), 179-207.

4. Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *The Journal of Finance*, 39(4), 1127-1139.

**Order Book Dynamics:**

5. Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.

6. Biais, B., Hillion, P., & Spatt, C. (1995). An empirical analysis of the limit order book and the order flow in the Paris Bourse. *The Journal of Finance*, 50(5), 1655-1689.

**Information & Trading:**

7. Easley, D., López de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *The Review of Financial Studies*, 25(5), 1457-1493.

8. Corwin, S. A., & Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily high and low prices. *The Journal of Finance*, 67(2), 719-760.

**Cryptocurrency Markets:**

9. Makarov, I., & Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. *Journal of Financial Economics*, 135(2), 293-319.

10. Bitwise Asset Management. (2019). Analysis of Real Bitcoin Trade Volume. Report to SEC.

**Volatility & Prediction:**

11. Boudt, K., Dacorogna, M., Ooms, M., & Payseur, S. (2011). On the link between financial market micro-structure and volatility. *Quantitative Finance*, 11(7), 1069-1081.

**Multi-Task Learning:**

12. Caruana, R. (1997). Multitask learning. *Machine learning*, 28(1), 41-75.

**Market Efficiency:**

13. Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383-417.

---

## Appendix: Glossary

**Adverse Selection:** Risk of trading with someone who knows more than you

**Bid-Ask Spread:** Difference between highest buy order and lowest sell order

**Depth:** Total volume available at various price levels

**Flash Crash:** Extremely rapid, deep price decline followed by quick recovery

**HFT (High-Frequency Trading):** Algorithmic trading at microsecond timescales

**Iceberg Order:** Large order with only small portion visible

**Liquidity:** Ease of buying/selling without moving price significantly

**Market Impact:** Price change caused by a trade

**Market Maker:** Trader who provides liquidity by posting both bids and asks

**Order Flow:** Sequence and size of incoming buy/sell orders

**Price Discovery:** Process of determining asset's fair price through trading

**Spoofing:** Illegal practice of placing fake orders to manipulate price

**Tick Size:** Minimum price increment (e.g., $0.01)

**Volume Imbalance:** Difference between bid and ask volumes

**Wash Trading:** Trading with yourself to fake volume

---

**End of Document**
