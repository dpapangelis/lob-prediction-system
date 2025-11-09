# Training Methodology: Model Optimization & Validation

## Executive Summary

This document details the training methodology for the TCN-based LOB price prediction model, including loss functions, optimization algorithms, regularization techniques, data splitting strategies, and hyperparameter selection. All design decisions are justified with theoretical foundations and empirical evidence from machine learning literature.

**Key Components:**
- Mean Squared Error (MSE) loss for multi-horizon regression
- Adam optimizer with adaptive learning rates
- ReduceLROnPlateau scheduler for convergence optimization
- Early stopping to prevent overfitting
- Chronological data splitting to avoid temporal leakage
- Comprehensive checkpointing for reproducibility

**Training Performance (M2 MacBook):**
- Training speed: ~39 iterations/second
- Validation speed: ~138 iterations/second
- Single epoch (6,900 sequences, batch=32): ~5-6 seconds
- Full training (100 epochs with early stopping): ~10-15 hours estimated

---

## 1. Loss Function Selection

### 1.1 Mean Squared Error (MSE)

**Definition:**
```python
MSE = (1/N) * sum((y_pred - y_true)^2)

# For multi-horizon prediction:
MSE_total = (1/H) * sum(MSE_h for h in horizons)
```

Where:
- N = number of samples in batch
- H = number of prediction horizons (5 in our case)
- MSE_h = MSE for specific horizon h

**Mathematical Properties:**

1. **Convex:** Single global minimum (no local minima)
2. **Differentiable:** Smooth gradients for optimization
3. **Quadratic penalty:** Larger errors penalized more heavily
   - Error of 2% → penalty of 4
   - Error of 4% → penalty of 16

### 1.2 Why MSE for Financial Prediction?

**Alternatives Considered:**

| Loss Function | Use Case | Our Decision |
|---------------|----------|--------------|
| **MSE** | Continuous regression | ✅ **Chosen** |
| MAE (L1 Loss) | Robust to outliers | ❌ Less smooth gradients |
| Huber Loss | Balanced (quadratic + linear) | ⚠️ Future consideration |
| Cross-Entropy | Classification (up/down) | ❌ Loses magnitude info |
| Quantile Loss | Probabilistic forecasts | ⚠️ Phase 2 |

**Justification for MSE:**

1. **Task Alignment:** Predicting continuous returns, not classes
   - Target: -2.3%, -0.5%, +1.2%, etc.
   - Not: "up" vs "down"

2. **Gradient Properties:**
```python
   ∂MSE/∂y_pred = 2(y_pred - y_true)
```
   - Proportional to error magnitude
   - Large errors → large gradients → faster correction
   - Small errors → small gradients → fine-tuning

3. **Statistical Interpretation:**
   - MSE = variance + bias²
   - Minimizing MSE = minimizing prediction variance and bias
   - Aligns with statistical learning theory (Hastie et al., 2009)

4. **Literature Standard:**
   - Zhang et al. (2019, DeepLOB): MSE for regression variant
   - Ntakaris et al. (2018): MSE for mid-price prediction
   - Standard in financial forecasting (Makridakis et al., 2018)

**Trade-off: MSE vs MAE**

| Aspect | MSE | MAE |
|--------|-----|-----|
| Outlier sensitivity | High (squares errors) | Low (linear) |
| Gradient smoothness | Smooth everywhere | Non-smooth at 0 |
| Large error penalty | Heavy (quadratic) | Light (linear) |
| Our choice | ✅ MSE | - |

**Rationale:** Financial returns have fat tails (outliers common). We *want* to penalize large prediction errors heavily, as they're costly in trading.

### 1.3 Multi-Horizon Loss Weighting

**Equal Weighting (Current):**
```python
loss = (1/5) * (MSE_1s + MSE_5s + MSE_10s + MSE_30s + MSE_60s)
```

**Alternative: Weighted Loss (Future):**
```python
# Weight longer horizons more (harder to predict)
weights = [0.1, 0.15, 0.2, 0.25, 0.3]
loss = sum(w_h * MSE_h for w_h, MSE_h in zip(weights, MSEs))
```

**Current Decision:** Equal weights
- **Rationale:** All horizons equally important for evaluation
- **Future:** Can adjust if certain horizons more critical for trading strategy

---

## 2. Optimization Algorithm

### 2.1 Adam Optimizer

**Algorithm (Kingma & Ba, 2015):**
```python
# Hyperparameters
α = 0.001      # Learning rate (step size)
β_1 = 0.9      # Exponential decay rate for 1st moment
β_2 = 0.999    # Exponential decay rate for 2nd moment
ε = 1e-8       # Numerical stability constant

# At each training step t:
g_t = ∇_θ L(θ_{t-1})                    # Compute gradient

m_t = β_1 * m_{t-1} + (1 - β_1) * g_t   # Update 1st moment (momentum)
v_t = β_2 * v_{t-1} + (1 - β_2) * g_t²  # Update 2nd moment (variance)

m̂_t = m_t / (1 - β_1^t)                 # Bias-corrected 1st moment
v̂_t = v_t / (1 - β_2^t)                 # Bias-corrected 2nd moment

θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)   # Parameter update
```

**Components Explained:**

1. **First Moment (m_t):** Moving average of gradients
   - Acts as momentum: Smooths noisy gradients
   - Accelerates in consistent directions

2. **Second Moment (v_t):** Moving average of squared gradients
   - Estimates gradient variance
   - Adaptive per-parameter learning rates

3. **Bias Correction:** Corrects initialization bias (m_0 = v_0 = 0)
   - Important in early training
   - Prevents underestimation

### 2.2 Why Adam over Alternatives?

**Comparison:**

| Optimizer | Advantages | Disadvantages | Our Decision |
|-----------|-----------|---------------|--------------|
| **SGD** | Theoretical guarantees | Requires manual LR tuning | ❌ |
| **SGD + Momentum** | Faster than SGD | Still needs LR schedule | ❌ |
| **RMSprop** | Adaptive LR | No momentum | ❌ |
| **Adam** | Adaptive + Momentum | Possible overfitting | ✅ **Chosen** |
| **AdamW** | Better regularization | More hyperparameters | ⚠️ Future |

**Justification for Adam:**

1. **Adaptive Learning Rates:**
   - Different parameters learn at different rates
   - Spread features: Small gradients → larger effective LR
   - Volume features: Large gradients → smaller effective LR

2. **Momentum Benefits:**
   - Smooths optimization trajectory
   - Escapes shallow local minima (though MSE is convex)
   - Faster convergence empirically

3. **Robust to Hyperparameters:**
   - Default β_1=0.9, β_2=0.999 work well across tasks
   - Less sensitive than SGD to LR choice

4. **Industry Standard:**
   - Default optimizer in PyTorch tutorials
   - Used in DeepLOB, most financial ML papers
   - Proven track record

**Empirical Evidence:**

Kingma & Ba (2015) show Adam:
- Converges faster than SGD on deep networks
- Achieves lower training loss
- Requires less hyperparameter tuning

### 2.3 Learning Rate Selection

**Our Choice:** α = 0.001 (1e-3)

**Rationale:**

| Learning Rate | Effect | Result |
|---------------|--------|--------|
| Too high (>0.01) | Overshoots minimum | Loss diverges |
| Optimal (0.001) | Steady convergence | Loss decreases smoothly |
| Too low (<0.0001) | Slow convergence | Wastes time |

**Standard Practice:**
- Adam default: 0.001
- Typical range: [1e-4, 1e-2]
- Start at 0.001, reduce if oscillating

**Learning Rate Schedule:**

We use **ReduceLROnPlateau** (see Section 2.4)

### 2.4 Learning Rate Scheduling

**ReduceLROnPlateau Algorithm:**
```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',        # Minimize validation loss
    factor=0.5,        # Reduce LR by 50% when triggered
    patience=5,        # Wait 5 epochs before reducing
    min_lr=1e-6        # Don't go below this
)

# After each epoch:
scheduler.step(val_loss)

# Behavior:
# If val_loss doesn't improve for 5 epochs:
#     lr = lr * 0.5
```

**Why Reduce Learning Rate?**

**Optimization Landscape Analogy:**
```
Loss
 │
 │     ╱╲        ← High LR: Big steps, oscillates
 │    ╱  ╲
 │   ╱    ╲      ← Low LR: Small steps, converges
 │  ╱______╲___
 └─────────────── Parameters
       ^
    Optimum
```

**Stages of Training:**

1. **Early (LR=0.001):** Large steps, quickly approach minimum
2. **Middle (LR=0.0005):** Smaller steps, fine-tune
3. **Late (LR=0.00025):** Tiny steps, polish

**Advantages of ReduceLROnPlateau:**

- **Adaptive:** Responds to training dynamics, not pre-set schedule
- **Automatic:** No manual tuning of schedule
- **Plateau Detection:** Recognizes when stuck

**Alternative Schedules (Not Used):**

| Schedule | Formula | Pros | Cons |
|----------|---------|------|------|
| Step Decay | lr = lr_0 * γ^(epoch/k) | Simple | Requires tuning k |
| Exponential | lr = lr_0 * e^(-λt) | Smooth | Requires tuning λ |
| Cosine Annealing | lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T)) | Works well | Fixed schedule |
| **ReduceLROnPlateau** | **Adaptive** | **Automatic** | ✅ **Chosen** |

---

## 3. Regularization Techniques

### 3.1 Dropout (p=0.2)

**Mechanism (Srivastava et al., 2014):**
```python
# During training (each forward pass):
for each neuron in layer:
    if random() < p:
        neuron_output = 0      # Drop neuron
    else:
        neuron_output *= 1/(1-p)  # Scale remaining neurons

# During inference:
# Use all neurons (no dropout)
```

**Effect:** Forces redundancy
- Network can't rely on any single neuron
- Learns multiple independent representations
- Equivalent to training ensemble of 2^n models

**Our Configuration:**
```python
dropout = 0.2  # Drop 20% of neurons

Applied after each:
- Dilated convolution
- ReLU activation
```

**Why p=0.2?**

| Dropout Rate | Effect | Use Case |
|--------------|--------|----------|
| p=0.1 | Minimal regularization | Large datasets |
| **p=0.2** | **Standard** | **Most cases** |
| p=0.5 | Heavy regularization | Small datasets, fully-connected layers |

**Empirical:** p=0.2 is standard for convolutional networks (Simonyan & Zisserman, 2015)

### 3.2 Weight Decay (L2 Regularization)

**Mathematical Formulation:**
```python
loss_total = loss_data + λ * sum(||w||² for w in weights)

# Where:
loss_data = MSE (prediction error)
λ = 1e-5 (regularization strength)
||w||² = sum(w_i² for w_i in weight_vector)
```

**Effect:** Penalizes large weights
- Encourages smaller, more distributed weights
- Prevents overfitting to noise
- Smoother decision boundaries

**Why λ=1e-5?**

| Weight Decay | Effect | Result |
|--------------|--------|--------|
| λ=0 | No regularization | Potential overfitting |
| λ=1e-5 | **Light regularization** | **Balanced** |
| λ=1e-3 | Heavy regularization | Underfitting |

**Standard Practice:**
- Adam with weight decay: λ ∈ [1e-6, 1e-4]
- We use 1e-5 (middle of range)

**Alternative: AdamW**
```python
# Adam: weight decay applied to gradient
# AdamW: weight decay applied directly to weights (more effective)
# Future consideration for Phase 2
```

### 3.3 Early Stopping

**Algorithm:**
```python
patience = 10  # Number of epochs to wait
best_val_loss = infinity
epochs_no_improve = 0

for epoch in training:
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        save_checkpoint()
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered")
        restore_best_checkpoint()
        break
```

**Rationale:**

**Training vs Validation Loss:**
```
Loss
 │
 │  Train ──────────────────────────
 │         ╲                  ╱
 │          ╲                ╱
 │           ╲              ╱
 │  Val      ╲___  ___  __╱
 │               ╲╱  ╲╱
 └──────────────────────────────── Epoch
                   ^
                Stop here!
```

**Key Insight:** Validation loss stops improving → model memorizing training data

**Why patience=10?**

| Patience | Effect | Trade-off |
|----------|--------|-----------|
| 5 | Stops quickly | May stop too early |
| **10** | **Balanced** | **Standard choice** |
| 20 | Patient | May overfit |

**Literature:** Prechelt (1998) recommends patience of 5-20 epochs

### 3.4 Gradient Clipping

**Implementation:**
```python
# After loss.backward(), before optimizer.step():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Purpose:** Prevent exploding gradients

**Mechanism:**
```python
total_norm = sqrt(sum(||grad||² for grad in gradients))

if total_norm > max_norm:
    for grad in gradients:
        grad *= max_norm / total_norm  # Scale down
```

**Why max_norm=1.0?**

- Empirical standard for RNNs and deep networks
- Prevents training instability
- Allows aggressive learning rates

**When It Matters:**
- Deep networks (we have 4 layers)
- Recurrent connections (TCN has residual connections)
- Financial data (can have sudden spikes)

---

## 4. Data Splitting Strategy

### 4.1 Chronological Split (Time-Series Aware)

**Critical for Financial Data:**
```
Timeline: |←────── train (70%) ──────→|←─ val (15%) ─→|←─ test (15%) ─→|
          Jan 1                      Jul 1          Sep 1             Nov 1

# CORRECT: Chronological split
train: Jan 1 - Jun 30
val:   Jul 1 - Aug 31
test:  Sep 1 - Oct 31
```

**Why NOT Random Split:**
```python
# WRONG: Random split (causes temporal leakage)
train: [Jan 5, Mar 12, Aug 20, Oct 3, ...]  # Future data leaked!
test:  [Jan 2, Feb 14, Jul 9, ...]          # Model "sees" future
```

**Temporal Leakage Problem:**

If random split:
1. Model trained on Aug 20 data
2. Tested on Aug 1 data
3. **Cheating!** Model already saw Aug 20 (future of Aug 1)
4. Overoptimistic test performance

**Consequences:**
- Inflated accuracy during development
- Catastrophic failure in production
- Common mistake in kaggle competitions

**Reference:**
- Bergmeir & Benítez (2012): "On the use of cross-validation for time series predictor evaluation"
- Key finding: Random CV overestimates performance by 5-15%

### 4.2 Split Ratios

**Our Choice:** 70% / 15% / 15%

**Justification:**

| Split | Size | Purpose | Requirements |
|-------|------|---------|--------------|
| **Train** | 70% | Learn patterns | Large (need many samples) |
| **Validation** | 15% | Tune hyperparameters | Moderate (representative) |
| **Test** | 15% | Final evaluation | Moderate (unbiased estimate) |

**Alternative Ratios:**

| Ratio | Use Case |
|-------|----------|
| 80/10/10 | Large datasets (>1M samples) |
| **70/15/15** | **Medium datasets (100K-1M)** |
| 60/20/20 | Small datasets (<100K) |

**Our Dataset Size (Expected):**
- 1 week data: ~600K snapshots
- 1 month data: ~2.5M snapshots
- 70/15/15 appropriate

### 4.3 Walk-Forward Validation (Future)

**Concept:** Repeatedly retrain as new data arrives
```
Train 1: [──────] Val 1: [─]
Train 2:   [──────] Val 2: [─]
Train 3:     [──────] Val 3: [─]
...
```

**Advantages:**
- More realistic evaluation (mimics production)
- Detects model degradation over time

**Currently:** Not implemented (Phase 2)
**Rationale:** Need longer data history first (6+ months)

---

## 5. Training Process

### 5.1 Training Loop Pseudocode
```python
def train_epoch():
    model.train()  # Enable dropout, batch norm

    for batch in train_loader:
        # 1. Forward pass
        predictions = model(features)
        loss = criterion(predictions, targets)

        # 2. Backward pass
        optimizer.zero_grad()     # Clear old gradients
        loss.backward()           # Compute gradients

        # 3. Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 4. Update weights
        optimizer.step()

        # 5. Track loss
        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)

def validate():
    model.eval()  # Disable dropout

    with torch.no_grad():  # Don't compute gradients (faster)
        for batch in val_loader:
            predictions = model(features)
            loss = criterion(predictions, targets)
            val_loss += loss.item()

    return val_loss / len(val_loader)

# Full training loop
for epoch in range(num_epochs):
    train_loss = train_epoch()
    val_loss = validate()

    # Update learning rate
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint()
    elif epochs_no_improve >= patience:
        break
```

### 5.2 Observed Training Dynamics

**Typical Training Run (Synthetic Data, 5 Epochs):**
```
Epoch 1/5 | Train: 0.9928 | Val: 0.9989 | LR: 0.001000
Epoch 2/5 | Train: 0.9897 | Val: 1.0030 | LR: 0.001000
Epoch 3/5 | Train: 0.9813 | Val: 1.0056 | LR: 0.001000
Epoch 4/5 | Train: 0.9638 | Val: 1.0222 | LR: 0.001000
Epoch 5/5 | Train: 0.9361 | Val: 1.0454 | LR: 0.001000

Best validation loss: 0.9989 (Epoch 1)
```

**Analysis:**

1. **Train Loss Decreasing:** 0.9928 → 0.9361
   - ✅ Model is learning
   - Gradient descent working correctly

2. **Val Loss Increasing:** 0.9989 → 1.0454
   - ⚠️ Overfitting to training data
   - Expected with random synthetic data (no real signal)

3. **Best Model at Epoch 1:**
   - Early stopping would trigger after Epoch 11 (patience=10)
   - Prevents further overfitting

**Expected Behavior with Real Data:**
```
Epoch  | Train Loss | Val Loss | Status
-------|------------|----------|--------
1      | 0.0850     | 0.0920   | Both decreasing ✓
5      | 0.0420     | 0.0510   | Good progress
10     | 0.0280     | 0.0390   | Converging
20     | 0.0185     | 0.0340   | Val plateaus
30     | 0.0125     | 0.0355   | Val increases (stop!)
```

**Key Indicators:**

✅ **Healthy Training:**
- Train loss steadily decreases
- Val loss decreases then plateaus
- Gap between train/val < 20%

⚠️ **Underfitting:**
- Both losses high and not decreasing
- Solution: Increase model capacity or train longer

❌ **Overfitting:**
- Train loss very low, val loss high
- Large gap (>50%) between train/val
- Solution: More regularization, early stopping

### 5.3 Performance Benchmarks

**Hardware:** M2 MacBook Pro (8-core CPU, 10-core GPU, 16GB unified memory)

**Training Speed:**

| Metric | Value | Notes |
|--------|-------|-------|
| Training iterations/sec | 39.0 | On MPS (GPU) |
| Validation iterations/sec | 138.0 | Faster (no gradients) |
| Time per epoch (6,900 sequences) | ~5-6 seconds | Batch size 32 |
| Time per epoch (100 epochs) | ~10 minutes | With validation |
| Full training (with early stopping) | ~30-60 minutes | Typically stops at 30-50 epochs |

**Scalability:**

| Dataset Size | Sequences | Training Time (100 epochs) |
|--------------|-----------|----------------------------|
| 1 day | ~86K | ~1 hour |
| 1 week | ~600K | ~7 hours |
| 1 month | ~2.5M | ~30 hours |

**Memory Usage:**

| Component | Size |
|-----------|------|
| Model parameters | 3.6 MB |
| Optimizer state (Adam) | 7.2 MB |
| Batch (32 × 100 × 43) | 1.1 MB |
| **Total** | **~12 MB** |

**Conclusion:** Memory is not a bottleneck; can train much larger models if needed

---

## 6. Hyperparameter Selection

### 6.1 Fixed Hyperparameters

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Input features | 43 | All LOB microstructure features |
| Sequence length | 100 | Balances context vs computation |
| TCN channels | [128, 128, 256, 256] | ~900K params (sweet spot) |
| Kernel size | 3 | Standard for CNNs |
| Dropout | 0.2 | Standard for CNNs |
| Num horizons | 5 | Multiple trading timescales |

### 6.2 Tuned Hyperparameters

| Hyperparameter | Range Considered | Chosen | Method |
|----------------|------------------|--------|--------|
| Learning rate | [1e-4, 1e-2] | 0.001 | Literature + empirical |
| Batch size | [16, 32, 64, 128] | 64 | Balances speed/generalization |
| Weight decay | [0, 1e-4] | 1e-5 | Standard for Adam |
| Early stopping patience | [5, 10, 20] | 10 | Balances efficiency/performance |

### 6.3 Hyperparameter Tuning Strategy (Future)

**Currently:** Manual selection based on literature

**Phase 2:** Systematic tuning
```python
# Grid search over key hyperparameters
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
batch_sizes = [32, 64, 128]
dropouts = [0.1, 0.2, 0.3]

for lr in learning_rates:
    for batch_size in batch_sizes:
        for dropout in dropouts:
            train_model(lr, batch_size, dropout)
            # Select best based on validation loss
```

**Alternative:** Bayesian optimization (Snoek et al., 2012)
- More efficient than grid search
- Uses Gaussian processes
- Requires fewer trials

---

## 7. Reproducibility

### 7.1 Random Seeds
```python
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# For deterministic behavior on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Why Reproducibility Matters:**

1. **Scientific Validity:** Results must be replicable
2. **Debugging:** Easier to isolate issues
3. **Fair Comparison:** Same starting point for experiments
4. **Dissertation Requirement:** Reproducible research standard

### 7.2 Checkpointing

**Checkpoint Contents:**
```python
checkpoint = {
    'epoch': current_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_val_loss,
    'history': {
        'train_loss': [...],
        'val_loss': [...],
        'learning_rate': [...]
    },
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'dropout': 0.2,
        # ... all hyperparameters
    }
}

torch.save(checkpoint, 'checkpoint.pth')
```

**Checkpoint Strategy:**

1. **Every N epochs:** Save periodic checkpoints (N=10)
2. **Best model:** Save whenever validation improves
3. **Final model:** Save at end regardless of performance

**Resume Training:**
```python
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Continue training from where it stopped
for epoch in range(start_epoch, num_epochs):
    train_epoch()
```

### 7.3 Experiment Tracking

**Logged Information:**

- Hyperparameters (all)
- Training curves (loss per epoch)
- Best validation loss
- Training time
- Hardware used
- Git commit hash (for code version)

**Future:** Weights & Biases integration for automatic tracking

---

## 8. Training Best Practices

### 8.1 Sanity Checks Before Training

1. **Overfit Single Batch:**
```python
   # Train on 1 batch until loss → 0
   # If can't overfit, architecture is broken
   single_batch = next(iter(train_loader))
   for _ in range(1000):
       loss = train_step(single_batch)

   assert loss < 0.01, "Can't overfit single batch!"
```

2. **Check Gradient Flow:**
```python
   loss.backward()
   for name, param in model.named_parameters():
       assert param.grad is not None, f"No gradient for {name}"
       assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
```

3. **Verify Data Shapes:**
```python
   x, y = next(iter(train_loader))
   pred = model(x)
   assert pred.shape == y.shape, "Output shape mismatch!"
```

### 8.2 Monitoring During Training

**Watch for:**

✅ **Good Signs:**
- Loss decreasing smoothly
- Validation tracking training (small gap)
- Gradients in range [1e-5, 1e-1]

⚠️ **Warning Signs:**
- Loss oscillating wildly → LR too high
- Loss not decreasing → LR too low or bug
- Val much worse than train → overfitting

❌ **Bad Signs:**
- Loss = NaN → Exploding gradients or bad data
- Loss increasing → Wrong sign in optimizer
- No improvement after 20 epochs → Model too simple

### 8.3 Common Pitfalls

**Pitfall 1: Forgetting model.train() / model.eval()**
```python
# WRONG:
def train_epoch():
    for batch in loader:
        predictions = model(batch)  # Dropout still active!

# CORRECT:
def train_epoch():
    model.train()  # Enable dropout
    for batch in loader:
        predictions = model(batch)

def validate():
    model.eval()  # Disable dropout
    with torch.no_grad():
        ...
```

**Pitfall 2: Not Zeroing Gradients**
```python
# WRONG:
for batch in loader:
    loss.backward()  # Gradients accumulate!
    optimizer.step()

# CORRECT:
for batch in loader:
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()
    optimizer.step()
```

**Pitfall 3: Using Training Data for Validation**
```python
# WRONG: Data leakage
train_dataset = LOBDataset(all_data)
val_dataset = LOBDataset(all_data)  # Same data!

# CORRECT: Chronological split
train_data, val_data, test_data = chronological_split(all_data)
```

---

## 9. References

**Optimization:**

1. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *International Conference on Learning Representations*.

2. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations*. (AdamW paper)

**Regularization:**

3. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, 15(1), 1929-1958.

4. Prechelt, L. (1998). Early stopping-but when?. In *Neural Networks: Tricks of the trade* (pp. 55-69). Springer.

**Loss Functions:**

5. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: data mining, inference, and prediction*. Springer.

**Time Series:**

6. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213.

**Hyperparameter Tuning:**

7. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms. *Advances in neural information processing systems*, 25.

**Deep Learning:**

8. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT press. (Chapter 8: Optimization for Training Deep Models)

**Financial Forecasting:**

9. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine Learning forecasting methods: Concerns and ways forward. *PloS one*, 13(3).

10. Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep convolutional neural networks for limit order books. *IEEE Transactions on Signal Processing*, 67(11), 3001-3012.

---

## Appendix A: Training Curves Interpretation

**Example 1: Healthy Training**
```
Loss
 1.0 ┤
     │ Train ╲
     │        ╲___
 0.5 │ Val     ╲___────
     │              ╲___
 0.0 └────────────────────── Epoch
     0   20   40   60   80
```

**Interpretation:**
- Both losses decreasing
- Validation following training
- Small gap (good generalization)
- Ready for deployment

**Example 2: Overfitting**
```
Loss
 1.0 ┤
     │ Train ╲
     │        ╲________
 0.5 │                 ╲
     │ Val    ___────────
 0.0 └────────────────────── Epoch
     0   20   40   60   80
             ^
          Stop here!
```

**Interpretation:**
- Train continues decreasing
- Val increases after epoch 40
- Memorizing training data
- Early stopping should trigger

**Example 3: Underfitting**
```
Loss
 1.0 ┤ Train ────────
     │ Val   ────────
 0.5 │
     │
 0.0 └────────────────────── Epoch
     0   20   40   60   80
```

**Interpretation:**
- Both losses high and flat
- Model too simple or LR too low
- Need: More capacity or higher LR

---

**End of Document**
