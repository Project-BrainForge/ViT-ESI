# Real Data Evaluation - Technical Implementation Guide

## Architecture Overview

```
Real Patient Data Evaluation Pipeline
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  Input Files                                                           │
│  ├─ Real EEG Data (.mat)                                             │
│  ├─ Resection Regions (.mat)                                         │
│  └─ Leadfield Matrix (.mat)                                          │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Step 1: Load Real Data                                          │ │
│  │  load_patient_eeg() → M (75 × 500)                              │ │
│  │  load_resection_regions() → indices [26 vertices]               │ │
│  │  get_source_space_info() → G (75 × 994)                         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Step 2: Create Synthetic Ground Truth                           │ │
│  │  create_synthetic_ground_truth_from_resection()                │ │
│  │    - Initialize j_true (994 × 500) with zeros                  │ │
│  │    - Apply temporal profile (gaussian/peak/uniform)             │ │
│  │    - Set resection region vertices to active                    │ │
│  │    - All other vertices remain inactive                         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       │                                                                │
│       ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Step 3: Compute 5 Evaluation Metrics                            │ │
│  │                                                                  │ │
│  │  Metric 1: nMSE                                                 │ │
│  │    - Compare EEG at peak time                                   │ │
│  │    - Formula: mean((eeg_real_norm - eeg_pred_norm)²)           │ │
│  │                                                                  │ │
│  │  Metric 2: AUC                                                  │ │
│  │    - Binary classification ROC at peak time                     │ │
│  │    - Resection region (1) vs others (0)                        │ │
│  │                                                                  │ │
│  │  Metric 3: PSNR                                                 │ │
│  │    - Signal quality: 20*log10(max/sqrt(MSE))                   │ │
│  │    - Computed over entire EEG signal                            │ │
│  │                                                                  │ │
│  │  Metric 4: Spatial Correlation (with model prediction)         │ │
│  │    - Correlation between resection region activations           │ │
│  │    - Range: -1 to +1, optimal: high positive                   │ │
│  │                                                                  │ │
│  │  Metric 5: Time Error (with model prediction)                  │ │
│  │    - Peak time difference in milliseconds                       │ │
│  │    - Requires model source predictions j_estimated             │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│       │                                                                │
│       ▼                                                                │
│  Output Files                                                         │
│  ├─ .mat file (metrics in MATLAB format)                            │
│  ├─ .txt file (metrics in text format)                              │
│  └─ .png file (visualization comparison)                            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Code Flow Diagram

### eval_real_data.py Main Execution

```python
main()
  ├─ Parse command line arguments
  │  ├─ patient_id: Patient identifier
  │  ├─ eeg_data: EEG file path
  │  ├─ resection_file: Anatomy file path
  │  ├─ leadfield: Leadfield matrix path
  │  └─ ... (other options)
  │
  ├─ Load real EEG
  │  load_patient_eeg(eeg_path)
  │  └─ Returns: M (n_channels × n_times)
  │
  ├─ Load resection regions
  │  load_resection_regions(anatomy_path)
  │  └─ Returns: resection_indices (n_resection_vertices,)
  │
  ├─ Load leadfield and dimensions
  │  get_source_space_info(leadfield_path)
  │  └─ Returns: n_channels, n_sources, leadfield
  │
  ├─ Create synthetic ground truth
  │  create_synthetic_ground_truth_from_resection()
  │  └─ Returns: j_true (n_sources × n_times)
  │
  ├─ Load model predictions (if provided)
  │  loadmat(model_predictions_path)
  │  └─ Returns: j_estimated (n_sources × n_times)
  │
  ├─ Compute evaluation metrics
  │  evaluate_real_data()
  │  ├─ Compute peak time: t_peak = argmax(activity)
  │  ├─ Metric 1: nMSE at peak time
  │  ├─ Metric 2: AUC (binary classification)
  │  ├─ Metric 3: PSNR (full signal)
  │  ├─ Metric 4: Spatial Correlation (if j_estimated)
  │  └─ Metric 5: Time Error (if j_estimated)
  │
  ├─ Save results
  │  ├─ save_evaluation_results(.mat)
  │  ├─ save_evaluation_results(.txt)
  │  └─ (optional) visualize_resection_vs_estimate(.png)
  │
  └─ Print summary
     └─ Console output with all metrics
```

---

## Data Format Specifications

### Input Data Shapes

| Data | Variable | Shape | Dtype | Source |
|------|----------|-------|-------|--------|
| Real EEG | M | (75, 500) | float64 | `ictal/examples/P{i}/sz_data/data{j}.mat['data'].T` |
| Resection Indices | resection_indices | (n_resection,) | int32 | `projected_resection_to_fs_cortex.mat['resection_region']` |
| Leadfield | G | (75, 994) | float64 | `anatomy/leadfield_75_20k.mat['G']` |

### Synthetic Ground Truth Generation

```python
j_true = zeros((n_sources=994, n_times=500))

# For each vertex in resection region:
for idx in resection_indices:
    # Create temporal profile
    if temporal_profile == 'gaussian':
        t = linspace(-4, 4, 500)
        envelope = exp(-t²)
    elif temporal_profile == 'peak':
        envelope = gaussian_centered(500)
    elif temporal_profile == 'uniform':
        envelope = ones(500)
    
    # Assign with spatial variation
    spatial_factor = 0.8 + 0.4 * sin(i / len(resection_indices) * π)
    j_true[idx, :] = spatial_factor * envelope * resection_weight(0.8)
```

### Metric Computation

```
Peak Time Identification:
  activity = |j_true[resection_indices, :]|
  t_peak = argmax(activity.mean(axis=0))

Metric 1 - nMSE:
  eeg_pred = G @ j_true
  eeg_real_norm = eeg_real[:, t_peak] / max(|eeg_real|)
  eeg_pred_norm = eeg_pred[:, t_peak] / max(|eeg_pred|)
  nMSE = mean((eeg_real_norm - eeg_pred_norm)²)

Metric 2 - AUC:
  binary_gt = zeros(n_sources)
  binary_gt[resection_indices] = 1
  signal = |j_estimated[:, t_peak]| / max(|j_estimated|)
  fpr, tpr, _ = roc_curve(binary_gt, signal)
  AUC = auc(fpr, tpr)

Metric 3 - PSNR:
  MSE = mean((eeg_real_norm - eeg_pred_norm)²) over full signal
  PSNR = 20 * log10(max_val / sqrt(MSE))

Metric 4 - Spatial Correlation:
  true_resection = j_true[resection_indices, :].mean(axis=0)
  est_resection = j_estimated[resection_indices, :].mean(axis=0)
  correlation = corrcoef(normalize(true_resection), normalize(est_resection))

Metric 5 - Time Error:
  t_peak_est = argmax(|j_estimated[resection_indices, :]|)
  TE = |t_peak - t_peak_est| * dt (in ms)
```

---

## Key Implementation Details

### 1. Data Normalization

```python
def normalize_signal(signal):
    """Normalize to max absolute value = 1"""
    max_val = abs(signal).max()
    return signal / max_val if max_val > 0 else signal
```

**Why important**: Metrics are normalized to account for amplitude scaling differences between simulated and real data.

### 2. Temporal Profile Generation

```
Gaussian Profile (Default):
  t = linspace(-4, 4, n_times)
  envelope = exp(-t²)
  Result: Bell curve peaking at center

Peak Profile:
  envelope[n_times//2:n_times//2+10] = [1, 0.8, 0.6, ...]
  Result: Sharp seizure onset with decay

Uniform Profile:
  envelope = ones(n_times)
  Result: Constant activation throughout
```

### 3. Forward Model Application

```python
def apply_forward_model(sources, leadfield):
    """
    Compute EEG from sources using leadfield
    
    Physics: V = G @ J
    where:
      V = EEG voltage (channels × times)
      G = leadfield matrix (channels × sources)
      J = source current (sources × times)
    """
    return leadfield @ sources
```

### 4. AUC Computation

```python
# Classification problem:
# Class 0 = Non-resection vertices (inactive)
# Class 1 = Resection vertices (active)

# At peak time:
binary_gt = zeros(n_sources)
binary_gt[resection_indices] = 1

# Predicted probability:
signal = abs(j_estimated[:, t_peak])
signal_normalized = signal / max(signal)

# ROC-AUC:
fpr, tpr, _ = roc_curve(binary_gt, signal_normalized)
auc_value = auc(fpr, tpr)
```

---

## Extension Points

### 1. Custom Temporal Profiles

**Location**: `utils_real_eval.py`, function `create_synthetic_ground_truth_from_resection()`

```python
elif temporal_profile == 'custom':
    # Implement your custom profile here
    temporal_envelope = your_custom_function(n_times)
```

### 2. Multi-Source Seizures

**Current**: Assumes single main seizure focus

**Extension needed**:
```python
# Detect multiple peaks in resection region
peaks = find_peaks(activity)[0]
for peak in peaks:
    # Treat as separate source
    # Compute metric separately
```

### 3. Adaptive Temporal Profile

**Current**: Fixed templates (gaussian, peak, uniform)

**Extension needed**:
```python
# Extract actual temporal dynamics from EEG
def extract_temporal_profile(eeg, resection_indices):
    # Compute ICA or spectral decomposition
    # Extract dominant temporal mode
    return temporal_profile
```

### 4. Spatial Smoothing

**Current**: Slight amplitude variation within resection region via sine function

**Better approach**:
```python
# Apply Gaussian smoothing within and around resection
from scipy.ndimage import gaussian_filter
j_smooth = gaussian_filter(j_true, sigma=2.0)
```

---

## Error Handling

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| File not found | Wrong path | Check path exists: `Path(file).exists()` |
| Shape mismatch | Inconsistent dimensions | Verify loadmat output shape |
| NaN in metrics | Zero division | Add epsilon: `max_val = max(abs(signal).max(), 1e-10)` |
| Memory error | Large batch processing | Process seizures individually |
| Slow runtime | Computing AUC on large dataset | Subsample for testing |

### Validation Checks

```python
# After loading data
assert eeg_real.shape[0] == n_channels, "Channel mismatch"
assert eeg_real.shape[1] == n_times, "Time mismatch"
assert len(resection_indices) < n_sources, "Invalid resection size"
assert leadfield.shape == (n_channels, n_sources), "Leadfield shape error"
```

---

## Performance Optimization

### Current Performance
- Single seizure evaluation: ~0.5-1.0 seconds
- Batch evaluation (10 seizures): ~15-20 seconds

### Bottlenecks
1. loading .mat files: ~10% time
2. AUC computation: ~30% time
3. PSNR computation: ~15% time
4. Visualization (if enabled): ~30% time

### Optimization Opportunities

1. **Parallel Processing**:
   ```python
   from multiprocessing import Pool
   with Pool(n_workers) as p:
       results = p.map(evaluate_single_seizure, seizure_list)
   ```

2. **Vectorized Metrics**:
   - Replace loops with numpy operations
   - Use torch.cuda for GPU acceleration

3. **Caching**:
   - Cache leadfield matrix across seizures
   - Pre-compute forward model

---

## Testing & Validation

### Unit Tests (TODO)

```python
# test_utils_real_eval.py
def test_load_patient_eeg():
    eeg = load_patient_eeg("test_data.mat")
    assert eeg.shape == (75, 500)

def test_create_synthetic_ground_truth():
    j_true = create_synthetic_ground_truth_from_resection(...)
    assert j_true.shape == (994, 500)
    assert abs(j_true.max()) <= 1.0
```

### Integration Tests (TODO)

```python
# test_eval_real_data.py
def test_single_patient_evaluation():
    metrics = evaluate_real_data(...)
    assert 'nmse' in metrics
    assert 0 <= metrics['auc'] <= 1
    assert metrics['psnr'] > 0
```

---

## Comparison with Simulated Evaluation

| Aspect | Simulated | Real |
|--------|-----------|------|
| Ground Truth | Known exactly | Derived from resection |
| EEG Data | Synthetic (NMM/SEREEGA) | Real patient seizure |
| Source Truth | Source indices + amplitudes | Resection region indices only |
| Timing | Precise (simulation parameter) | Approximate (can vary) |
| Signal Quality | Controlled | Variable (noise, artifacts) |
| Metrics | Direct comparison | Proxy comparison |
| Use | Algorithm development | Clinical validation |

---

## References

### Files Organization
```
real_data_evaluation/
├── __init__.py                 # Package initialization
├── eval_real_data.py          # Main single-seizure evaluation
├── batch_eval.py              # Batch processing script  
├── utils_real_eval.py         # Utility functions
├── example_usage.py           # Example demonstrations
├── README.md                  # Full documentation
├── QUICKSTART.md              # Quick start guide
└── TECHNICAL_GUIDE.md         # This file
```

### Related Documents
- [EVALUATION_METRICS_GUIDE.md](../EVALUATION_METRICS_GUIDE.md) - Metric definitions
- [EVALUATION_ARCHITECTURE_DIAGRAMS.md](../EVALUATION_ARCHITECTURE_DIAGRAMS.md) - Process flows
- [model_training/eval.py](../model_training/eval.py) - Simulated data evaluation (reference)

---

## Future Work

1. **Adaptive Temporal Profiles**: Learn from EEG data
2. **Multi-Source Seizures**: Handle multiple seizure origins
3. **Uncertainty Quantification**: Confidence intervals on metrics
4. **Real-time Processing**: Stream seizure data evaluation
5. **Comparison Framework**: Statistical tests between methods
6. **Clinical Integration**: Links to surgical outcomes

---

