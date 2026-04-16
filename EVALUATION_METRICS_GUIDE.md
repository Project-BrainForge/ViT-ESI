# ViT-ESI Evaluation Metrics Guide

This document provides a comprehensive overview of the evaluation metrics and their calculation methods used in the ViT-ESI codebase for assessing Electrical Source Imaging (ESI) models.

---

## Table of Contents
1. [Overview](#overview)
2. [Evaluation Metrics](#evaluation-metrics)
3. [Evaluation Pipeline](#evaluation-pipeline)
4. [Key Concepts](#key-concepts)
5. [Implementation Details](#implementation-details)
6. [Output Files](#output-files)

---

## Overview

The evaluation framework in `model_training/eval.py` assesses both neural network models (1D-CNN, LSTM, DeepSIF, ViT) and linear inverse methods (MNE, sLORETA) on synthetic EEG data with known source information.

**Key evaluation steps:**
- Load a trained model or instantiate linear inverse methods
- For each validation sample:
  - Compute the forward model (EEG prediction)
  - Extract or compute the source estimate
  - Compute multiple spatial and temporal metrics
  - Compare against ground truth source distribution
- Aggregate metrics across the validation set
- Save results to CSV files

---

## Evaluation Metrics

### 1. **Normalized Mean Squared Error (nMSE)**

**Purpose:** Measures signal reconstruction accuracy of source estimates.

**Definition:**
- Defined in: `model_training/utils/utl_metrics.py` → `nmse_t_fn()` / `nmse_fn()`
- Computed at time instant of maximum activity (t_eval_gt)

**Calculation:**
```
nMSE = mean((x_normalized - x_hat_normalized)²)
```

Where:
- `x_normalized = x / max(|x|)` (ground truth source normalized by max amplitude)
- `x_hat_normalized = x_hat / max(|x_hat|)` (estimated source normalized by max amplitude)
- Both normalized to zero-mean amplitude for comparison

**In eval.py (line ~997):**
```python
nmse_tmp = (
    (
        j_unscaled[:, t_eval_gt] / j_unscaled[:, t_eval_gt].abs().max()
        - j_hat[:, t_eval_gt] / j_hat[:, t_eval_gt].abs().max()
    ) ** 2
).mean()
```

**Interpretation:**
- Lower values are better (metric measures error)
- Normalized to 0-1 range after amplitude normalization
- Averaged across multiple sources

---

### 2. **Localization Error (LOC_ERROR)**

**Purpose:** Measures spatial accuracy of source localization.

**Definition:**
- Measures Euclidean distance between true seed location and estimated seed location
- Unit: **millimeters (mm)** (scaled by 1000 for printing)
- Defined in: `model_training/eval.py` (line ~972-976)

**Calculation:**
```
For each true seed position s:
  1. Find time t_eval_gt = argmax(|j[s, :]|)  → time of maximum activity
  2. Define evaluation zone around s (order=2 patches)
  3. Find estimated seed: s_hat = argmax(|j_hat[eval_zone, t_eval_gt]|)
  4. Calculate: LE = sqrt(sum((spos[s] - spos[s_hat])²))

Aggregate over all seeds: final_LE = mean(LE per seed)
```

**In eval.py:**
```python
le += torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
# ...averaged over len(seeds) at end
```

**Where:**
- `spos`: 3D coordinates of source positions (loaded from head model)
- `eval_zone`: Patch of neighboring vertices around true seed
  - Uses `get_patch(order=2, idx=s, neighbors=neighbors)` 
  - Excludes other active sources

**Interpretation:**
- Lower values are better (measured in mm)
- Typical values: few mm for good localization
- Relevant only when true seed is known

---

### 3. **Time Error (TIME_ERROR)**

**Purpose:** Measures accuracy of temporal information (when source activates).

**Definition:**
- Measures temporal distance between true peak and estimated peak
- Unit: **time steps** (converted to **milliseconds (ms)** for printing at 1000x scale)

**Calculation:**
```
For each true seed s:
  1. t_eval_gt = argmax(|j[s, :]|)  → time of max activity in ground truth
  2. s_hat = estimated seed location
  3. t_eval_pred = argmax(|j_hat[s_hat, :]|)  → time of max activity in estimate
  4. TE = |t_vec[t_eval_gt] - t_vec[t_eval_pred]|

Aggregate: final_TE = mean(TE per seed)
```

**In eval.py:**
```python
t_eval_gt = torch.argmax(j[s, :].abs())
t_eval_pred = torch.argmax(j_hat[s_hat, :].abs())
te += np.abs(t_vec[t_eval_gt] - t_vec[t_eval_pred])
```

**Interpretation:**
- Lower values are better (measured in time units)
- Indicates whether the model captures the temporal dynamics correctly
- Often very small (within 1-2 time steps)

---

### 4. **Area Under the ROC Curve (AUC)**

**Purpose:** Measures ability to distinguish active vs. inactive sources (binary classification task).

**Definition:**
- Evaluates source localization as a classification problem
- Active sources (in a patch) = class 1, other sources = class 0
- Defined in: `model_training/utils/utl_metrics.py` → `auc_t()`

**Calculation:**
```
At time t_eval_gt:
  1. Create binary ground truth: bin_gt[active_sources] = 1, else 0
  2. Normalize estimated signal: x_hat_unit = |x_hat| / max(|x_hat|)
  3. Compute ROC curve from (bin_gt, x_hat_unit)
  4. Calculate AUC = area_under_curve(ROC)

Aggregate: final_AUC = mean(AUC per seed)
```

**In utl_metrics.py:**
```python
def auc_t(x_gt, x_hat, t, thresh=False, act_thresh=0.1, act_src=None):
    # Get active sources (thresh=True means threshold-based, thresh=False means list)
    if thresh:
        Sa = torch.argwhere(x_gt.abs() > act_thresh * x_gt.abs().max())
    else:
        Sa = act_src
    
    bin_gt = torch.zeros_like(x_gt)
    bin_gt[Sa] = 1
    
    x_hat_unit = x_hat.abs() / x_hat.abs().max()
    fpr, tpr, _ = roc_curve(bin_gt, x_hat_unit)
    auc_value = auc(fpr, tpr)
    return auc_value
```

**In eval.py:**
```python
auc_val += met.auc_t(j_unscaled, j_hat, t_eval_gt, thresh=True, act_thresh=0.0)
```

**Interpretation:**
- AUC ranges from 0 to 1, where:
  - 1.0 = perfect discrimination between active/inactive sources
  - 0.5 = random classification
  - 0 = inverted classification
- Higher values are better
- Indicates whether the estimated sources reliably show activity in correct locations

---

### 5. **Peak Signal-to-Noise Ratio (PSNR)**

**Purpose:** Measures overall signal quality of estimated source distribution.

**Definition:**
- Computed using scikit-image: `skimage.metrics.peak_signal_noise_ratio`
- Measures reconstruction quality of entire source distribution at all time points

**Calculation:**
```
PSNR = 20 * log10(max_val / MSE)

where:
  max_val = max_val (in dB scale)
  MSE = mean((x_normalized - x_hat_normalized)²)
```

**In eval.py (line ~1008):**
```python
psnr_dict[method][c] = psnr(
    (j_unscaled / j_unscaled.abs().max()).numpy(),
    (j_hat / j_hat.abs().max()).numpy(),
    data_range=(
        (j_unscaled / j_unscaled.abs().max()).min()
        - (j_hat / j_hat.abs().max()).max()
    ),
)
```

**Interpretation:**
- Higher values are better (measured in dB)
- Typical PSNR values for good reconstruction: > 20 dB
- Indicates overall signal recovery quality
- More sensitive to global amplitude differences than localization

---

## Evaluation Pipeline

### High-Level Flow

```
1. Load Head Model & Data
   ├─ Load leadfield matrix (forward model)
   ├─ Load source space mesh and neighbors
   └─ Load validation dataset

2. For each validation sample i:
   ├─ Load EEG data (M) and source data (j)
   ├─ Load metadata (seeds, active regions, patches)
   │
   ├─ For each method (CNN, LSTM, DeepSIF, ViT, MNE, sLORETA):
   │  ├─ Compute/load source estimate j_hat
   │  ├─ For each seed in current sample:
   │  │  ├─ Find ground truth peak: t_eval_gt
   │  │  ├─ Find estimated peak location and time
   │  │  ├─ Calculate:
   │  │  │  ├─ Localization Error (LE)
   │  │  │  ├─ Time Error (TE)
   │  │  │  ├─ nMSE
   │  │  │  └─ AUC
   │  │  └─ Accumulate metrics
   │  └─ Calculate PSNR for full source distribution
   │
   └─ Store {LE, TE, nMSE, AUC, PSNR} for this sample+method

3. Aggregate Results
   ├─ For each method, compute mean and std over all samples
   ├─ Filter out noise-only samples (if any)
   └─ Generate summary statistics

4. Save Results to CSV
   ├─ metrics summary (mean ± std for each metric)
   └─ full distribution (all per-sample values for plotting)
```

### Key Data Structures

**Metric dictionaries (initialized at line ~770):**
```python
nmse_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
loc_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
time_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
auc_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
psnr_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
```

Each entry stores values across the validation set for later aggregation and analysis.

---

## Key Concepts

### 1. **Seeds and Patches**

**What is a seed?**
- A seed is the primary vertex/source region that is actively generating the signal
- In SEREEGA simulations: single seed per source
- In NMM simulations: one seed per active region

**Metadata Storage:**
```python
# SEREEGA format
md_dict[sample_id]["seeds"] = 42  # single seed or list of seeds

# NMM format
dataset_meta["selected_region"][sample_idx][:, 0]  # array of seed indices
```

**What is a patch?**
- A patch is a spatially contiguous region around a seed
- Defined by recursively finding neighboring vertices
- Used to define "active region" for evaluation

**Patch Generation:**
```python
def get_patch(order, idx, neighbors):
    """
    order: number of iterations to expand
    idx: central seed vertex index
    neighbors: adjacency information
    returns: array of vertex indices in the patch
    """
    # Iteratively expand from seed by finding neighbor vertices
```

**In evaluation:**
```python
# For NMM: use order=3 patch
patches[kk] = utl.get_patch(order=3, idx=seeds[kk], neighbors=neighbors)

# For SEREEGA: read patches from metadata
patches[kk] = val_ds.dataset.md_dict[md_keys[k]]["act_src"][f"patch_{kk+1}"]
```

### 2. **Evaluation Zone**

When finding the estimated seed, the algorithm uses an evaluation zone to prevent false positives:

```python
# Start with order=5 patch (larger search area)
eval_zone = utl.get_patch(order=5, idx=s, neighbors=neighbors)

# Remove other active sources
eval_zone = np.setdiff1d(eval_zone, other_sources)

# Further restrict to order=2 patch for final search
eval_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)

# Find seed with max activity at time t_eval_gt
s_hat = eval_zone[torch.argmax(j_hat[eval_zone, t_eval_gt].abs())]
```

This prevents the algorithm from picking sources in completely different brain regions.

### 3. **Overlap Handling**

When two patches overlap (for multi-source stimulation), the code handles it:

```python
if len(patches) >= 2:
    inter = list(set(patches[0]).intersection(patches[1]))
    if len(inter) > 0:
        overlapping_regions += 1
        # Keep the source with higher amplitude
        to_keep = torch.argmax(torch.Tensor([
            j[seeds[0], :].abs().max(),
            j[seeds[1], :].abs().max()
        ]))
        seeds = [seeds[to_keep]]  # Only evaluate strongest source
```

This prevents double-counting metrics for ambiguous cases.

### 4. **Scaling and Normalization**

Different scaling approaches depending on loss function used during training:

**For MSE loss (amplitude scaling):**
```python
j_hat = j_hat * val_ds.dataset.max_src[k]  # Scale by dataset maximum
```

**For Cosine loss (GFP scaling):**
```python
j_hat = utl.gfp_scaling(M_unscaled, j_hat, torch.from_numpy(fwd))
# Scales based on relative power using Global Field Power
```

---

## Implementation Details

### 1. **File Organization**

| File | Role |
|------|------|
| `eval.py` | Main evaluation script; computes all metrics per sample |
| `utl_metrics.py` | Metric calculation functions (MSE, nMSE, AUC) |
| `utl.py` | Utility functions (patch generation, GFP scaling) |
| `utl_simu.py` | Simulation utilities including `get_neighbors()` |
| `loaders.py` | Dataset loaders; provides metadata structure |

### 2. **Data Flow in eval.py**

```python
# Lines 760-775: Initialize metric dictionaries
nmse_dict, loc_error_dict, time_error_dict, auc_dict, psnr_dict

# Lines ~830-870: Load validation data
val_ds = EsiDatasetds_new(...)  # Contains metadata

# Lines ~930-1000: Main evaluation loop
for method in methods:
    for each validation sample:
        # Compute source estimate j_hat
        # For each seed:
        #   - Calculate LE, TE, nMSE, AUC
        # Calculate PSNR
        # Store in dictionaries

# Lines ~1020-1145: Results aggregation and saving
# Compute mean/std
# Save to CSV files
```

### 3. **Handling Different Simulation Types**

**SEREEGA:**
```python
if args.eval_simu_type.lower() == "sereega":
    seeds = val_ds.dataset.md_dict[md_keys[k]]["seeds"]
    patches = metadata-based patches
```

**NMM:**
```python
else:  # NMM
    seeds = list(val_ds.dataset.dataset_meta["selected_region"][k][:, 0])
    patches = computed via get_patch()
```

---

## Output Files

### 1. **Summary Statistics CSV**
**File:** `evaluation_metrics_*.csv`

**Contains:**
```
simu_name, src_space, method, method_info, valset, noise db,
mean nmse, std nmse,
mean loc error, std loc error,
mean auc, std auc,
mean time error, std time error,
mean psnr, std psnr
```

**Example row:**
```
mes_debug, fsav_994, cnn_1d, cnn_model.pt, 100, 5,
0.234, 0.067,
2.543, 1.234,
0.876, 0.045,
0.012, 0.003,
18.567, 2.345
```

### 2. **Full Distribution CSV**
**File:** `evaluation_*.csv`

**Contains:** DataFrame with columns
```
nmse, loc error, auc, time error, psnr
(one row per validation sample)
```

**Used for:** Plotting distributions, histograms, and generating publication figures

### 3. **Filename Convention**
```
evaluation_metrics_train_simu_{TRAIN_TYPE}_eval_simu_{EVAL_TYPE}_method_{METHOD}_
srcspace_{SRCSPACE}_dataset{DATASET}_n_train_{N_TRAIN}{SUFFIX}.csv
```

---

## Metrics Summary Table

| Metric | Unit | Optimal | Range | Calculated_per | Comments |
|--------|------|---------|-------|----------------|----------|
| **nMSE** | unitless | Low | 0-1 | seed per timepoint | Signal reconstruction error |
| **Loc Error** | mm | Low | 0-∞ | seed | Spatial accuracy |
| **Time Error** | ms | Low | 0-∞ | seed | Temporal accuracy |
| **AUC** | unitless | High | 0-1 | seed | Source discrimination |
| **PSNR** | dB | High | 0-∞ | sample | Overall signal quality |

---

## Example Metric Values (Typical)

For a well-trained model on synthetic data with SNR=5dB:

```
Mean nMSE:        0.15 ± 0.08
Mean Loc Error:   3.2 ± 2.1 mm
Mean Time Error:  1.5 ± 0.8 ms
Mean AUC:         0.92 ± 0.05
Mean PSNR:        22.3 ± 3.2 dB
```

---

## Running Evaluation

### Basic Command
```bash
python eval.py mes_debug \
  -root_simu /path/to/project \
  -results_path /path/to/results \
  -eval_simu_type sereega \
  -source_space fsav_994 \
  -electrode_montage standard_1020 \
  -orientation constrained \
  -leadfield_mat /path/to/anatomy/leadfield_75_20k.mat \
  -methods cnn_1d lstm deep_sif eeg_vit MNE sLORETA
```

### Key Arguments
- `-methods`: Specify which methods to evaluate
- `-train_run_dir`: Path to folder with trained model weights
- `-eeg_snr`: SNR level for evaluation (default: 5)
- `-n_times`: Number of time samples (default: 500)

---

## See Also

- [model_training/readme.md](model_training/readme.md) - Training and evaluation guide
- [README.md](README.md) - Overall project structure
- [Data Generation Documentation](data_generation/readme.md) - How synthetic data is created

