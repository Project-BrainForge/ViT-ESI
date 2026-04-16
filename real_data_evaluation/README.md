# Real Data Evaluation Framework

Evaluation of ESI models on real patient seizure data using resection regions as ground truth proxy.

## Overview

This framework adapts the 5-metric evaluation system used for simulated data to work with real patient EEG data. The key innovation is using **resection regions** (the clinically-determined seizure-generating zone) as a ground truth proxy to evaluate model predictions.

### Problem Statement
- **Simulated data**: Ground truth sources known exactly from simulation parameters
- **Real data**: Ground truth unknown, but resection regions indicate probable seizure origin
- **Solution**: Create synthetic ground truth from resection regions, use same 5 metrics

---

## Data Structure

### Input Files

```
ictal/examples/
├── P2/
│   ├── anatomy/
│   │   ├── cortex_fs_in_curryspace.mat
│   │   └── projected_resection_to_fs_cortex.mat    ← Resection region indices
│   └── sz_data/
│       ├── data1.mat                               ← Real EEG seizure 1
│       ├── data2.mat                               ← Real EEG seizure 2
│       └── data3.mat                               ← Real EEG seizure 3
├── P3/
│   └── ... (similar structure)
```

### File Contents

**`projected_resection_to_fs_cortex.mat`** fields:
- `resection_region`: Array of vertex indices (0 or 1-indexed) indicating resection zones
- `resection_region_before_sz_analysis`: Pre-analysis resection zone
- `projected_soz`: Seizure onset zone (2 vertices typically)

**`data{i}.mat`** fields:
- `data`: EEG array, shape `(n_times, n_channels)` 
  - Typical: (500, 75) for 500 ms at 500 Hz with 75 electrodes
  - Data range: typically −500 to +500 µV

---

## Methodology: Creating Synthetic Ground Truth

### Step 1: Load and Process Data

```
Real EEG (500 × 75)
    ↓
Resection indices from anatomy file
    ↓
Leadfield matrix (75 × 994)
```

### Step 2: Create Synthetic Source Distribution

**Key insight**: Seizure activity originates from resection region, so create ground truth where:

```python
j_true = zeros(n_sources=994, n_times=500)

# Assign activation to resection region vertices
for idx in resection_indices:
    # Apply temporal envelope to vertices
    j_true[idx, :] = amplitude * temporal_envelope(t)

# All other vertices remain zero (inactive)
```

### Step 3: Temporal Profile Options

Three options for temporal envelope shape:

**1. Gaussian (Default)** - Realistic physiological profile
```
Amplitude
    ^
    |     ___
    |    /   \
    |___/     \___
    _____________> Time
```
- Formula: `exp(-t²)` centered at middle of recording
- Best for: General seizure inference

**2. Peak** - Sharp seizure onset
```
Amplitude
    ^
    |     |
    |     |
    |_____|___
    _____________> Time
```
- Formula: Single spike at t_center with spreading
- Best for: Onset localization

**3. Uniform** - Constant activation
```
Amplitude
    ^
    |  -------
    |_________|___
    _____________> Time
```
- Formula: `ones(n_times)` 
- Best for: Sustained seizure activity

### Step 4: Apply Forward Model

```
Synthetic Ground Truth Sources
    j_true (994 × 500)
        ↓
    Leadfield @ j_true
        ↓
Predicted EEG
    eeg_predicted (75 × 500)
```

---

## The 5 Evaluation Metrics

All metrics compare real EEG against predicted EEG from synthetic ground truth or against model estimates.

### Metric 1: Normalized MSE (nMSE)
**What**: EEG reconstruction error at seizure peak
**Formula**: `mean((eeg_real_norm - eeg_pred_norm)²)` at t_peak
**Unit**: Unitless (0-1)
**Optimal**: Low (closer to 0)
**Interpretation**: How well does the model capture seizure-related EEG

### Metric 2: AUC (Area Under ROC Curve)
**What**: Binary classification of resection vs non-resection regions
**At peak time**:
- Class 1: Resection region vertices (active)
- Class 0: All other vertices (inactive)
**Formula**: Area under ROC curve of estimated source activations
**Unit**: 0-1
**Optimal**: High (closer to 1)
**Interpretation**: Discriminability of seizure-generating region

### Metric 3: Spatial Correlation
**What**: How well estimated sources correlate with ground truth in resection region
**Formula**: `pearson_correlation(j_true, j_estimated)` at peak time
**Unit**: -1 to +1
**Optimal**: High (closer to 1)
**Interpretation**: Spatial accuracy of localization

### Metric 4: Time Error (TE)
**What**: If model estimates available, time difference between peaks
**Available**: Only when comparing two estimates or when model predictions provided
**Unit**: Milliseconds (ms)
**Optimal**: Low (close to 0)
**Interpretation**: Temporal accuracy of peak detection

### Metric 5: PSNR (Peak Signal-to-Noise Ratio)
**What**: Overall quality of EEG reconstruction
**Formula**: `20 * log10(max_value / sqrt(MSE))`
**Unit**: dB (decibels)
**Optimal**: High (>20 dB is good)
**Interpretation**: Global EEG signal quality

---

## Usage

### Single Patient Evaluation

```bash
cd /path/to/ViT-ESI/real_data_evaluation

python eval_real_data.py P2 \
  -eeg_data ../ictal/examples/P2/sz_data/data1.mat \
  -resection_file ../ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat \
  -leadfield ../anatomy/leadfield_75_20k.mat \
  -results_path ./results \
  -model_name baseline \
  -temporal_profile gaussian \
  -visualize
```

**Arguments**:
- `patient_id`: Patient identifier (e.g., P2)
- `-eeg_data`: Path to real EEG file
- `-resection_file`: Path to resection anatomy file
- `-leadfield`: Path to leadfield matrix
- `-results_path`: Output directory
- `-model_name`: Name of model (optional, default: 'unknown')
- `-temporal_profile`: gaussian/peak/uniform (optional)
- `-model_predictions`: Path to model's source predictions (optional)
- `-visualize`: Generate comparison plots (flag)

### Batch Evaluation

Evaluate all patients and all seizures:

```bash
python batch_eval.py \
  -base_path ../ictal/examples \
  -leadfield ../anatomy/leadfield_75_20k.mat \
  -results_path ./results \
  -model_name vit_model \
  -temporal_profile gaussian \
  -patients P2 P3 \
  -visualize
```

---

## Output Files

### Results for Single Seizure

After running, you'll get:

**1. `.mat` file** (MATLAB-compatible)
```
eval_results_P2_baseline.mat
├── model: 'baseline'
├── nmse: 0.1523
├── auc: 0.8934
├── spatial_correlation: 0.7823
├── psnr: 22.34 (dB)
├── time_error: 1.5 (ms)
├── resection_size: 26
└── ... (other metrics, dimensions, metadata)
```

**2. `.txt` file** (Human-readable)
```
================================================================================
Real Data Evaluation Results - P2
================================================================================

model: baseline
t_peak: 247
nmse: 0.1523
loc_error: 0.2177
time_error: 1.5
auc: 0.8934
psnr: 22.34
spatial_correlation: 0.7823
eeg_reconstruction_error: 0.0342
resection_size: 26
n_sources: 994
n_channels: 75
n_times: 500
```

**3. Visualization** (if `-visualize` flag)
```
viz_P2_baseline_comparison.png
├── Ground truth activation distribution
├── Estimated activation distribution
├── Temporal activation profiles
└── Vertex-wise max activation comparison
```

### Batch Summary Report

```
batch_evaluation_summary.txt
```

---

## Practical Implementation Details

### Why This Approach Works

1. **Resection regions are clinically validated**: Neurosurgeons determine resection zones based on:
   - EEG seizure onset localization
   - Imaging abnormalities
   - Electrocorticography (if available)
   - Patient outcome tracking

2. **Creates realistic ground truth**: Seizures typically originate from resection region
   - Not a perfect ground truth (distributed sources possible)
   - But clinically meaningful ground truth

3. **Allows same metrics as simulated**: Using forward model and leadfield creates known ground truth

### Synthetic Ground Truth Quality

The synthetic ground truth is realistic because:

- **Spatial validity**: Uses actual resection region vertices
- **Temporal realism**: Gaussian/peak profiles match seizure onset dynamics
- **Anatomically grounded**: Based on individual patient anatomy

### Limitations to Consider

1. **Resection region may not be only active zone**: Seizures can recruit beyond resection region
2. **Temporal profile is assumed**: Real temporal dynamics unknown
3. **May overestimate model performance**: Since we're comparing against synthetic ground truth
4. **Potential for bias**: If model was trained on similar data

---

## File Organization

```
real_data_evaluation/
├── eval_real_data.py          ← Main single-seizure evaluation
├── batch_eval.py              ← Batch processing script
├── utils_real_eval.py         ← Utility functions
└── README.md                  ← This file

Expected results output:
results/
├── eval_results_P2_baseline.mat
├── eval_results_P2_baseline.txt
├── viz_P2_baseline_comparison.png
└── batch_evaluation_summary.txt
```

---

## Integration with Existing Evaluation Metrics

This framework uses the same metric functions as simulated data evaluation:

```python
# From model_training/utils/utl_metrics.py
import utl_metrics as met

met.auc_t(ground_truth, estimated, t=peak_time, thresh=True)
met.nmse_fn(normalized_signals)
```

So metrics are directly comparable between:
- Simulated data evaluation
- Real data evaluation
- Different models

---

## Examples

### Example 1: Evaluate CNN-1D on Patient P2, Seizure 1

```bash
python eval_real_data.py P2 \
  -eeg_data ../ictal/examples/P2/sz_data/data1.mat \
  -resection_file ../ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat \
  -leadfield ../anatomy/leadfield_75_20k.mat \
  -results_path ./results \
  -model_name cnn_1d \
  -model_predictions ./P2_seizure1_cnn_predictions.mat
```

### Example 2: Batch evaluation with Gaussian temporal profile

```bash
python batch_eval.py \
  -base_path ../ictal/examples \
  -leadfield ../anatomy/leadfield_75_20k.mat \
  -results_path ./batch_results \
  -model_name deep_sif \
  -temporal_profile gaussian \
  -visualize
```

---

## Troubleshooting

### Issue: "Anatomy file not found"
**Solution**: Check path to `projected_resection_to_fs_cortex.mat`

### Issue: "Leadfield dimension mismatch"
**Solution**: Ensure leadfield matches the number of sources in your head model (e.g., 994 for fsaverage)

### Issue: "NaN in metrics"
**Solution**: If model predictions not provided, some metrics will be NaN (e.g., time_error)

### Issue: "Temporal profile doesn't match seizure"
**Solution**: Try different profiles or implement custom temporal profile in `utils_real_eval.py`

---

## References

### Related Files
- [EVALUATION_METRICS_GUIDE.md](../EVALUATION_METRICS_GUIDE.md) - Detailed metrics documentation
- [EVALUATION_CODE_REFERENCE.md](../EVALUATION_CODE_REFERENCE.md) - Code snippets and references
- [model_training/eval.py](../model_training/eval.py) - Simulated data evaluation (comparison)

### Publications
- Sun et al. (2024): Seizure source imaging with deep neural networks
- Original DeepSIF paper: Constrained neural networks for source imaging

---

## Future Enhancements

Potential improvements to this framework:

1. **Adaptive temporal profiles**: Extract temporal profile from seizure EEG
2. **Multi-source seizures**: Handle cases with multiple seizure origins
3. **Spatio-temporal correlation**: More sophisticated spatial matching
4. **Uncertainty quantification**: Confidence intervals on metrics
5. **Cross-validation**: Train/test on different seizures
6. **Longitudinal analysis**: Track metrics across seizure progression

---

## Questions & Support

For issues with the evaluation framework:
1. Check that all input files exist and have correct paths
2. Verify leadfield dimensions match source space
3. Ensure real EEG data is properly shaped (time × channels)
4. Check MATLAB .mat file variable names match expected format

---
