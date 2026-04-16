# ViT-ESI Evaluation - Code Reference Guide

This document provides key code snippets and line references for understanding the evaluation implementation.

---

## Quick Navigation

| Concept | File | Lines | Related Functions |
|---------|------|-------|-------------------|
| **Metric Functions** | `utl_metrics.py` | 1-200 | `mse_fn()`, `nmse_fn()`, `auc_t()` |
| **Patch Generation** | `utl.py` | 88-104 | `get_patch()` |
| **Neighbor Computation** | `utl.py` | 120-140 | `get_neighbors()` |
| **Main Evaluation** | `eval.py` | 750-1000 | Metric calculation loop |
| **Results Aggregation** | `eval.py` | 1020-1145 | CSV generation |
| **Dataset Loading** | `loaders.py` | 1-100 | `EsiDatasetds_new` class |

---

## Core Metric Functions

### 1. MSE / nMSE Functions

**File:** `model_training/utils/utl_metrics.py`

```python
# Lines 15-21: Basic MSE
def mse_fn(x, x_hat):
    """Mean Squared Error (MSE) between x and x_hat"""
    assert x.shape == x_hat.shape
    mse_val = ((x - x_hat) ** 2).mean()
    return mse_val

# Lines 23-27: Batch MSE
def batch_mse_fn(x, x_hat):
    """MSE on batch x and x_hat"""
    return mse_fn(x, x_hat)

# Lines 29-55: nMSE at specific time point
def nmse_t_fn(x, x_hat, t):
    """
    Normalized MSE between x and x_hat at time t.
    Normalizes each to max amplitude, then computes MSE.
    """
    assert x.shape == x_hat.shape
    assert len(x.shape) == 2
    
    x = x[:, t].squeeze()
    x_hat = x_hat[:, t].squeeze()
    
    # Normalize to max amplitude
    if x.max() == 0. or x_hat.max() == 0.:
        x_n = x
        x_hat_n = x_hat
    else:
        x_n = x / x.abs().max()
        x_hat_n = x_hat / x_hat.abs().max()
    
    nmse_t_val = ((x_n - x_hat_n) ** 2).mean()
    return nmse_t_val

# Lines 57-81: Batch nMSE
def batch_nmse_fn(x, x_hat):
    """Normalized MSE for 3D batch tensor (B, space, time)"""
    assert x.shape == x_hat.shape
    assert len(x.shape) == 3
    
    max_scaler = x.view(x.shape[0], -1).max(dim=1)[0]
    nul_id = torch.argwhere(max_scaler == 0.).squeeze()
    max_scaler[nul_id] = 1
    x_n = x / max_scaler.view(x.shape[0], 1, 1)
    
    max_scaler = x_hat.view(x_hat.shape[0], -1).max(dim=1)[0]
    nul_id = torch.argwhere(max_scaler == 0.).squeeze()
    max_scaler[nul_id] = 1
    x_hat_n = x_hat / max_scaler.view(x.shape[0], 1, 1)
    
    return mse_fn(x_n, x_hat_n)
```

### 2. AUC Function

**File:** `model_training/utils/utl_metrics.py`

```python
# Lines 103-145: AUC calculation
def auc_t(x_gt, x_hat, t, thresh=False, act_thresh=0.1, act_src=None):
    """
    AUC at time t using ROC analysis.
    
    Parameters:
    - x_gt: ground truth source distribution
    - x_hat: estimated source data
    - t: time instant to study
    - thresh: threshold-based (True) or list-based (False) active source detection
    - act_thresh: threshold percentage (% of max amplitude) for active sources
    - act_src: explicit list of active source indices if thresh=False
    """
    x_gt = x_gt.squeeze()
    x_hat = x_hat.squeeze()
    assert len(x_gt.shape) == 2
    assert len(x_gt.shape) == len(x_hat.shape)
    
    x_gt = x_gt[:, t]
    x_hat = x_hat[:, t]
    
    # Get active sources
    if thresh:
        Sa = torch.argwhere(x_gt.abs() > act_thresh * x_gt.abs().max())
    else:
        Sa = act_src
    
    # Binarize: active=1, inactive=0
    bin_gt = torch.zeros_like(x_gt, dtype=int)
    bin_gt[Sa] = 1
    
    # Normalize estimated signal
    if x_hat.abs().max() == 0.:
        x_hat_unit = x_hat.abs()
    else:
        x_hat_unit = x_hat.abs() / x_hat.abs().max()
    
    # Compute ROC and AUC
    fpr, tpr, _ = roc_curve(bin_gt, x_hat_unit)
    auc_value = auc(fpr, tpr)
    
    return auc_value
```

**Usage in eval.py (line ~985):**
```python
auc_val += met.auc_t(j_unscaled, j_hat, t_eval_gt, thresh=True, act_thresh=0.0)
```

---

## Patch & Neighbor Functions

### 1. Get Patch Function

**File:** `model_training/utils/utl.py`

```python
# Lines 88-104: Generate patch around seed
def get_patch(order, idx, neighbors):
    """
    Generate a spatially contiguous patch around a seed vertex.
    
    Parameters:
    - order: number of expansion iterations (0=seed only, 1=seed+neighbors, etc)
    - idx: central seed vertex index
    - neighbors: adjacency matrix (each row contains neighbors, padded with 0)
    
    Returns:
    - unique array of vertex indices in the patch
    """
    new_idx = np.array([idx], dtype=np.int64)
    
    if order == 0:
        return new_idx
    else:
        # Iteratively expand: for each order, find one ring of neighbors
        for _ in range(order):
            neighb = np.unique(neighbors[new_idx, :])
            neighb = neighb[neighb > 0].astype(np.int64)
            new_idx = np.append(new_idx, neighb)
        
        return np.unique(new_idx)

# Example usage in eval.py:
# Order-3 patch for NMM evaluation zone
patches[kk] = utl.get_patch(order=3, idx=seeds[kk], neighbors=neighbors)

# Order-2 patch for final search zone
eval_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)
```

### 2. Get Neighbors Function

**File:** `model_training/utils/utl.py`

```python
# Lines 120-140: Compute neighbor relationships from triangulation
def get_neighbors(tris, verts):
    """
    Compute neighbor relationships from surface triangulation.
    
    Parameters:
    - tris: list of 2 arrays, triangulation for each hemisphere
    - verts: list of 2 arrays, vertex indices for each hemisphere
    
    Returns:
    - neighb_array: (n_vertices, max_neighbors) array of neighbor indices,
                    padded with 0
    """
    n_verts = len(verts[0]) + len(verts[1])
    neighbors = [list() for _ in range(n_verts)]
    
    for hem in range(2):  # Left and right hemisphere
        i = 0
        idx_tris_old = np.sort(np.unique(tris[hem])).astype(np.int64)
        idx_vert_old = np.sort(np.unique(verts[hem])).astype(np.int64)
        
        missing_verts = np.setdiff1d(idx_tris_old, idx_vert_old)
        idx_vert_new = np.arange(0, len(idx_vert_old))
        
        vertices_lin = np.zeros((idx_vert_old.max() + 1, 1))
        vertices_lin[idx_vert_old, 0] = idx_vert_new
        vertices_lin = vertices_lin.astype(np.int64)
        
        for v in verts[hem]:
            triangles_of_v = np.squeeze(tris[hem] == v)
            triangles_of_v = np.squeeze(tris[hem][np.sum(triangles_of_v, axis=1) > 0])
            
            neighbors_of_v = np.unique(triangles_of_v)
            neighbors_of_v = neighbors_of_v[neighbors_of_v != v]
            neighbors_of_v = np.setdiff1d(neighbors_of_v, missing_verts)
            
            neighbors[i] = list(vertices_lin[neighbors_of_v, 0])
            i += 1
    
    # Convert to array, padding with 0
    l_max = np.amax(np.array([len(l) for l in neighbors]))
    neighb_array = np.zeros((len(neighbors), l_max))
    for i in range(len(neighbors)):
        l = neighbors[i]
        neighb_array[i, :len(l)] = l
        if len(l) < l_max:
            neighb_array[i, len(l):] = None
    
    return neighb_array.astype(np.int64)
```

### 3. GFP Scaling Function

**File:** `model_training/utils/utl.py`

```python
# Lines 76-84: Global Field Power scaling
def gfp_scaling(M, j_pred, G):
    """
    Scale estimated sources based on Global Field Power (GFP).
    
    Parameters:
    - M: ground truth EEG data (channels × time)
    - j_pred: estimated source distribution, unscaled (sources × time)
    - G: leadfield matrix (channels × sources)
    
    Returns:
    - j_pred_scaled: scaled source estimate
    """
    j_pred_scaled = torch.zeros_like(j_pred)
    M_pred = G @ j_pred  # Predicted EEG from estimated sources
    
    # Time instant by time instant scaling
    for t in range(j_pred.shape[1]):
        if torch.std(M_pred[:, t]) == 0:
            denom = 1
        else:
            denom = torch.std(M_pred[:, t])
        
        # Scale by ratio of standard deviations
        j_pred_scaled[:, t] = j_pred[:, t] * (torch.std(M[:, t]) / denom)
    
    return j_pred_scaled

# Used in eval.py for models trained with cosine loss:
j_hat = utl.gfp_scaling(M_unscaled, j_hat, torch.from_numpy(fwd))
```

---

## Main Evaluation Loop

### Initialization

**File:** `model_training/eval.py` (lines ~750-775)

```python
# 770-774: Initialize metric dictionaries
nmse_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
loc_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
time_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
auc_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
psnr_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}

# 765: Counter
c = 0
noise_only_eeg_data = []
overlapping_regions = 0
```

### Main Loop - Sample Iteration

**File:** `model_training/eval.py` (lines ~780-830)

```python
# for each validation sample
for batch_idx, val_batch in enumerate(val_loader):
    c += 1  # Sample counter
    
    # Unpack batch
    M, j_unscaled = val_batch[0], val_batch[1]
    M = M.to(device)
    
    # Get sample metadata key
    k = batch_idx
    md_keys = list(val_ds.dataset.md_dict.keys())
    
    # Skip if noise-only
    if (j_unscaled.abs().sum() == 0):
        noise_only_eeg_data.append(batch_idx)
        continue
```

### Method Computation

**File:** `model_training/eval.py` (lines ~860-930)

```python
# For each evaluation method
for method in methods:
    if method == "gt":
        j_hat = j_unscaled  # Ground truth
    
    elif method in linear_methods:
        # Use MNE inverse
        lambda2 = 1.0 / (args.eeg_snr ** 2)
        inv_op = mne.minimum_norm.make_inverse_operator(
            info=eeg.info,
            forward=fwd_regions,
            noise_cov=noise_cov,
            loose=0,
            depth=0,
            verbose=False,
        )
        stc_hat = mne.minimum_norm.apply_inverse_raw(
            raw=eeg,
            inverse_operator=inv_op,
            lambda2=lambda2,
            method=method,
            verbose=False,
        )
        j_hat = torch.from_numpy(stc_hat.data)
    
    elif method == "cnn_1d":
        with torch.no_grad():
            j_hat = cnn.model(M.unsqueeze(0)).squeeze()
        if cnn1d_params["loss"] == "cosine":
            j_hat = utl.gfp_scaling(M_unscaled, j_hat, torch.from_numpy(fwd))
        else:
            j_hat = j_hat * val_ds.dataset.max_src[k]
    
    elif method == "lstm":
        with torch.no_grad():
            j_hat = lstm(M.unsqueeze(0)).squeeze()
        if lstm_params["loss"] == "cosine":
            j_hat = utl.gfp_scaling(M_unscaled, j_hat, torch.from_numpy(fwd))
        else:
            j_hat = j_hat * val_ds.dataset.max_src[k]
    
    elif method == "deep_sif":
        with torch.no_grad():
            j_hat = deep_sif(M.unsqueeze(0)).squeeze()
        if deep_sif_params["loss"] == "cosine":
            j_hat = utl.gfp_scaling(M_unscaled, j_hat, torch.from_numpy(fwd))
        else:
            j_hat = j_hat * val_ds.dataset.max_src[k]
    
    elif method == "eeg_vit":
        with torch.no_grad():
            j_hat = eeg_vit(M.unsqueeze(0)).squeeze()
        if vit_params.get("loss", args.train_loss) == "cosine":
            j_hat = utl.gfp_scaling(M_unscaled, j_hat, torch.from_numpy(fwd))
        else:
            j_hat = j_hat * val_ds.dataset.max_src[k]
```

### Metrics Calculation per Seed

**File:** `model_training/eval.py` (lines ~930-1005)

```python
# Get seeds and patches based on simulation type
if args.eval_simu_type.lower() == "sereega":
    seeds = val_ds.dataset.md_dict[md_keys[k]]["seeds"]
    if type(seeds) is int:
        seeds = [seeds]
else:  # NMM
    seeds = list(val_ds.dataset.dataset_meta["selected_region"][k][:, 0])
    seeds = [s.astype(int) for s in seeds]
    if type(seeds) is int:
        seeds = [seeds]

# Get patches
patches = [[] for _ in range(len(seeds))]
if args.eval_simu_type.lower() == "nmm":
    for kk in range(len(seeds)):
        patches[kk] = utl.get_patch(order=3, idx=seeds[kk], neighbors=neighbors)
else:
    for kk in range(len(seeds)):
        patches[kk] = val_ds.dataset.md_dict[md_keys[k]]["act_src"][f"patch_{kk+1}"]

# Handle overlapping patches
if len(patches) >= 2:
    inter = list(set(patches[0]).intersection(patches[1]))
    if len(inter) > 0:
        overlapping_regions += 1
        to_keep = torch.argmax(torch.Tensor([
            j[seeds[0], :].abs().max(),
            j[seeds[1], :].abs().max()
        ]))
        seeds = [seeds[to_keep]]
        patches = [patches[to_keep]]

act_src = [s for l in patches for s in l]

# Calculate metrics for each seed
le = 0
te = 0
nmse = 0
auc_val = 0
seeds_hat = []

for kk in range(len(seeds)):
    s = seeds[kk]
    other_sources = np.setdiff1d(act_src, patches[kk])
    
    # Find ground truth peak
    t_eval_gt = torch.argmax(j[s, :].abs())
    
    # Define evaluation zone (order=2 patch around seed)
    eval_zone = utl.get_patch(order=5, idx=s, neighbors=neighbors)
    eval_zone = np.setdiff1d(eval_zone, other_sources)
    eval_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)
    
    # Find estimated seed
    s_hat = eval_zone[torch.argmax(j_hat[eval_zone, t_eval_gt].abs())]
    t_eval_pred = torch.argmax(j_hat[s_hat, :].abs())
    
    # Calculate Localization Error
    le += torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
    
    # Calculate Time Error
    te += np.abs(t_vec[t_eval_gt] - t_vec[t_eval_pred])
    
    # Calculate AUC
    auc_val += met.auc_t(j_unscaled, j_hat, t_eval_gt, thresh=True, act_thresh=0.0)
    
    # Calculate nMSE at peak time
    nmse_tmp = (
        (
            j_unscaled[:, t_eval_gt] / j_unscaled[:, t_eval_gt].abs().max()
            - j_hat[:, t_eval_gt] / j_hat[:, t_eval_gt].abs().max()
        ) ** 2
    ).mean()
    nmse += nmse_tmp
    
    seeds_hat.append(s_hat)

# Average over all seeds
le = le / len(seeds)
te = te / len(seeds)
nmse = nmse / len(seeds)
auc_val = auc_val / len(seeds)

# Store metrics
time_error_dict[method][c] = te
loc_error_dict[method][c] = le
nmse_dict[method][c] = nmse
auc_dict[method][c] = auc_val

# Calculate PSNR over entire signal
psnr_dict[method][c] = psnr(
    (j_unscaled / j_unscaled.abs().max()).numpy(),
    (j_hat / j_hat.abs().max()).numpy(),
    data_range=(
        (j_unscaled / j_unscaled.abs().max()).min()
        - (j_hat / j_hat.abs().max()).max()
    ),
)
```

---

## Results Aggregation & Saving

### Summary Statistics

**File:** `model_training/eval.py` (lines ~1020-1090)

```python
# Print results
for method in methods:
    print(f" >>>>>>>>>>>>>>> Results method {method} <<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"mean time error: {time_error_dict[method].mean()*1e3} [ms]")
    print(f"mean localisation error: {loc_error_dict[method].mean()*1e3} [mm]")
    print(f"mean nmse at instant of max activity: {nmse_dict[method].mean():.4f}")
    print(f"psnr for total source distrib: {psnr_dict[method].mean():.4f} [dB]")
    print(f"auc: {auc_dict[method].mean():.4f}")

# Save to CSV - Summary metrics
for method in methods:
    my_values = [{
        "simu_name": args.simu_name,
        "src_space": head_model.source_space.src_sampling,
        "method": method,
        "method_info": method_info,
        "valset": str(n_val_samples),
        "noise db": f"{args.eeg_snr}",
        "mean nmse": f"{nmse_dict[method].mean()}",
        "std nmse": f"{nmse_dict[method].std()}",
        "mean loc error": f"{loc_error_dict[method].mean()}",
        "std loc error": f"{loc_error_dict[method].std()}",
        "mean auc": f"{auc_dict[method].mean()}",
        "std auc": f"{auc_dict[method].std()}",
        "mean time error": f"{time_error_dict[method].mean()}",
        "std time error": f"{time_error_dict[method].std()}",
        "mean psnr": f"{psnr_dict[method].mean()}",
        "std psnr": f"{psnr_dict[method].std()}",
    }]
    
    with open(
        f"{eval_results_path}/evaluation_metrics_{suffix_save_metrics}.csv",
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(my_values)
```

### Full Distribution

**File:** `model_training/eval.py` (lines ~1110-1145)

```python
# Save full distribution for plotting
for method in methods:
    my_values = {
        "nmse": np.squeeze(nmse_dict[method]),
        "loc error": np.squeeze(loc_error_dict[method]),
        "auc": np.squeeze(auc_dict[method]),
        "time error": np.squeeze(time_error_dict[method]),
        "psnr": np.squeeze(psnr_dict[method]),
    }
    
    df = pd.DataFrame(data=my_values)
    
    df.to_csv(
        f"{eval_results_path}/evaluation_{suffix_save_metrics}.csv"
    )
    
    print(f">>>>>>> results saved in :{eval_results_path}/evaluation_{suffix_save_metrics}.csv")
```

---

## Data Structures

### Validation Dataset Metadata

**File:** `model_training/loaders.py`

```python
# SEREEGA format
md_dict[sample_id] = {
    "seeds": seed_vertex_index or [indices],
    "act_src": {
        "patch_1": [vertices in active region],
        "patch_2": [vertices in active region 2 if multi-source],
        ...
    },
    "order": source patch order
}

# NMM format
dataset_meta["selected_region"][sample_idx] = (n_seeds, 1) array
    - Each row contains a seed vertex index for that sample
```

### Metric Dictionary Format

**File:** `model_training/eval.py`

```python
nmse_dict = {
    "cnn_1d": np.array([[0.12], [0.15], ..., [0.14]]),  # shape: (n_samples, 1)
    "lstm": np.array([[0.11], [0.14], ..., [0.13]]),
    "deep_sif": np.array([[0.10], [0.13], ..., [0.12]]),
    "eeg_vit": np.array([[0.09], [0.12], ..., [0.11]]),
    "MNE": np.array([[0.18], [0.22], ..., [0.20]]),
    "sLORETA": np.array([[0.20], [0.25], ..., [0.22]]),
}

# Similar structure for loc_error_dict, time_error_dict, auc_dict, psnr_dict
```

---

## Key Parameters

### Evaluation Configuration

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `eeg_snr` | 5 | SNR level (dB) for noise in EEG |
| `n_times` | 500 | Number of time samples in signal |
| `eval_simu_type` | - | "SEREEGA" or "NMM" |
| `source_space` | "ico3" | Source mesh resolution |
| `electrode_montage` | "standard_1020" | EEG electrode layout |
| `orientation` | "constrained" | "constrained" or "unconstrained" |

### Patch Orders (Expansion Levels)

```python
order=0: Only the seed vertex
order=1: Seed + immediate neighbors
order=2: Seed + neighbors + neighbors-of-neighbors
order=3: 3-ring expansion
order=5: Large search region (used in eval_zone)
```

---

## Common Issues & Solutions

### 1. Metric Values Seem Off

```python
# Check if data is normalized correctly:
if x_hat.abs().max() == 0:
    # Handle zero signal
    x_hat_unit = x_hat.abs()
else:
    x_hat_unit = x_hat.abs() / x_hat.abs().max()
```

### 2. Seed Not Found in Eval Zone

```python
# Evaluation zone computation ensures constrained search
eval_zone = utl.get_patch(order=5, idx=s, neighbors=neighbors)  # Initial search
eval_zone = np.setdiff1d(eval_zone, other_sources)  # Remove competing sources
eval_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)  # Restrict to local
```

### 3. Overlapping Regions Handling

```python
# When patches overlap, keep the source with highest amplitude
if len(inter) > 0:
    overlapping_regions += 1
    to_keep = torch.argmax(torch.Tensor([
        j[seeds[0], :].abs().max(),
        j[seeds[1], :].abs().max()
    ]))
    seeds = [seeds[to_keep]]  # Only evaluate strongest
```

---

## References

- Main evaluation: [model_training/eval.py](model_training/eval.py)
- Metrics library: [model_training/utils/utl_metrics.py](model_training/utils/utl_metrics.py)
- Utilities: [model_training/utils/utl.py](model_training/utils/utl.py)
- Full guide: [EVALUATION_METRICS_GUIDE.md](EVALUATION_METRICS_GUIDE.md)
- Diagrams: [EVALUATION_ARCHITECTURE_DIAGRAMS.md](EVALUATION_ARCHITECTURE_DIAGRAMS.md)

