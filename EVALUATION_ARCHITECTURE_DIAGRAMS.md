# ViT-ESI Evaluation Architecture Diagrams

## 1. Metrics Calculation Flow for a Single Sample

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SINGLE VALIDATION SAMPLE (k)                        │
│                                                                         │
│  Inputs: EEG (M), Ground Truth Sources (j), Metadata (seeds, patches) │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        FOR EACH METHOD (CNN_1D, LSTM, DeepSIF, ViT, MNE, sLORETA)     │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ COMPUTE SOURCE ESTIMATE j_hat                                   │  │
│  │ ├─ Neural Networks: Pass EEG through model                      │  │
│  │ └─ Linear Methods: Apply inverse operator with leadfield       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ FOR EACH SEED in sample                                         │  │
│  │                                                                  │  │
│  │  1. Locate Ground Truth Peak                                   │  │
│  │     └─ t_eval_gt = argmax(|j[seed, :]|)                        │  │
│  │                                                                  │  │
│  │  2. Locate Estimated Seed (within eval_zone)                   │  │
│  │     ├─ eval_zone = patch around true seed                      │  │
│  │     └─ s_hat = argmax(|j_hat[eval_zone, t_eval_gt]|)           │  │
│  │                                                                  │  │
│  │  3. Calculate Localization Error (LE) ──────┐                  │  │
│  │     └─ LE = ||spos[seed] - spos[s_hat]||    │                  │  │
│  │                                              │                  │  │
│  │  4. Locate Estimated Peak Time              │ Accumulated      │  │
│  │     └─ t_eval_pred = argmax(|j_hat[s_hat,  │ over all seeds  │  │
│  │                             :]|)            │ and averaged    │  │
│  │                                              │                  │  │
│  │  5. Calculate Time Error (TE) ──────────────┤                  │  │
│  │     └─ TE = |t_eval_gt - t_eval_pred|       │                  │  │
│  │                                              │                  │  │
│  │  6. Calculate nMSE at peak time ────────────┤                  │  │
│  │     └─ nMSE = mean((j_norm - j_hat_norm)²)  │                  │  │
│  │                                              │                  │  │
│  │  7. Calculate AUC (ROC analysis) ───────────┤                  │  │
│  │     ├─ Active sources = class 1              │                  │  │
│  │     ├─ Inactive sources = class 0            │                  │  │
│  │     └─ AUC = area_under_roc_curve()          │                  │  │
│  │                                              ▼                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ CALCULATE PSNR (over entire signal)                          │  │
│  │ └─ Peak SNR = 20*log10(max_val / normalized_MSE)             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                    │                                │
└────────────────────────────────────┼────────────────────────────────┘
                                     │
                                     ▼
        ┌────────────────────────────────────────────────┐
        │ STORE FOR METHOD k:                            │
        │ ├─ nmse_dict[method][k]                        │
        │ ├─ loc_error_dict[method][k]                   │
        │ ├─ time_error_dict[method][k]                  │
        │ ├─ auc_dict[method][k]                         │
        │ └─ psnr_dict[method][k]                        │
        └────────────────────────────────────────────────┘
```

---

## 2. Evaluation Data Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    LOAD & INITIALIZE                             │
│                                                                  │
│  ├─ Head Model (leadfield G, source positions, neighbors)       │
│  ├─ Validation Dataset (EEG, source, metadata)                 │
│  └─ Model Weights / Inverse Operators                          │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              MAIN EVALUATION LOOP                                │
│   for sample_idx in range(n_validation_samples):                │
│                                                                  │
│   Load: M (EEG), j (source), metadata (seeds, patches)         │
└──────────────────────────────────────────────────────────────────┘
        │
        ├─→ For each method:
        │   ├─→ Compute/Load j_hat
        │   ├─→ For each seed:
        │   │   ├─→ Calculate LE, TE, nMSE, AUC
        │   │   └─→ Accumulate
        │   └─→ Calculate PSNR
        │
        └─→ Store all metrics in dictionaries
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              AGGREGATE RESULTS                                   │
│                                                                  │
│  For each method:                                               │
│  ├─ Compute mean(metric) and std(metric)                        │
│  ├─ Filter out noise-only samples                               │
│  └─ Generate summary statistics                                 │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│              SAVE RESULTS TO CSV                                 │
│                                                                  │
│  ├─ evaluation_metrics_*.csv (summary table)                    │
│  │  └─ mean ± std for each metric across validation set         │
│  │                                                               │
│  └─ evaluation_*.csv (full distribution)                        │
│     └─ per-sample metrics for plotting                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Seed Localization Process

```
Ground Truth Source Distribution           Estimated Source Distribution
        (j)                                        (j_hat)

    ┌─────────┐                               ┌─────────┐
    │  Brain  │                               │  Brain  │
    │         │                               │         │
    │    ●    │ ← seed (s)                    │    ◆    │ ← est. seed (s_hat)
    │   /|\   │                               │   /|\   │
    │  / | \  │ ← order-2 patch               │  / | \  │
    │ ●  ●  ● │   (eval zone for finding      │ ○  ◆  ○ │
    └─────────┘   estimated seed)             └─────────┘
         │                                          │
         └──────────────────────────────────────────┘
                          │
                          ▼
         Calculate Euclidean distance
         between seed positions in 3D space
                          │
                          ▼
         Localization Error (LE) = ||spos[s] - spos[s_hat]||
                in millimeters (mm)
```

---

## 4. Temporal Localization

```
Ground Truth Time Series            Estimated Time Series
        j[s, :]                             j_hat[s_hat, :]

    Amplitude                          Amplitude
       │                                  │
    1.0│    ┌──┐                       1.0│       ┌──┐
       │    │  │                          │       │  │
    0.5│    │  │                       0.5│   ┌───┘  └───┐
       │  ┌─┘  └─┐                        │   │          │
    0.0└──┴───────┴────► Time (samples)  0.0└──┴──────────┴────► Time
         0   50  100                       0   52   100

         Peak at t=50 sample           Peak at t=52 sample
         (t_eval_gt)                   (t_eval_pred)

                          │
                          ▼
         Time Error = |t_eval_gt - t_eval_pred| = |50 - 52| = 2 samples
         = 2 * dt (where dt = sampling interval)
         = 2 * (1/500) s = 4 ms (for 500 Hz sampling)
```

---

## 5. AUC Calculation (ROC Analysis)

```
Ground Truth (binary)        Estimated Signal (continuous)
j at t=50 (peak)             j_hat at t=50 (peak)

Sources    Class             Sources    Score (normalized)
─────────  ──────            ──────     ──────────────────
1          1 (active)        1          0.92  ← high
2          1 (active)        2          0.87  ← high
3          0 (inactive)      3          0.15  ← low
4          0 (inactive)      4          0.23  ← low
5          0 (inactive)      5          0.08  ← low
...                          ...        ...

              │
              ▼
    ┌─────────────────────┐
    │ ROC Curve           │
    │ (True Positive Rate │
    │  vs False Positive  │
    │  Rate)              │
    │                     │
    │   ╱─────           │
    │  ╱    ╱             │
    │ ╱    ╱              │
    │────────────────     │
    │      ╱─────         │
    └─────────────────────┘
              │
              ▼
        AUC = 0.94
    (High: good discrimination
     between active/inactive)
```

---

## 6. nMSE Calculation at Peak Time

```
Ground Truth Sources (normalized)    Estimated Sources (normalized)
j_norm = j / max(|j|)               j_hat_norm = j_hat / max(|j_hat|)

at t = t_eval_gt (peak time):

Source     j_norm    j_hat_norm    Error²
───────    ──────    ──────────    ──────
1          1.00      0.95          0.0025
2          1.00      0.92          0.0064
3          0.50      0.45          0.0025
4          0.20      0.25          0.0025
5          0.10      0.08          0.0004
...        ...       ...           ...

                     │
                     ▼
    nMSE = mean(Error²)
         = (0.0025 + 0.0064 + 0.0025 + 0.0025 + 0.0004 + ...) / n_sources
         ≈ 0.15
    
    (Lower is better: 0 = perfect, 1 = completely wrong)
```

---

## 7. PSNR Calculation (Full Signal)

```
Ground Truth (full spatiotemporal)   Estimated (full spatiotemporal)
j (n_sources × n_times)              j_hat (n_sources × n_times)

Normalize both:
  j_norm = j / max(|j|)
  j_hat_norm = j_hat / max(|j_hat|)

              │
              ▼
    MSE = mean((j_norm - j_hat_norm)²)
         over all spatial and temporal indices
    
              │
              ▼
    PSNR = 20 × log₁₀(max_value / √MSE) [in dB]
    
    Example:
      MSE = 0.01
      PSNR = 20 × log₁₀(1.0 / √0.01)
           = 20 × log₁₀(10.0)
           = 20 × 1.0
           = 20 dB
    
    (Higher PSNR = better: >20 dB is good reconstruction)
```

---

## 8. Metric Aggregation Over Validation Set

```
For all validation samples k=1 to N:
┌─────────────────────────────────────────────────┐
│ Sample 1     │ Sample 2     │ ... │ Sample N    │
├──────────────┼──────────────┼─────┼─────────────┤
│LE: 2.1 mm    │ LE: 3.5 mm   │ ... │ LE: 2.8 mm  │
│TE: 0.5 ms    │ TE: 1.2 ms   │ ... │ TE: 0.8 ms  │
│nMSE: 0.12    │ nMSE: 0.18   │ ... │ nMSE: 0.14  │
│AUC: 0.95     │ AUC: 0.91    │ ... │ AUC: 0.93   │
│PSNR: 24 dB   │ PSNR: 20 dB  │ ... │ PSNR: 22 dB │
└─────────────────────────────────────────────────┘
        │
        ▼
    Compute Statistics:
    ┌────────────────────────────────┐
    │ Metric      │ Mean  │ Std Dev │
    ├─────────────┼───────┼─────────┤
    │ LE (mm)     │ 2.8   │ 0.65    │
    │ TE (ms)     │ 0.83  │ 0.24    │
    │ nMSE        │ 0.148 │ 0.026   │
    │ AUC         │ 0.930 │ 0.025   │
    │ PSNR (dB)   │ 22.0  │ 1.85    │
    └────────────────────────────────┘
        │
        ▼
    Save to CSV files:
    ├─ evaluation_metrics_*.csv ← Summary table
    └─ evaluation_*.csv         ← Full distribution
```

---

## 9. File I/O Summary

```
INPUT FILES
───────────────────────────────────────────────────────────────
Simulation Data:
├─ EEG: simulation/{...}/eeg_{sample}.mat or .npy
├─ Sources: simulation/{...}/sources_{sample}.mat or .npy
└─ Metadata: simulation/{...}/metadata.json

Head Model:
├─ Leadfield: anatomy/leadfield_75_20k.mat (G matrix)
├─ Source Space: anatomy/sources_fsav_994.mat or computed
└─ Neighbors: stored in head_model.neighbors

Trained Models:
├─ CNN: results/{dataset}/trained_models/{run}/1dcnn_model.pt
├─ LSTM: results/{dataset}/trained_models/{run}/lstm_model.pt
├─ DeepSIF: results/{dataset}/trained_models/{run}/deepsif_model.pt
└─ ViT: results/{dataset}/trained_models/{run}/vit_model.pt


OUTPUT FILES
───────────────────────────────────────────────────────────────
Results CSV:
├─ evaluation_metrics_{config}_{method}.csv
│  └─ 1 row: method performance summary (mean ± std)
│
└─ evaluation_{config}_{method}.csv
   └─ N rows: per-sample metrics (for plotting)

Example Config String:
train_simu_sereega_eval_simu_sereega_method_cnn_1d_
srcspace_fsav_994_datasetmes_debug_n_train_1000_test.csv
```

---

## 10. Method Comparison at a Glance

```
Method         │ Type     │ Training │ Inference │ Scalability │ Interpretability
───────────────┼──────────┼──────────┼───────────┼─────────────┼─────────────────
 1D-CNN        │ Neural   │ Required │ Fast      │ Medium      │ Low
 LSTM          │ Neural   │ Required │ Medium    │ High        │ Low
 DeepSIF       │ Neural   │ Required │ Fast      │ Medium      │ Medium
 ViT (EEG-ViT) │ Neural   │ Required │ Medium    │ High        │ Medium
 MNE           │ Linear   │ None     │ Very Fast │ Very High   │ Very High
 sLORETA       │ Linear   │ None     │ Very Fast │ Very High   │ High

Metrics evaluated for all methods:
├─ LE (mm)       : Spatial accuracy
├─ TE (ms)       : Temporal accuracy
├─ nMSE          : Signal reconstruction
├─ AUC           : Source discrimination
└─ PSNR (dB)     : Overall signal quality
```

