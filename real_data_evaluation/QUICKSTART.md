"""
Real Data Evaluation - Quick Start Guide

Follow these 3 steps to evaluate real patient seizure data.
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Print nice header."""
    print("\n" + "="*80)
    print(text)
    print("="*80)


def step1_understand_approach():
    """Step 1: Understand the approach."""
    print_header("STEP 1: Understand the Evaluation Approach")
    
    print("""
Real Data Evaluation Overview:

PROBLEM:
  - Simulated data: Ground truth sources known from simulation
  - Real data: Ground truth unknown

SOLUTION:
  Use resection regions as ground truth proxy
  
WORKFLOW:
  1. Load real EEG: ictal/examples/P{i}/sz_data/data{j}.mat
  2. Load resection regions: ictal/examples/P{i}/anatomy/projected_resection_to_fs_cortex.mat
  3. Create synthetic source distribution where:
     - Resection region vertices are ACTIVE
     - All other vertices are INACTIVE
  4. Apply forward model to create predicted EEG
  5. Compare model predictions against synthetic ground truth
  
WHY THIS WORKS:
  - Resection regions are clinically validated seizure zones
  - Forward model provides known ground truth
  - Same 5 metrics as simulated data (comparable results)
  - Allows unbiased model evaluation on real data

KEY ASSUMPTION:
  Seizure primarily originates from resection region
  (Usually true, but not absolute)
  """)


def step2_prepare_data():
    """Step 2: Prepare data."""
    print_header("STEP 2: Prepare Your Data")
    
    print("""
Required files:

1. Real EEG data:
   Location: ictal/examples/P{PatientID}/sz_data/data{N}.mat
   Content: Variable 'data' with shape (500, 75) = (time × channels)
   ✓ Found: Check ictal/examples/ for patient folders

2. Resection anatomy:
   Location: ictal/examples/P{PatientID}/anatomy/projected_resection_to_fs_cortex.mat
   Content: Variable 'resection_region' with vertex indices
   ✓ Found: Check anatomy/ subfolder in each patient

3. Leadfield matrix:
   Location: anatomy/leadfield_75_20k.mat
   Content: Matrix (75 channels × 994 sources)
   ✓ Found: Check anatomy/ folder in project root

4. Source space mesh (optional):
   Location: anatomy/sources_fsav_994.mat (for visualization)
   
If any files missing, populate from your experimental setup.
    """)


def step3_run_evaluation():
    """Step 3: Run evaluation."""
    print_header("STEP 3: Run Evaluation")
    
    print("""
OPTION A: Single Seizure Evaluation

cd real_data_evaluation
python eval_real_data.py P2 \\
  -eeg_data ../ictal/examples/P2/sz_data/data1.mat \\
  -resection_file ../ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat \\
  -leadfield ../anatomy/leadfield_75_20k.mat \\
  -results_path ./results \\
  -model_name baseline \\
  -temporal_profile gaussian \\
  -visualize

Output:
  ✓ eval_results_P2_baseline.mat      (metrics in MATLAB format)
  ✓ eval_results_P2_baseline.txt      (metrics in text format)
  ✓ viz_P2_baseline_comparison.png    (visualization)


OPTION B: Batch Evaluation (All Patients)

python batch_eval.py \\
  -base_path ../ictal/examples \\
  -leadfield ../anatomy/leadfield_75_20k.mat \\
  -results_path ./results/batch \\
  -model_name my_model \\
  -temporal_profile gaussian \\
  -visualize

Output:
  ✓ Multiple results files (one per seizure)
  ✓ batch_evaluation_summary.txt


OPTION C: With Model Predictions

If you have model predictions (j_hat), include:

python eval_real_data.py P2 \\
  -eeg_data ../ictal/examples/P2/sz_data/data1.mat \\
  -resection_file ../ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat \\
  -leadfield ../anatomy/leadfield_75_20k.mat \\
  -results_path ./results \\
  -model_name my_model \\
  -model_predictions predictions_P2_sz1.mat \\
  -visualize

This additionally computes:
  - Localization Error
  - Time Error  
  - Spatial Correlation
  - EEG Reconstruction Error
    """)


def step4_interpret_results():
    """Step 4: Interpret results."""
    print_header("STEP 4: Interpret Results")
    
    print("""
Output metrics and interpretation:

1. nMSE (Normalized Mean Squared Error)
   Range: 0-1
   Optimal: LOW (close to 0)
   Meaning: EEG reconstruction accuracy
   Interpretation:
     - ~0.1: Excellent fit
     - ~0.2: Good fit
     - ~0.4: Moderate fit
     - >0.5: Poor fit

2. AUC (Area Under ROC Curve)
   Range: 0-1
   Optimal: HIGH (close to 1)
   Meaning: Can discriminate resection region from others
   Interpretation:
     - >0.9: Excellent discrimination
     - 0.8-0.9: Good discrimination
     - 0.7-0.8: Fair discrimination
     - <0.7: Poor discrimination

3. PSNR (Peak Signal-to-Noise Ratio)
   Range: 0-∞ dB
   Optimal: HIGH (>20 dB)
   Meaning: Overall EEG signal quality
   Interpretation:
     - >25 dB: Excellent quality
     - 20-25 dB: Good quality
     - 10-20 dB: Moderate quality
     - <10 dB: Poor quality

4. Spatial Correlation (with model predictions)
   Range: -1 to +1
   Optimal: HIGH (close to 1)
   Meaning: Resection region matches estimated activation
   Interpretation:
     - >0.8: Excellent spatial match
     - 0.6-0.8: Good spatial match
     - 0.4-0.6: Moderate spatial match
     - <0.4: Poor spatial match

5. Time Error (with model predictions)
   Range: 0-∞ ms
   Optimal: LOW (close to 0)
   Meaning: Peak timing accuracy
   Interpretation:
     - <5 ms: Excellent timing
     - 5-10 ms: Good timing
     - 10-20 ms: Fair timing
     - >20 ms: Poor timing


COMPARING MODELS:

Model A vs Model B on same patient/seizure:
  - Compare each metric directly
  - Model with consistently lower errors = better
  - Look for trade-offs (e.g., good AUC but poor PSNR)

ACROSS TEMPORAL PROFILES:

  If you evaluated with different profiles (gaussian, peak, uniform):
  - Profiles that give similar metrics = robust model
  - Large metric variation = profile-dependent results
  - Choose profile that matches clinical seizure observations
    """)


def step5_visualizations():
    """Step 5: Understand visualizations."""
    print_header("STEP 5: Understanding Visualizations")
    
    print("""
Generated visualization: viz_P2_baseline_comparison.png

Contains 4 subplots:

1. TOP-LEFT: Ground Truth Activation Distribution
   - Histogram of activation values in resection region
   - Should show concentrated distribution around peak values

2. TOP-RIGHT: Estimated Activation Distribution
   - Histogram of model's activation in resection region
   - Compare shape with ground truth
   - If similar: good spatial detection

3. BOTTOM-LEFT: Temporal Activation Profile
   - Two curves: Ground truth (blue) and Estimated (orange)
   - Ground truth: Peak at center of measurement window
   - Estimated: Should follow similar temporal pattern
   - Vertical line: Peak time marking
   - If overlapping: good temporal accuracy

4. BOTTOM-RIGHT: Vertex-wise Max Activation Comparison
   - Scatter plot: X=GT max, Y=Estimated max
   - Points should cluster along diagonal y=x
   - If points below diagonal: model underestimates
   - If points above diagonal: model overestimates
  
INTERPRETATION:
  - All plots highly correlated = excellent model
  - Scattered/different patterns = model needs improvement
    """)


def step6_advanced():
    """Step 6: Advanced usage."""
    print_header("STEP 6: Advanced Usage")
    
    print("""
CUSTOMIZATION OPTIONS:

1. Temporal Profile:
   - gaussian: Gradual rise and fall (default)
   - peak: Sharp seizure onset
   - uniform: Sustained seizure activity
   
   Choose based on your patient's seizure characteristics.

2. Resection Weight:
   - Controls amplitude of resection region
   - Default: 0.8 (normalized)
   - Modify in utils_real_eval.py if needed

3. Custom Ground Truth:
   - Edit create_synthetic_ground_truth_from_resection()
   - Implement patient-specific temporal dynamics
   - Add multiple seizure sources if applicable

4. Model Predictions Format:
   - Save as .mat file with 'sources' or 'j_estimated'
   - Shape: (n_sources, n_times) = (994, 500)
   - Same scale as ground truth

INTEGRATION WITH TRAINING:

1. Train on simulated data
2. Evaluate on simulated validation set (test generalization)
3. Evaluate on real data using this framework (test real-world performance)
4. Compare metrics across conditions
5. Iterate on model architecture based on real data results


BATCH PROCESSING:

Process all patients and seizures:

python batch_eval.py \\
  -base_path ../ictal/examples \\
  -leadfield ../anatomy/leadfield_75_20k.mat \\
  -results_path ./results \\
  -model_name production_model \\
  -patients P2 P3 P4 \\
  -visualize

Then analyze results across all patients for comprehensive evaluation.
    """)


def main():
    """Run all steps."""
    print("\n" + "#"*80)
    print("# Real Data Evaluation - Quick Start Guide")
    print("#"*80)
    print("""
This guide walks through evaluating ESI model predictions on real patient
seizure data using resection regions as clinically-informed ground truth.

3 Files in this folder:
  1. eval_real_data.py  - Main evaluation script (single seizure)
  2. batch_eval.py      - Batch processing (multiple seizures)
  3. utils_real_eval.py - Utility functions

For complete details, see README.md
    """)
    
    steps = [
        ("1", "Understand Approach", step1_understand_approach),
        ("2", "Prepare Data", step2_prepare_data),
        ("3", "Run Evaluation", step3_run_evaluation),
        ("4", "Interpret Results", step4_interpret_results),
        ("5", "Visualizations", step5_visualizations),
        ("6", "Advanced Usage", step6_advanced),
    ]
    
    for step_num, title, func in steps:
        try:
            func()
            print("\n" + "-"*80)
            if step_num < "6":
                input(f"Press Enter to continue to Step {int(step_num)+1}...")
        except KeyboardInterrupt:
            print("\n\nQuick Start interrupted.")
            return
        except Exception as e:
            print(f"\nError in Step {step_num}: {e}")
            continue
    
    print_header("Quick Start Complete")
    print("""
Next steps:
1. Modify scripts with your patient IDs and file paths
2. Run single patient evaluation first
3. Then batch evaluate all patients
4. Compare results across temporal profiles
5. Integrate model predictions when available

For questions, refer to:
  - README.md (full documentation)
  - eval_real_data.py (code comments)
  - EVALUATION_METRICS_GUIDE.md (metric details)
    """)


if __name__ == "__main__":
    main()
