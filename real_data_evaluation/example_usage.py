"""
Quick Start Example - Real Data Evaluation

This script demonstrates how to evaluate real patient data using the evaluation framework.
"""

import os
import sys
from pathlib import Path

# Add parent directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from real_data_evaluation.eval_real_data import main as eval_main


def example_single_patient():
    """
    Example 1: Evaluate a single patient seizure
    
    This evaluates the real EEG data from patient P2, seizure 1 against
    synthetic ground truth derived from resection regions.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Patient Evaluation")
    print("="*80)
    
    patient_id = "P2"
    eeg_file = "ictal/examples/P2/sz_data/data1.mat"
    resection_file = "ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat"
    leadfield_file = "anatomy/leadfield_75_20k.mat"
    results_dir = "real_data_evaluation/results/example1"
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Run evaluation
    sys.argv = [
        'example_usage.py',
        patient_id,
        '-eeg_data', eeg_file,
        '-resection_file', resection_file,
        '-leadfield', leadfield_file,
        '-results_path', results_dir,
        '-model_name', 'baseline_gaussian',
        '-temporal_profile', 'gaussian',
        '-visualize'
    ]
    
    try:
        eval_main()
    except SystemExit:
        pass  # eval_main calls sys.exit()
    
    print(f"\nResults saved to: {results_dir}")


def example_compare_temporal_profiles():
    """
    Example 2: Compare different temporal profiles
    
    Evaluates the same EEG data with different temporal assumptions
    to see how ground truth modeling affects metrics.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Comparing Temporal Profiles")
    print("="*80)
    
    patient_id = "P2"
    eeg_file = "ictal/examples/P2/sz_data/data1.mat"
    resection_file = "ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat"
    leadfield_file = "anatomy/leadfield_75_20k.mat"
    
    profiles = ['gaussian', 'peak', 'uniform']
    
    for profile in profiles:
        print(f"\n--- Evaluating with {profile} temporal profile ---")
        
        results_dir = f"real_data_evaluation/results/example2_{profile}"
        os.makedirs(results_dir, exist_ok=True)
        
        sys.argv = [
            'example_usage.py',
            patient_id,
            '-eeg_data', eeg_file,
            '-resection_file', resection_file,
            '-leadfield', leadfield_file,
            '-results_path', results_dir,
            '-model_name', f'baseline_{profile}',
            '-temporal_profile', profile,
        ]
        
        try:
            eval_main()
        except SystemExit:
            pass
        
        print(f"Results saved to: {results_dir}")
    
    print("\n" + "="*80)
    print("Compare the results files to see how temporal profile affects metrics")
    print("="*80)


def example_batch_patients():
    """
    Example 3: Batch evaluation of multiple patients
    
    This would evaluate all patients found in ictal/examples/
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Evaluation (Setup Only)")
    print("="*80)
    
    print("""
To evaluate multiple patients, use the batch_eval.py script:

    cd real_data_evaluation
    python batch_eval.py \\
      -base_path ../ictal/examples \\
      -leadfield ../anatomy/leadfield_75_20k.mat \\
      -results_path ./results/batch \\
      -model_name my_model \\
      -temporal_profile gaussian \\
      -patients P2 P3 \\
      -visualize

This will automatically:
- Find all seizures (data1.mat, data2.mat, data3.mat) for each patient
- Load resection regions from anatomy files
- Create synthetic ground truth
- Compute all 5 metrics
- Generate comparison visualizations
- Save results in organized folder structure
    """)


def example_with_model_predictions():
    """
    Example 4: Evaluate with model predictions (pseudo-code)
    
    Shows how to evaluate model predictions against synthetic ground truth.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Evaluation with Model Predictions")
    print("="*80)
    
    print("""
If you have model source predictions (j_hat), save them as:
    
    model_predictions.mat containing:
        - 'sources': shape (n_sources, n_times)  or
        - 'j_estimated': shape (n_sources, n_times)

Then run:
    
    python eval_real_data.py P2 \\
      -eeg_data ../ictal/examples/P2/sz_data/data1.mat \\
      -resection_file ../ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat \\
      -leadfield ../anatomy/leadfield_75_20k.mat \\
      -results_path ./results \\
      -model_name my_cnn \\
      -model_predictions model_predictions.mat \\
      -visualize

This will additionally compute:
    - Localization Error (spatial): Correlation-based error
    - Time Error: Difference in peak times between true and estimated
    - Spatial Correlation: How well resection region matches estimated activation
    - EEG Reconstruction Error: MSE between predicted and real EEG
    """)


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("# Real Data Evaluation Framework - Examples")
    print("#"*80)
    
    print("""
This example demonstrates the real data evaluation framework.

The framework evaluates ESI model predictions on real patient seizure data by:
1. Loading real EEG from ictal/examples/P{i}/sz_data/data{j}.mat
2. Loading resection regions from ictal/examples/P{i}/anatomy/projected_resection_to_fs_cortex.mat
3. Creating synthetic ground truth where resection regions are active
4. Computing 5 evaluation metrics comparing model predictions to synthetic ground truth

Metrics computed:
- nMSE: EEG reconstruction error at seizure peak
- AUC: Source discrimination (resection vs others)
- Spatial Correlation: How well resection region detected
- Time Error: Peak time difference (if model predictions provided)
- PSNR: Overall EEG quality

Files:
- eval_real_data.py: Single patient evaluation
- batch_eval.py: Multiple patient batch processing
- utils_real_eval.py: Utility functions
- README.md: Full documentation
    """)
    
    input("\nPress Enter to continue to examples...")
    
    # Run examples
    try:
        example_single_patient()
    except Exception as e:
        print(f"Example 1 error: {e}")
    
    try:
        example_compare_temporal_profiles()
    except Exception as e:
        print(f"Example 2 error: {e}")
    
    try:
        example_batch_patients()
    except Exception as e:
        print(f"Example 3 error: {e}")
    
    try:
        example_with_model_predictions()
    except Exception as e:
        print(f"Example 4 error: {e}")
    
    print("\n" + "#"*80)
    print("# Examples Complete")
    print("#"*80)
    print("""
For more information, see real_data_evaluation/README.md

Next steps:
1. Modify examples to match your patient IDs and file paths
2. Implement temporal profile that matches your seizure data
3. Run batch evaluation on all patients
4. Compare results across temporal profiles
5. Add model predictions when available
    """)


if __name__ == "__main__":
    main()
