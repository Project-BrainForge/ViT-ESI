"""
Real Data Evaluation Framework

Evaluation of ESI models on real patient seizure data using resection regions
as clinically-informed ground truth proxy.

Main Modules:
    - eval_real_data: Single seizure evaluation
    - batch_eval: Multiple patient/seizure batch processing
    - utils_real_eval: Utility functions for data loading and processing
"""

__version__ = "1.0.0"
__author__ = "ViT-ESI Team"

from .utils_real_eval import (
    load_patient_eeg,
    load_resection_regions,
    create_synthetic_ground_truth,
    apply_forward_model,
    normalize_signal,
    find_peak_time_region,
    get_source_space_info,
    create_patient_metadata,
    save_evaluation_results,
    visualize_resection_vs_estimate,
)

__all__ = [
    'load_patient_eeg',
    'load_resection_regions',
    'create_synthetic_ground_truth',
    'apply_forward_model',
    'normalize_signal',
    'find_peak_time_region',
    'get_source_space_info',
    'create_patient_metadata',
    'save_evaluation_results',
    'visualize_resection_vs_estimate',
]
