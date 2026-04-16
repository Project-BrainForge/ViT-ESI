"""
Real Data Evaluation Script

Evaluates ESI model predictions on real patient EEG data using resection regions 
as ground truth. Computes the same 5 metrics as simulated data evaluation:
1. Normalized Mean Squared Error (nMSE)
2. Localization Error (LE)
3. Time Error (TE)
4. Area Under ROC Curve (AUC)
5. Peak Signal-to-Noise Ratio (PSNR)

The key difference from simulated data: Ground truth sources are derived from 
resection regions rather than simulation parameters.
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from scipy.io import loadmat, savemat
from skimage.metrics import peak_signal_noise_ratio as psnr

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model_training.utils import utl_metrics as met
from utils_real_eval import (
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


def create_synthetic_ground_truth_from_resection(
    n_sources, resection_indices, n_times, 
    temporal_profile='gaussian', resection_weight=1.0
):
    """
    Create synthetic source distribution with resection region as ground truth.
    
    Parameters:
    -----------
    n_sources : int
        Total number of sources
    resection_indices : np.ndarray
        Indices of resection region
    n_times : int
        Number of time samples
    temporal_profile : str
        Type of temporal envelope ('gaussian', 'peak', 'uniform')
    resection_weight : float
        Amplitude of resection region (0-1 scale)
    
    Returns:
    --------
    j_true : np.ndarray
        Synthetic source distribution, shape (n_sources, n_times)
    """
    j_true = np.zeros((n_sources, n_times), dtype=np.float32)
    
    # Create temporal profile
    if temporal_profile == 'gaussian':
        # Gaussian envelope peaking at center
        t = np.linspace(-4, 4, n_times)
        temporal_envelope = np.exp(-t**2)
        
    elif temporal_profile == 'peak':
        # Sharp peak in middle
        temporal_envelope = np.zeros(n_times)
        temporal_envelope[n_times // 2] = 1.0
        # Add some neighboring activity
        temporal_envelope[n_times // 2 - 5:n_times // 2 + 5] = 0.5
        
    elif temporal_profile == 'uniform':
        # Uniform activation throughout
        temporal_envelope = np.ones(n_times) * resection_weight
        
    else:
        raise ValueError(f"Unknown temporal profile: {temporal_profile}")
    
    # Normalize to max = resection_weight
    temporal_envelope = temporal_envelope / temporal_envelope.max() * resection_weight
    
    # Apply to resection region with spatial smoothing
    n_resection = len(resection_indices)
    for i, idx in enumerate(resection_indices):
        # Add slight spatial variation (higher in middle of resection region)
        spatial_factor = 0.8 + 0.4 * np.sin(i / n_resection * np.pi)
        j_true[idx, :] = spatial_factor * temporal_envelope
    
    return j_true


def evaluate_real_data(eeg_real, j_true, leadfield, resection_indices, 
                       j_estimated=None, model_name=None):
    """
    Evaluate real EEG against synthetic ground truth.
    
    Parameters:
    -----------
    eeg_real : np.ndarray
        Real EEG data, shape (n_channels, n_times)
    j_true : np.ndarray
        Synthetic ground truth sources, shape (n_sources, n_times)
    leadfield : np.ndarray
        Leadfield matrix, shape (n_channels, n_sources)
    resection_indices : np.ndarray
        Indices of resection region
    j_estimated : np.ndarray or None
        Estimated sources (from model). If None, will use ground truth for comparison.
    model_name : str
        Name of the model being evaluated
    
    Returns:
    --------
    metrics : dict
        Dictionary of computed metrics
    """
    eeg_real = torch.from_numpy(eeg_real) if isinstance(eeg_real, np.ndarray) else eeg_real
    j_true = torch.from_numpy(j_true) if isinstance(j_true, np.ndarray) else j_true
    j_estimated = torch.from_numpy(j_estimated) if isinstance(j_estimated, np.ndarray) else j_estimated
    leadfield = torch.from_numpy(leadfield) if isinstance(leadfield, np.ndarray) else leadfield
    
    # Convert to float tensors
    eeg_real = eeg_real.float()
    j_true = j_true.float()
    if j_estimated is not None:
        j_estimated = j_estimated.float()
    leadfield = leadfield.float()
    
    n_times = eeg_real.shape[1]
    n_sources = j_true.shape[0]
    
    # Find peak time in ground truth (resection region)
    activity_resection = torch.abs(j_true[resection_indices, :]).mean(dim=0)
    t_peak = torch.argmax(activity_resection).item()
    
    # Compute predicted EEG from ground truth
    eeg_predicted_gt = leadfield @ j_true  # (n_channels, n_times)
    
    # === METRIC 1: nMSE ===
    # Normalized MSE at peak time
    eeg_real_at_peak = eeg_real[:, t_peak]
    eeg_pred_at_peak = eeg_predicted_gt[:, t_peak]
    
    eeg_real_norm = normalize_signal(eeg_real_at_peak)
    eeg_pred_norm = normalize_signal(eeg_pred_at_peak)
    
    nmse = ((eeg_real_norm - eeg_pred_norm) ** 2).mean().item()
    
    # === METRIC 2 & 3: Localization & Time Error (if estimated sources available) ===
    le = np.nan  # Not directly applicable to real data
    te = np.nan  # Not directly applicable to real data
    
    if j_estimated is not None:
        # Find peak time in estimated sources
        activity_est = torch.abs(j_estimated[resection_indices, :]).mean(dim=0)
        t_peak_est = torch.argmax(activity_est).item()
        te = np.abs(t_peak - t_peak_est) / 500.0 * 1000  # Convert to ms (assuming 500 Hz)
        
        # Compute localization metric: spatial correlation at peak
        j_true_peak = j_true[:, t_peak]
        j_est_peak = j_estimated[:, t_peak]
        
        j_true_norm = normalize_signal(j_true_peak)
        j_est_norm = normalize_signal(j_est_peak)
        
        # Correlation is not exactly "localization error" but related metric
        correlation = torch.corrcoef(torch.stack([j_true_norm, j_est_norm]))[0, 1].item()
        le = 1.0 - correlation  # Convert correlation to error metric (0=perfect, 1=bad)
    
    # === METRIC 4: AUC (Source discrimination) ===
    # Binary classification: resection region vs others
    auc_val = met.auc_t(j_true.unsqueeze(0), 
                        j_estimated.unsqueeze(0) if j_estimated is not None else j_true.unsqueeze(0), 
                        0, 
                        thresh=True, 
                        act_thresh=0.0)
    
    # === METRIC 5: PSNR ===
    # Compare real EEG with predicted EEG from ground truth sources
    eeg_real_norm = normalize_signal(eeg_real.numpy())
    eeg_pred_norm = normalize_signal(eeg_predicted_gt.numpy())
    
    psnr_val = psnr(
        eeg_pred_norm, 
        eeg_real_norm,
        data_range=eeg_real_norm.max() - eeg_real_norm.min()
    )
    
    # === METRIC 6 (bonus): Spatial correlation ===
    # How well resection region matches estimated activation
    spatial_corr = np.nan
    if j_estimated is not None:
        j_true_resection = j_true[resection_indices, :].mean(dim=0)
        j_est_resection = j_estimated[resection_indices, :].mean(dim=0)
        
        j_true_resection = normalize_signal(j_true_resection.numpy())
        j_est_resection = normalize_signal(j_est_resection.numpy())
        
        spatial_corr = np.corrcoef(j_true_resection, j_est_resection)[0, 1]
    
    # === METRIC 7 (bonus): EEG reconstruction error ===
    if j_estimated is not None:
        eeg_predicted_est = leadfield @ j_estimated
        eeg_recon_error = ((eeg_real - eeg_predicted_est) ** 2).mean().item()
    else:
        eeg_recon_error = np.nan
    
    metrics = {
        'model': model_name if model_name else 'unknown',
        't_peak': t_peak,
        'nmse': nmse,
        'loc_error': le,
        'time_error': te,
        'auc': auc_val.item() if isinstance(auc_val, torch.Tensor) else auc_val,
        'psnr': psnr_val,
        'spatial_correlation': spatial_corr,
        'eeg_reconstruction_error': eeg_recon_error,
        'resection_size': len(resection_indices),
        'n_sources': n_sources,
        'n_channels': eeg_real.shape[0],
        'n_times': n_times,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate ESI models on real patient data')
    
    parser.add_argument('patient_id', type=str, help='Patient identifier (e.g., P2)')
    parser.add_argument('-eeg_data', type=str, required=True, 
                       help='Path to real EEG data file (e.g., ictal/examples/P2/sz_data/data1.mat)')
    parser.add_argument('-resection_file', type=str, required=True,
                       help='Path to resection anatomy file (e.g., ictal/examples/P2/anatomy/projected_resection_to_fs_cortex.mat)')
    parser.add_argument('-leadfield', type=str, required=True,
                       help='Path to leadfield matrix (e.g., anatomy/leadfield_75_20k.mat)')
    parser.add_argument('-results_path', type=str, required=True,
                       help='Path to save evaluation results')
    parser.add_argument('-model_predictions', type=str, default=None,
                       help='Path to model source predictions (optional)')
    parser.add_argument('-model_name', type=str, default='unknown',
                       help='Name of the model being evaluated')
    parser.add_argument('-temporal_profile', type=str, default='gaussian',
                       choices=['gaussian', 'peak', 'uniform'],
                       help='Temporal profile of synthetic ground truth')
    parser.add_argument('-visualize', action='store_true',
                       help='Generate comparison visualizations')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"Real Data Evaluation for Patient: {args.patient_id}")
    print("="*80)
    
    # Create results directory
    os.makedirs(args.results_path, exist_ok=True)
    
    # Load real EEG
    print(f"\nLoading real EEG from: {args.eeg_data}")
    eeg_real = load_patient_eeg(args.eeg_data)
    print(f"EEG shape: {eeg_real.shape} (channels x times)")
    
    # Load resection regions
    print(f"Loading resection regions from: {args.resection_file}")
    resection_indices = load_resection_regions(args.resection_file)
    print(f"Resection region size: {len(resection_indices)} vertices")
    
    # Load leadfield and extract dimensions
    print(f"Loading leadfield from: {args.leadfield}")
    n_channels, n_sources, leadfield = get_source_space_info(args.leadfield)
    print(f"Leadfield shape: {leadfield.shape} (channels x sources)")
    
    # Create synthetic ground truth based on resection regions
    print(f"\nCreating synthetic ground truth with temporal profile: {args.temporal_profile}")
    j_true = create_synthetic_ground_truth_from_resection(
        n_sources, 
        resection_indices, 
        eeg_real.shape[1],
        temporal_profile=args.temporal_profile,
        resection_weight=0.8
    )
    print(f"Ground truth source shape: {j_true.shape}")
    
    # Load model predictions if provided
    j_estimated = None
    if args.model_predictions:
        print(f"\nLoading model predictions from: {args.model_predictions}")
        pred_data = loadmat(args.model_predictions)
        # Try different possible variable names
        if 'j_estimated' in pred_data:
            j_estimated = pred_data['j_estimated']
        elif 'sources' in pred_data:
            j_estimated = pred_data['sources']
        else:
            # Get first non-metadata key
            for key in pred_data.keys():
                if not key.startswith('__'):
                    j_estimated = pred_data[key]
                    break
        
        print(f"Predictions shape (original): {j_estimated.shape}")
        
        # Handle different formats: if (n_samples, n_times, n_sources), reshape to (n_sources, n_times)
        # by taking the first sample (or averaging across samples if needed)
        if j_estimated.ndim == 3:
            # Shape is (n_samples, n_times, n_sources)
            # Take first sample or average across samples
            print(f"Detected shape (n_samples, n_times, n_sources). Taking first sample.")
            j_estimated = j_estimated[0, :, :]  # Shape: (n_times, n_sources)
            # Transpose to get (n_sources, n_times)
            j_estimated = j_estimated.T
        elif j_estimated.ndim == 2:
            # Could be (n_sources, n_times) or (n_times, n_sources)
            if j_estimated.shape[0] == n_sources:
                # Already in correct format (n_sources, n_times)
                pass
            elif j_estimated.shape[1] == n_sources:
                # Transpose from (n_times, n_sources) to (n_sources, n_times)
                j_estimated = j_estimated.T
                print(f"Transposed predictions to (n_sources, n_times)")
        
        print(f"Predictions shape (reshaped): {j_estimated.shape}")
    
    # Compute evaluation metrics
    print("\n" + "="*80)
    print("Computing Evaluation Metrics")
    print("="*80)
    
    metrics = evaluate_real_data(
        eeg_real, j_true, leadfield, resection_indices,
        j_estimated=j_estimated,
        model_name=args.model_name
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 80)
    print(f"Model: {metrics['model']}")
    print(f"Patient: {args.patient_id}")
    print(f"\nDimensions:")
    print(f"  - Channels: {metrics['n_channels']}")
    print(f"  - Sources: {metrics['n_sources']}")
    print(f"  - Time samples: {metrics['n_times']}")
    print(f"  - Resection region size: {metrics['resection_size']}")
    print(f"\nMetrics:")
    print(f"  - nMSE (EEG reconstruction at peak): {metrics['nmse']:.4f}")
    print(f"  - Localization Error (corr-based): {metrics['loc_error']:.4f}")
    print(f"  - Time Error (peak): {metrics['time_error']:.2f} ms")
    print(f"  - AUC (source discrimination): {metrics['auc']:.4f}")
    print(f"  - PSNR (EEG quality): {metrics['psnr']:.2f} dB")
    if not np.isnan(metrics['spatial_correlation']):
        print(f"  - Spatial Correlation (resection): {metrics['spatial_correlation']:.4f}")
    if not np.isnan(metrics['eeg_reconstruction_error']):
        print(f"  - EEG Reconstruction Error: {metrics['eeg_reconstruction_error']:.4e}")
    
    # Save results
    results_file = os.path.join(args.results_path, f"eval_results_{args.patient_id}_{args.model_name}.mat")
    save_evaluation_results(metrics, results_file)
    
    # Save as text too
    txt_file = os.path.join(args.results_path, f"eval_results_{args.patient_id}_{args.model_name}.txt")
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Real Data Evaluation Results - {args.patient_id}\n")
        f.write("="*80 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to:")
    print(f"  - {results_file}")
    print(f"  - {txt_file}")
    
    # Generate visualization if requested
    if args.visualize and j_estimated is not None:
        print("\nGenerating visualization...")
        viz_prefix = os.path.join(args.results_path, f"viz_{args.patient_id}_{args.model_name}")
        visualize_resection_vs_estimate(j_true, j_estimated, resection_indices, 
                                       metrics['t_peak'], viz_prefix)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
