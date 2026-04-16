"""
Utilities for evaluating ESI models on real patient data.

Real data evaluation uses resection regions as ground truth, creating a synthetic
source distribution where resection regions are active. The same 5 metrics from 
the simulated evaluation are computed.
"""

import numpy as np
import torch
from scipy.io import loadmat, savemat
from pathlib import Path


def load_patient_eeg(eeg_path):
    """
    Load real EEG data from patient file.
    
    Parameters:
    -----------
    eeg_path : str
        Path to .mat file containing EEG data
        Expected key: 'data' with shape (n_times, n_channels)
    
    Returns:
    --------
    M : np.ndarray or torch.Tensor
        EEG data, shape (n_channels, n_times)
        Note: transposed from file for consistency with simulated data format
    """
    data = loadmat(eeg_path)
    eeg = data['data']  # shape: (n_times, n_channels)
    M = eeg.T  # Transpose to (n_channels, n_times)
    return M


def load_resection_regions(anatomy_path):
    """
    Load resection region indices from patient anatomy file.
    
    Parameters:
    -----------
    anatomy_path : str
        Path to 'projected_resection_to_fs_cortex.mat' file
    
    Returns:
    --------
    resection_indices : np.ndarray
        Array of vertex indices in resection region, shape (n_resection_vertices,)
    """
    data = loadmat(anatomy_path)
    resection_region = data['resection_region'].flatten()  # shape: (n_vertices,)
    
    # Convert to 0-indexed (MATLAB uses 1-indexing)
    resection_indices = resection_region - 1
    
    return resection_indices.astype(np.int32)


def create_synthetic_ground_truth(n_sources, resection_indices, duration=1.0, 
                                   temporal_profile='gaussian'):
    """
    Create synthetic source distribution based on resection regions.
    
    This creates a ground truth source distribution where:
    - Resection region vertices have temporal activation
    - Other vertices are inactive (zero)
    - Temporal profile represents when sources were active
    
    Parameters:
    -----------
    n_sources : int
        Total number of source vertices (e.g., 994 for fsav_994)
    resection_indices : np.ndarray
        Indices of active sources (resection region)
    duration : float
        Duration in seconds (used for temporal extent estimation)
    temporal_profile : str
        Type of temporal profile: 'gaussian', 'peak', 'uniform'
        - 'gaussian': Bell curve peaking at mid-signal
        - 'peak': Sharp peak (spike)
        - 'uniform': Uniform activation throughout
    
    Returns:
    --------
    j_true : np.ndarray
        Synthetic source distribution, shape (n_sources, n_times)
    n_times : int
        Number of time samples
    """
    # Estimate n_times from duration (assuming 500 Hz sampling rate)
    fs = 500
    n_times = int(duration * fs)
    
    # Create source distribution
    j_true = np.zeros((n_sources, n_times), dtype=np.float32)
    
    # Create temporal profile
    if temporal_profile == 'gaussian':
        # Gaussian profile peaking at center
        t = np.linspace(-3, 3, n_times)
        temporal_envelope = np.exp(-t**2)  # Gaussian
        
    elif temporal_profile == 'peak':
        # Single peak at middle
        temporal_envelope = np.zeros(n_times)
        temporal_envelope[n_times // 2] = 1.0
        
    elif temporal_profile == 'uniform':
        # Uniform throughout
        temporal_envelope = np.ones(n_times)
    
    else:
        raise ValueError(f"Unknown temporal profile: {temporal_profile}")
    
    # Normalize temporal envelope
    temporal_envelope = temporal_envelope / temporal_envelope.max()
    
    # Assign activation to resection region
    # Add some spatial structure: vertices closer to center have higher amplitude
    for i, idx in enumerate(resection_indices):
        # Assign amplitude proportional to position in array (or could add spatial smoothing)
        amplitude = 1.0 + 0.1 * np.sin(i / len(resection_indices) * np.pi)
        j_true[idx, :] = amplitude * temporal_envelope
    
    return j_true, n_times


def apply_forward_model(sources, leadfield):
    """
    Compute predicted EEG from source distribution using leadfield.
    
    Parameters:
    -----------
    sources : np.ndarray or torch.Tensor
        Source distribution, shape (n_sources, n_times)
    leadfield : np.ndarray or torch.Tensor
        Leadfield matrix, shape (n_channels, n_sources)
    
    Returns:
    --------
    eeg_predicted : np.ndarray or torch.Tensor
        Predicted EEG, shape (n_channels, n_times)
    """
    if isinstance(sources, np.ndarray):
        eeg_predicted = leadfield @ sources  # Matrix multiplication
    else:  # torch.Tensor
        eeg_predicted = leadfield @ sources
    
    return eeg_predicted


def normalize_signal(signal):
    """
    Normalize signal by max absolute value.
    
    Parameters:
    -----------
    signal : np.ndarray or torch.Tensor
        Input signal
    
    Returns:
    --------
    signal_normalized : np.ndarray or torch.Tensor
        Normalized signal (max absolute value = 1)
    """
    max_val = np.abs(signal).max() if isinstance(signal, np.ndarray) else signal.abs().max()
    
    if max_val == 0:
        return signal
    
    return signal / max_val


def find_peak_time_region(source_distribution, resection_indices, order_around_peak=50):
    """
    Find time point of maximum activity in resection region.
    
    Parameters:
    -----------
    source_distribution : np.ndarray or torch.Tensor
        Source distribution, shape (n_sources, n_times)
    resection_indices : np.ndarray
        Indices of resection region
    order_around_peak : int
        Number of samples around peak to consider as "peak region"
    
    Returns:
    --------
    t_peak : int
        Time index of peak activity
    peak_region : tuple
        (t_start, t_end) - time range around peak
    peak_activity : float
        Maximum activity value at peak
    """
    # Get activity in resection region
    activity = np.abs(source_distribution[resection_indices, :])
    
    # Find peak across all vertices in resection region and all times
    if isinstance(activity, torch.Tensor):
        max_activity_per_time = activity.max(dim=0)[0]
        t_peak = max_activity_per_time.argmax().item()
    else:
        max_activity_per_time = activity.max(axis=0)
        t_peak = np.argmax(max_activity_per_time)
    
    peak_activity = max_activity_per_time[t_peak]
    
    # Define region around peak
    t_start = max(0, t_peak - order_around_peak)
    t_end = min(source_distribution.shape[1], t_peak + order_around_peak)
    peak_region = (t_start, t_end)
    
    return t_peak, peak_region, peak_activity


def compute_spatial_correlation(j_true, j_estimated, resection_indices, t_peak):
    """
    Compute spatial correlation at peak time between true and estimated sources.
    
    Parameters:
    -----------
    j_true : np.ndarray or torch.Tensor
        Ground truth sources, shape (n_sources, n_times)
    j_estimated : np.ndarray or torch.Tensor
        Estimated sources, shape (n_sources, n_times)
    resection_indices : np.ndarray
        Indices of resection region
    t_peak : int
        Time index of peak activity
    
    Returns:
    --------
    correlation : float
        Spatial correlation coefficient (Pearson's r)
    """
    true_at_peak = j_true[:, t_peak]
    est_at_peak = j_estimated[:, t_peak]
    
    # Normalize
    true_norm = normalize_signal(true_at_peak)
    est_norm = normalize_signal(est_at_peak)
    
    # Compute correlation
    if isinstance(true_norm, torch.Tensor):
        correlation = torch.corrcoef(torch.stack([true_norm, est_norm]))[0, 1].item()
    else:
        correlation = np.corrcoef(true_norm, est_norm)[0, 1]
    
    return correlation


def get_source_space_info(leadfield_path):
    """
    Extract source space information from leadfield.
    
    Parameters:
    -----------
    leadfield_path : str
        Path to leadfield .mat file
    
    Returns:
    --------
    n_channels : int
        Number of EEG channels
    n_sources : int
        Number of source vertices
    leadfield : np.ndarray
        Leadfield matrix, shape (n_channels, n_sources)
    """
    data = loadmat(leadfield_path)
    
    # Try different possible variable names
    if 'G' in data:
        leadfield = data['G']
    elif 'fwd' in data:
        leadfield = data['fwd']
    else:
        # Get first non-metadata key
        for key in data.keys():
            if not key.startswith('__'):
                leadfield = data[key]
                break
    
    n_channels, n_sources = leadfield.shape
    
    return n_channels, n_sources, leadfield


def create_patient_metadata(patient_id, eeg_file, resection_file, 
                            leadfield_file, output_dir):
    """
    Create metadata dictionary with all patient information for evaluation.
    
    Parameters:
    -----------
    patient_id : str
        Patient identifier (e.g., 'P2')
    eeg_file : str
        Path to EEG data file
    resection_file : str
        Path to resection region file
    leadfield_file : str
        Path to leadfield file
    output_dir : str
        Directory to save metadata
    
    Returns:
    --------
    metadata : dict
        Dictionary containing all patient information
    """
    # Load data
    M = load_patient_eeg(eeg_file)
    resection_indices = load_resection_regions(resection_file)
    n_channels, n_sources, leadfield = get_source_space_info(leadfield_file)
    
    # Create synthetic ground truth
    n_times = M.shape[1]
    j_true = np.zeros((n_sources, n_times))
    
    # Set resection region to high activation at certain timepoints
    for idx in resection_indices:
        # Create simple temporal profile: peak at middle
        t = np.arange(n_times)
        j_true[idx, :] = np.exp(-((t - n_times/2)**2) / (2 * (n_times/10)**2))
    
    metadata = {
        'patient_id': patient_id,
        'eeg_file': eeg_file,
        'resection_file': resection_file,
        'leadfield_file': leadfield_file,
        'n_channels': n_channels,
        'n_sources': n_sources,
        'n_times': n_times,
        'resection_indices': resection_indices.tolist(),
        'resection_size': len(resection_indices),
        'eeg_shape': M.shape,
        'leadfield_shape': leadfield.shape,
    }
    
    # Save metadata
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_dir}/{patient_id}_metadata.npy", metadata, allow_pickle=True)
    
    return metadata


def save_evaluation_results(results_dict, output_file):
    """
    Save evaluation results to .mat file for MATLAB visualization.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing evaluation metrics
    output_file : str
        Path to save results
    """
    savemat(output_file, results_dict)
    print(f"Results saved to {output_file}")


def visualize_resection_vs_estimate(j_true, j_estimated, resection_indices, 
                                    t_peak, output_prefix):
    """
    Create visualization comparing resection region activation vs estimated.
    
    Parameters:
    -----------
    j_true : np.ndarray
        Ground truth sources
    j_estimated : np.ndarray
        Estimated sources
    resection_indices : np.ndarray
        Resection region indices
    t_peak : int
        Peak time index
    output_prefix : str
        Prefix for output visualization files
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Spatial activation at peak time
    ax = axes[0, 0]
    true_at_peak = j_true[:, t_peak]
    ax.hist(true_at_peak[resection_indices], bins=20, alpha=0.7, label='Resection region')
    ax.set_xlabel('Activation amplitude')
    ax.set_ylabel('Count')
    ax.set_title('Ground truth activation distribution')
    ax.legend()
    
    # 2. Estimated activation at peak time
    ax = axes[0, 1]
    est_at_peak = j_estimated[:, t_peak]
    ax.hist(est_at_peak[resection_indices], bins=20, alpha=0.7, label='Resection region', color='orange')
    ax.set_xlabel('Activation amplitude')
    ax.set_ylabel('Count')
    ax.set_title('Estimated activation distribution')
    ax.legend()
    
    # 3. Temporal activation of resection region
    ax = axes[1, 0]
    true_temporal = np.abs(j_true[resection_indices, :]).mean(axis=0)
    est_temporal = np.abs(j_estimated[resection_indices, :]).mean(axis=0)
    ax.plot(true_temporal, label='Ground truth', linewidth=2)
    ax.plot(est_temporal, label='Estimated', linewidth=2, alpha=0.7)
    ax.axvline(t_peak, color='red', linestyle='--', label='Peak time')
    ax.set_xlabel('Time (samples)')
    ax.set_ylabel('Mean activation')
    ax.set_title('Temporal activation profile')
    ax.legend()
    
    # 4. Comparison of max activation per vertex
    ax = axes[1, 1]
    true_max = np.abs(j_true).max(axis=1)
    est_max = np.abs(j_estimated).max(axis=1)
    ax.scatter(true_max, est_max, alpha=0.5, s=10)
    ax.set_xlabel('Ground truth max activation')
    ax.set_ylabel('Estimated max activation')
    ax.set_title('Vertex-wise max activation comparison')
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_comparison.png", dpi=150)
    print(f"Visualization saved to {output_prefix}_comparison.png")
    plt.close()
