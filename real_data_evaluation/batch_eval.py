"""
Batch Real Data Evaluation Script

Evaluates ESI models on multiple patient seizure data in one run.
Generates summary report across all patients.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from eval_real_data import main as eval_single_patient


class BatchEvaluator:
    """Process multiple patients and aggregate results."""
    
    def __init__(self, base_path, leadfield_path, results_path):
        """
        Initialize batch evaluator.
        
        Parameters:
        -----------
        base_path : str
            Base path to ictal/examples folder
        leadfield_path : str
            Path to leadfield .mat file
        results_path : str
            Output directory for results
        """
        self.base_path = Path(base_path)
        self.leadfield_path = leadfield_path
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def find_patients(self):
        """Find all patient folders (P1, P2, P3, etc.)."""
        patients = []
        for item in sorted(self.base_path.iterdir()):
            if item.is_dir() and item.name.startswith('P'):
                patients.append(item.name)
        return patients
    
    def find_seizures(self, patient_id):
        """Find all seizure recordings (data1.mat, data2.mat, etc.) for a patient."""
        patient_path = self.base_path / patient_id / 'sz_data'
        seizures = []
        
        if patient_path.exists():
            for i in range(1, 10):  # Check up to 10 seizures
                seizure_file = patient_path / f'data{i}.mat'
                if seizure_file.exists():
                    seizures.append(seizure_file)
        
        return seizures
    
    def evaluate_patient(self, patient_id, model_name='baseline', 
                        temporal_profile='gaussian', visualize=False):
        """
        Evaluate all seizures for a patient.
        
        Parameters:
        -----------
        patient_id : str
            Patient identifier (e.g., 'P2')
        model_name : str
            Name of model being evaluated
        temporal_profile : str
            Temporal profile for synthetic ground truth
        visualize : bool
            Whether to generate visualizations
        
        Returns:
        --------
        patient_results : dict
            Aggregated results for the patient
        """
        patient_path = self.base_path / patient_id
        anatomy_path = patient_path / 'anatomy' / 'projected_resection_to_fs_cortex.mat'
        
        if not anatomy_path.exists():
            print(f"Anatomy file not found for {patient_id}: {anatomy_path}")
            return None
        
        seizures = self.find_seizures(patient_id)
        if not seizures:
            print(f"No seizure data found for {patient_id}")
            return None
        
        print(f"\nEvaluating {patient_id} ({len(seizures)} seizures)...")
        
        seizure_results = []
        for seizure_file in seizures:
            seizure_num = seizure_file.stem[-1]  # Extract number from 'data1', 'data2', etc.
            
            try:
                print(f"  Processing {seizure_file.name}...", end=' ')
                
                # Call the main evaluation function
                sys.argv = [
                    'eval_real_data.py',
                    patient_id,
                    '-eeg_data', str(seizure_file),
                    '-resection_file', str(anatomy_path),
                    '-leadfield', self.leadfield_path,
                    '-results_path', str(self.results_path),
                    '-model_name', model_name,
                    '-temporal_profile', temporal_profile,
                ]
                
                if visualize:
                    sys.argv.append('-visualize')
                
                # Import and run eval_real_data
                from eval_real_data import main as eval_main
                try:
                    eval_main()
                    print("✓")
                except SystemExit:
                    pass  # eval_main calls sys.exit()
                except Exception as e:
                    print(f"✗ ({str(e)})")
                    continue
                
                seizure_results.append({
                    'patient': patient_id,
                    'seizure': seizure_num,
                    'file': seizure_file.name,
                    'status': 'completed'
                })
            
            except Exception as e:
                print(f"✗ (Error: {str(e)})")
                seizure_results.append({
                    'patient': patient_id,
                    'seizure': seizure_num,
                    'file': seizure_file.name,
                    'status': f'error: {str(e)}'
                })
        
        patient_result = {
            'patient': patient_id,
            'seizure_count': len(seizures),
            'processed_count': sum(1 for r in seizure_results if r['status'] == 'completed'),
            'seizure_details': seizure_results
        }
        
        return patient_result
    
    def generate_summary_report(self, all_results, output_file=None):
        """
        Generate summary report of all evaluations.
        
        Parameters:
        -----------
        all_results : list
            List of patient results
        output_file : str
            Optional output file for report
        """
        if output_file is None:
            output_file = self.results_path / 'batch_evaluation_summary.txt'
        
        report = []
        report.append("="*80)
        report.append("BATCH REAL DATA EVALUATION SUMMARY")
        report.append("="*80)
        report.append("")
        
        for patient_result in all_results:
            report.append(f"Patient: {patient_result['patient']}")
            report.append(f"  Seizures found: {patient_result['seizure_count']}")
            report.append(f"  Successfully processed: {patient_result['processed_count']}")
            report.append("")
        
        report.append(f"Total:  {sum(r['seizure_count'] for r in all_results)} seizures")
        report.append(f"Processed: {sum(r['processed_count'] for r in all_results)} seizures")
        report.append("")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)
        print(f"\nSummary report saved to: {output_file}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation of real patient data')
    
    parser.add_argument('-base_path', type=str, default='ictal/examples',
                       help='Path to ictal/examples folder')
    parser.add_argument('-leadfield', type=str, required=True,
                       help='Path to leadfield matrix')
    parser.add_argument('-results_path', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('-model_name', type=str, default='baseline',
                       help='Name of model being evaluated')
    parser.add_argument('-temporal_profile', type=str, default='gaussian',
                       choices=['gaussian', 'peak', 'uniform'],
                       help='Temporal profile for synthetic ground truth')
    parser.add_argument('-patients', type=str, nargs='+', default=None,
                       help='Specific patients to evaluate (e.g., P2 P3). If not specified, evaluates all found.')
    parser.add_argument('-visualize', action='store_true',
                       help='Generate visualizations for each seizure')
    
    args = parser.parse_args()
    
    # Initialize batch evaluator
    evaluator = BatchEvaluator(args.base_path, args.leadfield, args.results_path)
    
    # Find patients
    if args.patients:
        patients = args.patients
    else:
        patients = evaluator.find_patients()
    
    print("="*80)
    print(f"Batch Evaluation of Real Patient Data")
    print(f"Found patients: {patients}")
    print("="*80)
    
    # Evaluate each patient
    all_results = []
    for patient_id in patients:
        result = evaluator.evaluate_patient(
            patient_id,
            model_name=args.model_name,
            temporal_profile=args.temporal_profile,
            visualize=args.visualize
        )
        if result:
            all_results.append(result)
    
    # Generate summary report
    evaluator.generate_summary_report(all_results)
    
    print("\nBatch evaluation complete!")


if __name__ == "__main__":
    main()
