#!/usr/bin/env python3
"""
Script to extract metrics from results.json files and export to Excel.

Usage:
    python export_metrics.py -m Common/4D_SOTA/output/N3DV/sear_steak --start 0 --end 299 -o metrics.xlsx
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Export metrics from results.json to Excel")
    parser.add_argument("-m", "--model_path", type=str, 
                        help="Path to the model output directory (e.g., cook_spinach)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame number (default: 0)")
    parser.add_argument("--end", type=int, default=299,
                        help="End frame number (default: 299)")
    parser.add_argument("-o", "--output", type=str, 
                        help="Output Excel file path (default: metrics.xlsx)")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    # Collect all metrics
    iter1_data = []
    iter6000_data = []
    missing_frames = []
    
    print(f"Collecting metrics from frames {args.start} to {args.end}...")
    for frame_num in tqdm(range(args.start, args.end + 1)):
        frame_dir = model_path / f"frame_{frame_num}"
        results_file = frame_dir / "results.json"
        
        if not results_file.exists():
            missing_frames.append(frame_num)
            continue
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract metrics for each iteration
            for iteration_key, metrics in results.items():
                # Extract iteration number from key like "ours_1000"
                iteration = int(iteration_key.replace("ours_", ""))
                
                if iteration == 1:
                    row = {
                        'frame': frame_num,
                        'iteration': int(iteration),
                        'PSNR': metrics.get('PSNR', None),
                        'DSSIM_1': metrics.get('DSSIM_1', None),
                        'DSSIM_2': metrics.get('DSSIM_2', None),
                        'LPIPS_alex': metrics.get('LPIPS_alex', None)
                    }
                    iter1_data.append(row)
                
                if iteration == 6000:
                    row = {
                        'iteration': int(iteration),
                        'frame': frame_num,
                        'PSNR': metrics.get('PSNR', None),
                        'DSSIM_1': metrics.get('DSSIM_1', None),
                        'DSSIM_2': metrics.get('DSSIM_2', None),
                        'LPIPS_alex': metrics.get('LPIPS_alex', None)
                    }
                    iter6000_data.append(row)
        
        except Exception as e:
            print(f"Error reading frame {frame_num}: {e}")
            missing_frames.append(frame_num)
    
    all_data = iter1_data + iter6000_data
    
    if not all_data:
        print("Error: No data collected!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by frame and iteration
    df = df.sort_values(['iteration', 'frame'])
    
    print(f"\nCollected {len(df)} metric entries from {df['frame'].nunique()} frames")
    if missing_frames:
        print(f"Missing {len(missing_frames)} frames")
    
    # Create Excel writer with multiple sheets
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: All data (long format)
        df.to_excel(writer, sheet_name='All Data', index=False)
        
        # Sheet 2-4: Pivot tables for each metric
        for metric in ['PSNR', 'DSSIM_1', 'DSSIM_2', 'LPIPS_alex']:
            pivot = df.pivot(index='frame', columns='iteration', values=metric)
            pivot = pivot.sort_index(axis=1)  # Sort columns by iteration
            pivot.to_excel(writer, sheet_name=metric)
        
        # Sheet 5: Summary statistics
        summary_data = []
        for iteration in sorted(df['iteration'].unique()):
            iter_df = df[df['iteration'] == iteration]
            summary_data.append({
                'iteration': iteration,
                'PSNR_mean': iter_df['PSNR'].mean(),
                'PSNR_std': iter_df['PSNR'].std(),
                'DSSIM_1_mean': iter_df['DSSIM_1'].mean(),
                'DSSIM_1_std': iter_df['DSSIM_1'].std(),
                'DSSIM_2_mean': iter_df['DSSIM_2'].mean(),
                'DSSIM_2_std': iter_df['DSSIM_2'].std(),
                'LPIPS_alex_mean': iter_df['LPIPS_alex'].mean(),
                'LPIPS_alex_std': iter_df['LPIPS_alex'].std(),
                'num_frames': len(iter_df)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nExcel file saved to: {output_path}")
    print(f"\nSheets created:")
    print(f"  - All Data: Raw data in long format")
    print(f"  - PSNR: Frame x Iteration pivot table")
    print(f"  - DSSIM_1: Frame x Iteration pivot table")
    print(f"  - DSSIM_2: Frame x Iteration pivot table")
    print(f"  - LPIPS_alex: Frame x Iteration pivot table")
    print(f"  - Summary: Mean and std for each iteration")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()