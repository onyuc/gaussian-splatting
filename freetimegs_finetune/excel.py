#!/usr/bin/env python3
"""
Script to extract metrics from results.json files and export to Excel.

Usage:
    python export_metrics.py -m /path/to/cook_spinach --start 0 --end 299 -o metrics.xlsx
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
                        default="/scratch/rchkl2380/Workspace/4D_SOTA/train_outputs/3dgs_output/freetime_finetune_col/cook_spinach",
                        help="Path to the model output directory (e.g., cook_spinach)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start frame number (default: 0)")
    parser.add_argument("--end", type=int, default=299,
                        help="End frame number (default: 299)")
    parser.add_argument("-o", "--output", type=str, 
                        default="/scratch/rchkl2380/Workspace/4D_SOTA/train_outputs/3dgs_output/freetime_finetune_col/cook_spinach/metrics.xlsx",
                        help="Output Excel file path (default: metrics.xlsx)")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        return
    
    # Collect all metrics
    all_data = []
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
                iteration = iteration_key.replace("ours_", "")
                
                row = {
                    'frame': frame_num,
                    'iteration': int(iteration),
                    'SSIM': metrics.get('SSIM', None),
                    'PSNR': metrics.get('PSNR', None),
                    'LPIPS': metrics.get('LPIPS', None)
                }
                all_data.append(row)
        
        except Exception as e:
            print(f"Error reading frame {frame_num}: {e}")
            missing_frames.append(frame_num)
    
    if not all_data:
        print("Error: No data collected!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by frame and iteration
    df = df.sort_values(['frame', 'iteration'])
    
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
        for metric in ['SSIM', 'PSNR', 'LPIPS']:
            pivot = df.pivot(index='frame', columns='iteration', values=metric)
            pivot = pivot.sort_index(axis=1)  # Sort columns by iteration
            pivot.to_excel(writer, sheet_name=metric)
        
        # Sheet 5: Summary statistics
        summary_data = []
        for iteration in sorted(df['iteration'].unique()):
            iter_df = df[df['iteration'] == iteration]
            summary_data.append({
                'iteration': iteration,
                'SSIM_mean': iter_df['SSIM'].mean(),
                'SSIM_std': iter_df['SSIM'].std(),
                'PSNR_mean': iter_df['PSNR'].mean(),
                'PSNR_std': iter_df['PSNR'].std(),
                'LPIPS_mean': iter_df['LPIPS'].mean(),
                'LPIPS_std': iter_df['LPIPS'].std(),
                'num_frames': len(iter_df)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"\nExcel file saved to: {output_path}")
    print(f"\nSheets created:")
    print(f"  - All Data: Raw data in long format")
    print(f"  - SSIM: Frame x Iteration pivot table")
    print(f"  - PSNR: Frame x Iteration pivot table")
    print(f"  - LPIPS: Frame x Iteration pivot table")
    print(f"  - Summary: Mean and std for each iteration")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()