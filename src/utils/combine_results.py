#!/usr/bin/env python3
"""
Combine multiple Experiment 1 CSV results files into one aggregated file.

Usage:
    python -m src.utils.combine_results <csv1> <csv2> [csv3 ...] -o output.csv
    python src/utils/combine_results.py results/exp1.csv results/exp2.csv -o results/combined.csv
"""

import pandas as pd
import sys
import argparse
from pathlib import Path
from datetime import datetime


def combine_csv_files(csv_paths: list, output_path: str = None):
    """
    Combine multiple Experiment 1 CSV files into one.
    
    Args:
        csv_paths: List of paths to CSV files to combine
        output_path: Optional output path (default: combined_<timestamp>.csv)
    
    Returns:
        Path to the combined CSV file
    """
    if len(csv_paths) < 2:
        raise ValueError("Need at least 2 CSV files to combine")
    
    # Read all CSVs
    dataframes = []
    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            print(f"Warning: File not found: {csv_path}, skipping...")
            continue
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        dataframes.append(df)
    
    if len(dataframes) == 0:
        raise ValueError("No valid CSV files found")
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by recommendation_type, then by trial_number for consistency
    combined_df = combined_df.sort_values(
        by=['recommendation_type', 'trial_number'],
        kind='stable'
    ).reset_index(drop=True)
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use directory of first CSV
        output_dir = Path(csv_paths[0]).parent
        output_path = output_dir / f"experiment1_combined_{timestamp}.csv"
    
    # Save combined CSV
    combined_df.to_csv(output_path, index=False)
    
    print(f"\nCombined {len(dataframes)} files:")
    for csv_path in csv_paths:
        print(f"  - {csv_path}")
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Output saved to: {output_path}")
    
    # Print summary by house if house data is available
    if 'house_address' in combined_df.columns:
        print("\nSummary by house:")
        house_counts = combined_df['house_address'].value_counts()
        for house, count in house_counts.items():
            print(f"  {house}: {count} trials")
    
    return str(output_path)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Combine multiple Experiment 1 CSV results files"
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="CSV files to combine"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output CSV file path (default: combined_<timestamp>.csv)"
    )
    
    args = parser.parse_args()
    
    try:
        output_path = combine_csv_files(args.csv_files, args.output)
        print(f"\n✓ Successfully combined files!")
        print(f"  You can now use this combined file for analysis and plotting:")
        print(f"  python -m src.analysis.run_tests {output_path}")
        print(f"  python -m src.analysis.visualization {output_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

