#!/usr/bin/env python3
"""
Plot multiple Experiment 1 CSV files at once.

Usage:
    python -m src.analysis.plot_multiple results/experiment1_*.csv
    python -m src.analysis.plot_multiple results/experiment1_20251124_1*.csv
    
    # With shell globbing (zsh/bash):
    python -m src.analysis.plot_multiple results/experiment1_20251124_1*.csv > plot_output.txt 2>&1
"""

import sys
import glob
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
from pathlib import Path
from src.analysis.visualization import (
    plot_experiment1_results,
    plot_experiment1_by_race,
    plot_experiment1_by_gender,
    plot_experiment1_by_house
)


def plot_multiple_csvs(pattern: str, output_dir: str = None):
    """
    Plot multiple CSV files matching a pattern.
    
    Args:
        pattern: Glob pattern for CSV files (e.g., "results/experiment1_*.csv")
        output_dir: Optional output directory (default: same as CSV files)
    """
    # Expand glob pattern
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print()
    
    # Generate plots for each file
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        print(f"Processing: {csv_path.name}")
        print("-" * 70)
        
        try:
            # Main plot
            plot_experiment1_results(csv_file)
            
            # By race
            plot_experiment1_by_race(csv_file)
            
            # By gender
            plot_experiment1_by_gender(csv_file)
            
            # By house (if applicable)
            try:
                plot_experiment1_by_house(csv_file)
            except Exception as e:
                print(f"  (Skipping house plot: {e})")
            
            print(f"✓ Completed: {csv_path.name}\n")
            
        except Exception as e:
            print(f"✗ Error processing {csv_path.name}: {e}\n")
            continue
    
    print("=" * 70)
    print(f"Finished processing {len(csv_files)} file(s)")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.plot_multiple <glob_pattern> [file1] [file2] ...")
        print("Example: python -m src.analysis.plot_multiple results/experiment1_20251124_1*.csv")
        print("Or with shell expansion: python -m src.analysis.plot_multiple results/exp1.csv results/exp2.csv")
        sys.exit(1)
    
    # If multiple arguments, treat as list of files (shell-expanded glob)
    # If single argument with *, treat as glob pattern
    if len(sys.argv) > 2 or ('*' not in sys.argv[1] and not Path(sys.argv[1]).exists()):
        # Multiple files provided or pattern needs expansion
        if len(sys.argv) > 2:
            # Shell already expanded - use as-is
            csv_files = sys.argv[1:]
        else:
            # Single pattern - expand it
            pattern = sys.argv[1]
            csv_files = sorted(glob.glob(pattern))
    else:
        # Single file path
        csv_files = [sys.argv[1]]
    
    if not csv_files:
        print(f"No files found matching: {sys.argv[1:]}")
        sys.exit(1)
    
    # Process each file
    print(f"Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    print()
    
    # Generate plots for each file
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        print(f"Processing: {csv_path.name}")
        print("-" * 70)
        
        try:
            # Main plot
            plot_experiment1_results(csv_file)
            
            # By race
            plot_experiment1_by_race(csv_file)
            
            # By gender
            plot_experiment1_by_gender(csv_file)
            
            # By house (if applicable)
            try:
                plot_experiment1_by_house(csv_file)
            except Exception as e:
                print(f"  (Skipping house plot: {e})")
            
            print(f"✓ Completed: {csv_path.name}\n")
            
        except Exception as e:
            print(f"✗ Error processing {csv_path.name}: {e}\n")
            continue
    
    print("=" * 70)
    print(f"Finished processing {len(csv_files)} file(s)")


if __name__ == "__main__":
    main()

