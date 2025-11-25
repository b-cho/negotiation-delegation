#!/bin/bash
# Script to visualize Experiment 1 results aggregated by house

# Usage: ./plot_by_house.sh results/experiment1_three_houses.csv

if [ $# -eq 0 ]; then
    echo "Usage: $0 <csv_file>"
    echo "Example: $0 results/experiment1_three_houses.csv"
    exit 1
fi

CSV_FILE="$1"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: File not found: $CSV_FILE"
    exit 1
fi

echo "Generating house aggregation plot for: $CSV_FILE"
python3 << EOF
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from src.analysis.visualization import plot_experiment1_by_house

plot_experiment1_by_house('$CSV_FILE', title='Price Recommendations Across Houses')
print("✓ Plot saved!")
EOF

