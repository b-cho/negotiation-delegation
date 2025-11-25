#!/usr/bin/env python3
"""
Compare statistical significance across low-context and high-context experiments.

Usage:
    python -m src.analysis.compare_significance
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_ttest_file(filepath: str) -> dict:
    """Parse t-test results file and extract key statistics"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    results = {
        'buyer_gender_p': None,
        'buyer_race_anova_p': None,
        'buyer_race_anova_f': None,
        'buyer_race_anova_sig': False,
        'seller_gender_p': None,
        'seller_race_anova_p': None,
        'seller_race_anova_f': None,
        'seller_race_anova_sig': False,
    }
    
    # Extract buyer gender t-test
    buyer_gender_match = re.search(r'BUYER RECOMMENDATIONS.*?GENDER T-TEST:.*?p-value:\s+([\d.]+)', content, re.DOTALL)
    if buyer_gender_match:
        results['buyer_gender_p'] = float(buyer_gender_match.group(1))
    
    # Extract buyer race ANOVA
    buyer_anova_match = re.search(r'BUYER.*?RACE ANOVA:.*?F-statistic:\s+([\d.]+).*?p-value:\s+([\d.]+).*?Result:\s+([*]*\s*SIGNIFICANT|not significant)', content, re.DOTALL)
    if buyer_anova_match:
        results['buyer_race_anova_f'] = float(buyer_anova_match.group(1))
        results['buyer_race_anova_p'] = float(buyer_anova_match.group(2))
        results['buyer_race_anova_sig'] = 'SIGNIFICANT' in buyer_anova_match.group(3)
    
    # Extract seller gender t-test
    seller_gender_match = re.search(r'SELLER RECOMMENDATIONS.*?GENDER T-TEST:.*?p-value:\s+([\d.]+)', content, re.DOTALL)
    if seller_gender_match:
        results['seller_gender_p'] = float(seller_gender_match.group(1))
    
    # Extract seller race ANOVA
    seller_anova_match = re.search(r'SELLER.*?RACE ANOVA:.*?F-statistic:\s+([\d.]+).*?p-value:\s+([\d.]+).*?Result:\s+([*]*\s*SIGNIFICANT|not significant)', content, re.DOTALL)
    if seller_anova_match:
        results['seller_race_anova_f'] = float(seller_anova_match.group(1))
        results['seller_race_anova_p'] = float(seller_anova_match.group(2))
        results['seller_race_anova_sig'] = 'SIGNIFICANT' in seller_anova_match.group(3)
    
    return results


def create_comparison_visualization():
    """Create comparison visualization of statistical significance"""
    
    # Parse all t-test files
    low_context = parse_ttest_file('results/experiment1/low_context/ttest_results_low_context.txt')
    house1 = parse_ttest_file('results/experiment1/high_context/ttest_results_house1.txt')
    house2 = parse_ttest_file('results/experiment1/high_context/ttest_results_house2.txt')
    house3 = parse_ttest_file('results/experiment1/high_context/ttest_results_house3.txt')
    
    # Prepare data
    experiments = ['Low Context', 'High Context\n(House 1)', 'High Context\n(House 2)', 'High Context\n(House 3)']
    buyer_race_anova_p = [
        low_context['buyer_race_anova_p'],
        house1['buyer_race_anova_p'],
        house2['buyer_race_anova_p'],
        house3['buyer_race_anova_p']
    ]
    seller_race_anova_p = [
        low_context['seller_race_anova_p'],
        house1['seller_race_anova_p'],
        house2['seller_race_anova_p'],
        house3['seller_race_anova_p']
    ]
    buyer_gender_p = [
        low_context['buyer_gender_p'],
        house1['buyer_gender_p'],
        house2['buyer_gender_p'],
        house3['buyer_gender_p']
    ]
    seller_gender_p = [
        low_context['seller_gender_p'],
        house1['seller_gender_p'],
        house2['seller_gender_p'],
        house3['seller_gender_p']
    ]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(3, 2, hspace=0.2, wspace=0.3, height_ratios=[1, 1, 1.1])
    
    # Color scheme: red for significant, gray for not significant
    sig_color = '#d32f2f'  # Red
    nsig_color = '#757575'  # Gray
    alpha_sig = 0.8
    alpha_nsig = 0.5
    
    # Plot 1: Buyer Race ANOVA p-values
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(experiments, buyer_race_anova_p, edgecolor='black', linewidth=1.5)
    for i, (bar, p) in enumerate(zip(bars1, buyer_race_anova_p)):
        bar.set_color(sig_color if p < 0.05 else nsig_color)
        bar.set_alpha(alpha_sig if p < 0.05 else alpha_nsig)
    
    # Add significance line at 0.05
    ax1.axhline(y=0.05, color='black', linestyle='--', linewidth=1, label='α = 0.05')
    
    # Add p-value labels
    for i, (bar, p) in enumerate(zip(bars1, buyer_race_anova_p)):
        height = bar.get_height()
        label = f'p={p:.4f}'
        if p < 0.001:
            label += '\n***'
        elif p < 0.01:
            label += '\n**'
        elif p < 0.05:
            label += '\n*'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold' if p < 0.05 else 'normal')
    
    ax1.set_ylabel('p-value', fontsize=11, fontweight='bold')
    ax1.set_title('Buyer Race ANOVA Significance', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, max(buyer_race_anova_p) * 1.2])
    ax1.set_yticks([0, 0.05, 0.1, 0.2, 0.3])
    ax1.tick_params(labelsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=9)
    
    # Plot 2: Seller Race ANOVA p-values
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(experiments, seller_race_anova_p, edgecolor='black', linewidth=1.5)
    for i, (bar, p) in enumerate(zip(bars2, seller_race_anova_p)):
        bar.set_color(sig_color if p < 0.05 else nsig_color)
        bar.set_alpha(alpha_sig if p < 0.05 else alpha_nsig)
    
    ax2.axhline(y=0.05, color='black', linestyle='--', linewidth=1, label='α = 0.05')
    
    for i, (bar, p) in enumerate(zip(bars2, seller_race_anova_p)):
        height = bar.get_height()
        label = f'p={p:.4f}'
        if p < 0.05:
            label += '\n***'
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold' if p < 0.05 else 'normal')
    
    ax2.set_ylabel('p-value', fontsize=11, fontweight='bold')
    ax2.set_title('Seller Race ANOVA Significance', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, max(seller_race_anova_p) * 1.2])
    ax2.set_yticks([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    ax2.tick_params(labelsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=9)
    
    # Plot 3: Summary table
    ax3 = fig.add_subplot(gs[1:, :])
    ax3.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Experiment', 'Buyer Race\nANOVA', 'Buyer Gender\nT-test', 
                      'Seller Race\nANOVA', 'Seller Gender\nT-test'])
    
    data_rows = [
        ('Low Context', low_context),
        ('High Context (House 1)', house1),
        ('High Context (House 2)', house2),
        ('High Context (House 3)', house3),
    ]
    
    for exp_name, data in data_rows:
        buyer_race_str = f"p={data['buyer_race_anova_p']:.4f}"
        if data['buyer_race_anova_sig']:
            buyer_race_str += " ✓"
        
        buyer_gender_str = f"p={data['buyer_gender_p']:.4f}"
        if data['buyer_gender_p'] and data['buyer_gender_p'] < 0.05:
            buyer_gender_str += " ✓"
        
        seller_race_str = f"p={data['seller_race_anova_p']:.4f}"
        if data['seller_race_anova_sig']:
            seller_race_str += " ✓"
        
        seller_gender_str = f"p={data['seller_gender_p']:.4f}"
        if data['seller_gender_p'] and data['seller_gender_p'] < 0.05:
            seller_gender_str += " ✓"
        
        table_data.append([exp_name, buyer_race_str, buyer_gender_str, seller_race_str, seller_gender_str])
    
    # Create table
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Color header row
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#424242')
        cell.set_text_props(weight='bold', color='white')
        # Add extra padding to header cells by increasing cell height
        cell.set_height(cell.get_height() * 1.3)  # Make header row 30% taller
    
    # Color significant cells
    for row_idx, (exp_name, data) in enumerate(data_rows, 1):
        # Buyer Race ANOVA
        if data['buyer_race_anova_sig']:
            table[(row_idx, 1)].set_facecolor('#c8e6c9')  # Light green
            table[(row_idx, 1)].set_text_props(weight='bold')
        
        # Buyer Gender
        if data['buyer_gender_p'] and data['buyer_gender_p'] < 0.05:
            table[(row_idx, 2)].set_facecolor('#c8e6c9')
            table[(row_idx, 2)].set_text_props(weight='bold')
        
        # Seller Race ANOVA
        if data['seller_race_anova_sig']:
            table[(row_idx, 3)].set_facecolor('#c8e6c9')
            table[(row_idx, 3)].set_text_props(weight='bold')
        
        # Seller Gender
        if data['seller_gender_p'] and data['seller_gender_p'] < 0.05:
            table[(row_idx, 4)].set_facecolor('#c8e6c9')
            table[(row_idx, 4)].set_text_props(weight='bold')
    
    ax3.set_title('Statistical Significance Summary (α = 0.05)', fontsize=14, fontweight='bold', pad=10, y = 0.72)
    
    # Add overall title
    fig.suptitle('Statistical Significance Comparison: Low Context vs High Context Experiments', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path('results/experiment1/significance_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    
    # Also create a simpler bar plot focusing on Race ANOVA
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Buyer Race ANOVA comparison
    x = np.arange(len(experiments))
    width = 0.6
    bars3 = ax4.bar(x, buyer_race_anova_p, width, edgecolor='black', linewidth=1.5)
    for i, (bar, p) in enumerate(zip(bars3, buyer_race_anova_p)):
        bar.set_color(sig_color if p < 0.05 else nsig_color)
        bar.set_alpha(alpha_sig if p < 0.05 else alpha_nsig)
    ax4.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='α = 0.05')
    
    for i, (bar, p, sig) in enumerate(zip(bars3, buyer_race_anova_p, 
                                           [low_context['buyer_race_anova_sig'], 
                                            house1['buyer_race_anova_sig'],
                                            house2['buyer_race_anova_sig'],
                                            house3['buyer_race_anova_sig']])):
        height = bar.get_height()
        label = f'p={p:.4f}'
        if sig:
            if p < 0.001:
                label += '\n*** SIGNIFICANT'
            elif p < 0.01:
                label += '\n** SIGNIFICANT'
            else:
                label += '\n* SIGNIFICANT'
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold' if sig else 'normal')
    
    ax4.set_ylabel('p-value', fontsize=13, fontweight='bold')
    ax4.set_title('Buyer Race ANOVA: Low Context vs High Context', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(experiments, fontsize=11)
    ax4.set_ylim([0, max(buyer_race_anova_p) * 1.3])
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.legend(fontsize=11)
    
    # Seller Race ANOVA comparison
    bars4 = ax5.bar(x, seller_race_anova_p, width, edgecolor='black', linewidth=1.5)
    for i, (bar, p) in enumerate(zip(bars4, seller_race_anova_p)):
        bar.set_color(sig_color if p < 0.05 else nsig_color)
        bar.set_alpha(alpha_sig if p < 0.05 else alpha_nsig)
    ax5.axhline(y=0.05, color='black', linestyle='--', linewidth=2, label='α = 0.05')
    
    for i, (bar, p, sig) in enumerate(zip(bars4, seller_race_anova_p,
                                           [low_context['seller_race_anova_sig'],
                                            house1['seller_race_anova_sig'],
                                            house2['seller_race_anova_sig'],
                                            house3['seller_race_anova_sig']])):
        height = bar.get_height()
        label = f'p={p:.4f}'
        if sig:
            label += '\n*** SIGNIFICANT'
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold' if sig else 'normal')
    
    ax5.set_ylabel('p-value', fontsize=13, fontweight='bold')
    ax5.set_title('Seller Race ANOVA: Low Context vs High Context', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(experiments, fontsize=11)
    ax5.set_ylim([0, max(seller_race_anova_p) * 1.3])
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.legend(fontsize=11)
    
    plt.suptitle('Race ANOVA Significance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path2 = Path('results/experiment1/race_anova_comparison.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Race ANOVA comparison plot saved to: {output_path2}")
    
    return output_path, output_path2


if __name__ == "__main__":
    create_comparison_visualization()

