#!/usr/bin/env python3
"""
Reusable script to run t-tests on Experiment 1 CSV results files.

Usage:
    python -m src.analysis.run_tests <path_to_csv>
    python src/analysis/run_tests.py results/experiment1_20251124_184322.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from scipy import stats
from itertools import combinations


def run_t_tests_on_csv(csv_path: str, significance_level: float = 0.05):
    """
    Run t-tests on Experiment 1 CSV results file.
    
    Args:
        csv_path: Path to CSV results file
        significance_level: Significance level for tests (default 0.05)
    
    Returns:
        Dictionary with test results
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Separate buyers and sellers
    buyers_df = df[df['recommendation_type'] == 'buyer'].copy()
    sellers_df = df[df['recommendation_type'] == 'seller'].copy()
    
    results = {
        'csv_path': csv_path,
        'total_trials': len(df),
        'buyer_trials': len(buyers_df),
        'seller_trials': len(sellers_df),
        'significance_level': significance_level,
        'buyers': {},
        'sellers': {}
    }
    
    # Helper function to run t-tests
    def run_t_tests_for_role(role_df, role_name):
        """Run t-tests for gender and race groups"""
        role_results = {}
        
        # Gender t-test
        male_data = role_df[role_df[f'{role_name}_gender'] == 'Male']['recommended_price'].values
        female_data = role_df[role_df[f'{role_name}_gender'] == 'Female']['recommended_price'].values
        
        if len(male_data) > 0 and len(female_data) > 0:
            t_stat, p_value = stats.ttest_ind(male_data, female_data)
            pooled_std = np.sqrt(((len(male_data) - 1) * np.var(male_data, ddof=1) + 
                                 (len(female_data) - 1) * np.var(female_data, ddof=1)) / 
                                (len(male_data) + len(female_data) - 2))
            cohens_d = (np.mean(male_data) - np.mean(female_data)) / pooled_std if pooled_std > 0 else 0
            
            role_results['gender'] = {
                'group1': 'Male',
                'group2': 'Female',
                'group1_mean': float(np.mean(male_data)),
                'group2_mean': float(np.mean(female_data)),
                'group1_std': float(np.std(male_data)),
                'group2_std': float(np.std(female_data)),
                'group1_n': len(male_data),
                'group2_n': len(female_data),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < significance_level,
                'cohens_d': float(cohens_d),
                'effect_size': interpret_effect_size(abs(cohens_d))
            }
        
        # Race pairwise t-tests
        race_groups = {}
        for race in ['White', 'Black', 'Hispanic', 'Asian']:
            race_data = role_df[role_df[f'{role_name}_race'] == race]['recommended_price'].values
            if len(race_data) > 0:
                race_groups[race] = race_data
        
        race_comparisons = []
        for race1, race2 in combinations(race_groups.keys(), 2):
            data1 = race_groups[race1]
            data2 = race_groups[race2]
            
            t_stat, p_value = stats.ttest_ind(data1, data2)
            pooled_std = np.sqrt(((len(data1) - 1) * np.var(data1, ddof=1) + 
                                 (len(data2) - 1) * np.var(data2, ddof=1)) / 
                                (len(data1) + len(data2) - 2))
            cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std if pooled_std > 0 else 0
            
            race_comparisons.append({
                'group1': race1,
                'group2': race2,
                'group1_mean': float(np.mean(data1)),
                'group2_mean': float(np.mean(data2)),
                'group1_std': float(np.std(data1)),
                'group2_std': float(np.std(data2)),
                'group1_n': len(data1),
                'group2_n': len(data2),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < significance_level,
                'cohens_d': float(cohens_d),
                'effect_size': interpret_effect_size(abs(cohens_d))
            })
        
        role_results['race'] = race_comparisons
        
        # ANOVA for race
        if len(race_groups) > 2:
            race_data_list = list(race_groups.values())
            f_stat, p_value_anova = stats.f_oneway(*race_data_list)
            all_data = np.concatenate(race_data_list)
            grand_mean = np.mean(all_data)
            ss_between = sum(len(group) * (np.mean(group) - grand_mean) ** 2 for group in race_data_list)
            ss_total = np.sum((all_data - grand_mean) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            role_results['race_anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value_anova),
                'significant': p_value_anova < significance_level,
                'eta_squared': float(eta_squared),
                'effect_size': interpret_effect_size(np.sqrt(eta_squared))
            }
        
        return role_results
    
    # Run tests for buyers and sellers
    results['buyers'] = run_t_tests_for_role(buyers_df, 'buyer')
    results['sellers'] = run_t_tests_for_role(sellers_df, 'seller')
    
    return results


def interpret_effect_size(effect_size: float) -> str:
    """Interpret effect size"""
    if effect_size < 0.2:
        return "negligible"
    elif effect_size < 0.5:
        return "small"
    elif effect_size < 0.8:
        return "medium"
    else:
        return "large"


def print_results(results: dict):
    """Print formatted t-test results"""
    print("="*80)
    print("FORMAL T-TEST ANALYSIS")
    print("="*80)
    print(f"CSV file: {results['csv_path']}")
    print(f"Total trials: {results['total_trials']}")
    print(f"Buyer trials: {results['buyer_trials']}")
    print(f"Seller trials: {results['seller_trials']}")
    print(f"Significance level: α = {results['significance_level']}")
    print()
    
    # Print buyer results
    print("="*80)
    print("BUYER RECOMMENDATIONS")
    print("="*80)
    print_role_results(results['buyers'], "BUYER")
    
    # Print seller results
    print("\n" + "="*80)
    print("SELLER RECOMMENDATIONS")
    print("="*80)
    print_role_results(results['sellers'], "SELLER")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nSignificant differences (p < 0.05):")
    print()
    
    buyer_sig = [c for c in results['buyers'].get('race', []) if c['significant']]
    seller_sig = [c for c in results['sellers'].get('race', []) if c['significant']]
    
    print("BUYERS:")
    if results['buyers'].get('gender', {}).get('significant', False):
        print("  ✓ Gender: Male vs Female")
    else:
        print("  ✗ Gender: No significant difference")
    if results['buyers'].get('race_anova', {}).get('significant', False):
        print("  ✓ Race: Overall ANOVA significant")
        if buyer_sig:
            print(f"    Significant pairwise comparisons:")
            for comp in buyer_sig:
                print(f"      - {comp['group1']} vs {comp['group2']} (p={comp['p_value']:.4f}, d={comp['cohens_d']:.3f})")
    else:
        if buyer_sig:
            print("  ⚠ Race: No overall ANOVA significance, but pairwise differences found:")
            for comp in buyer_sig:
                print(f"      - {comp['group1']} vs {comp['group2']} (p={comp['p_value']:.4f}, d={comp['cohens_d']:.3f})")
        else:
            print("  ✗ Race: No significant differences")
    
    print("\nSELLERS:")
    if results['sellers'].get('gender', {}).get('significant', False):
        print("  ✓ Gender: Male vs Female")
    else:
        print("  ✗ Gender: No significant difference")
    if results['sellers'].get('race_anova', {}).get('significant', False):
        print("  ✓ Race: Overall ANOVA significant")
        if seller_sig:
            print(f"    Significant pairwise comparisons:")
            for comp in seller_sig:
                print(f"      - {comp['group1']} vs {comp['group2']} (p={comp['p_value']:.4f}, d={comp['cohens_d']:.3f})")
    else:
        if seller_sig:
            print("  ⚠ Race: No overall ANOVA significance, but pairwise differences found:")
            for comp in seller_sig:
                print(f"      - {comp['group1']} vs {comp['group2']} (p={comp['p_value']:.4f}, d={comp['cohens_d']:.3f})")
        else:
            print("  ✗ Race: No significant differences")


def print_role_results(role_results: dict, role_name: str):
    """Print results for a single role (buyer or seller)"""
    # Gender test
    if 'gender' in role_results:
        g = role_results['gender']
        print("\nGENDER T-TEST:")
        print("-" * 80)
        print(f"Male   (n={g['group1_n']:3d}): ${g['group1_mean']:,.0f} ± ${g['group1_std']:,.0f}")
        print(f"Female (n={g['group2_n']:3d}): ${g['group2_mean']:,.0f} ± ${g['group2_std']:,.0f}")
        print()
        print(f"t-statistic: {g['t_statistic']:7.3f}")
        print(f"p-value:     {g['p_value']:7.4f}")
        print(f"Cohen's d:   {g['cohens_d']:7.3f} ({g['effect_size']})")
        sig = "*** SIGNIFICANT" if g['significant'] else "not significant"
        print(f"Result:      {sig}")
    
    # Race pairwise tests
    if 'race' in role_results:
        print("\nRACE PAIRWISE T-TESTS:")
        print("-" * 80)
        print(f"{'Comparison':<25} {'Mean1':<12} {'Mean2':<12} {'t-stat':<10} {'p-value':<12} {'Cohen\'s d':<12} {'Result'}")
        print("-" * 80)
        for comp in role_results['race']:
            sig = "***" if comp['significant'] else ""
            print(f"{comp['group1']:8s} vs {comp['group2']:8s}  "
                  f"${comp['group1_mean']:>9,.0f}  ${comp['group2_mean']:>9,.0f}  "
                  f"{comp['t_statistic']:>8.3f}  {comp['p_value']:>10.4f}  "
                  f"{comp['cohens_d']:>10.3f}  {sig}")
    
    # Race ANOVA
    if 'race_anova' in role_results:
        anova = role_results['race_anova']
        print("\nRACE ANOVA:")
        print("-" * 80)
        print(f"F-statistic: {anova['f_statistic']:.3f}")
        print(f"p-value:     {anova['p_value']:.4f}")
        print(f"Eta-squared: {anova['eta_squared']:.4f}")
        print(f"Effect size: {anova['effect_size']}")
        sig = "*** SIGNIFICANT" if anova['significant'] else "not significant"
        print(f"Result:      {sig}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python -m src.analysis.run_tests <path_to_csv> [significance_level]")
        print("Example: python -m src.analysis.run_tests results/experiment1_20251124_184322.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    significance_level = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Run tests
    results = run_t_tests_on_csv(csv_path, significance_level)
    
    # Print results
    print_results(results)
    
    return results


if __name__ == "__main__":
    main()

