"""Visualization utilities for experiment results"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def plot_experiment1_results(
    csv_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create a bar plot with error bars for Experiment 1 results
    
    Args:
        csv_path: Path to Experiment 1 CSV results file
        output_path: Optional path to save the plot (default: same directory as CSV)
        title: Optional plot title
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Create race+gender combination code
    def get_demographic(row):
        if pd.notna(row.get('buyer_race')):
            race = str(row['buyer_race'])
            gender = str(row['buyer_gender'])
        else:
            race = str(row['seller_race'])
            gender = str(row['seller_gender'])
        
        # Map to codes: WM, WW, BM, BW, HM, HW, AM, AW
        if race == 'Black' and gender == 'Female':
            return 'BW'  # Black Women
        elif race == 'Black' and gender == 'Male':
            return 'BM'  # Black Men
        elif race == 'White' and gender == 'Male':
            return 'WM'  # White Men
        elif race == 'White' and gender == 'Female':
            return 'WW'  # White Women
        elif race == 'Hispanic' and gender == 'Female':
            return 'HW'  # Hispanic Women
        elif race == 'Hispanic' and gender == 'Male':
            return 'HM'  # Hispanic Men
        elif race == 'Asian' and gender == 'Male':
            return 'AM'  # Asian Men
        elif race == 'Asian' and gender == 'Female':
            return 'AW'  # Asian Women
        else:
            # Fallback - use first letters
            return f"{race[0]}{gender[0]}"
    
    df['demographic'] = df.apply(get_demographic, axis=1)
    
    # Calculate statistics by recommendation type and demographic
    buyers_df = df[df['recommendation_type'] == 'buyer'].copy()
    sellers_df = df[df['recommendation_type'] == 'seller'].copy()
    
    # Group and calculate stats
    buyer_stats = buyers_df.groupby('demographic')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    buyer_stats['se'] = buyer_stats['std'] / np.sqrt(buyer_stats['count'])  # Standard error
    
    seller_stats = sellers_df.groupby('demographic')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    seller_stats['se'] = seller_stats['std'] / np.sqrt(seller_stats['count'])  # Standard error
    
    # Define order: WM, WW, BM, BW, HM, HW, AM, AW (all 8 categories)
    demographic_order = ['WM', 'WW', 'BM', 'BW', 'HM', 'HW', 'AM', 'AW']
    demographic_labels = [
        'White Men\n(WM)', 'White Women\n(WW)',
        'Black Men\n(BM)', 'Black Women\n(BW)',
        'Hispanic Men\n(HM)', 'Hispanic Women\n(HW)',
        'Asian Men\n(AM)', 'Asian Women\n(AW)'
    ]
    
    # Reorder data to match demographic_order
    buyer_ordered = buyer_stats.set_index('demographic').reindex(demographic_order).reset_index()
    seller_ordered = seller_stats.set_index('demographic').reindex(demographic_order).reset_index()
    
    # Create figure with two subplots (wider to accommodate 8 categories)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Set up x positions
    x = np.arange(len(demographic_order))
    width = 0.6  # Width of bars
    
    # Subplot 1: Buyers
    buyer_bars = ax1.bar(
        x,
        buyer_ordered['mean'],
        width,
        yerr=buyer_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#2E86AB',
        edgecolor='black',
        linewidth=1
    )
    
    ax1.set_xlabel('Demographic Group', fontsize=12)
    ax1.set_ylabel('Recommended Price ($)', fontsize=12)
    ax1.set_title('Buyer Price Recommendations', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(demographic_labels)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels on buyer bars
    for i, bar in enumerate(buyer_bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + buyer_ordered.iloc[i]['se'],
            f'${height:,.0f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Subplot 2: Sellers
    seller_bars = ax2.bar(
        x,
        seller_ordered['mean'],
        width,
        yerr=seller_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#A23B72',
        edgecolor='black',
        linewidth=1
    )
    
    ax2.set_xlabel('Demographic Group', fontsize=12)
    ax2.set_ylabel('Recommended Price ($)', fontsize=12)
    ax2.set_title('Seller Price Recommendations', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(demographic_labels)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels on seller bars
    for i, bar in enumerate(seller_bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + seller_ordered.iloc[i]['se'],
            f'${height:,.0f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_plot.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Only show if interactive backend is available
    import matplotlib
    backend = matplotlib.get_backend()
    if backend.lower() != 'agg':
        plt.show()
    else:
        plt.close(fig)  # Close to free memory in batch mode
    
    return fig, (ax1, ax2)


def plot_experiment1_by_race(
    csv_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create a bar plot aggregated by race only (ignoring gender)
    
    Args:
        csv_path: Path to Experiment 1 CSV results file
        output_path: Optional path to save the plot (default: same directory as CSV)
        title: Optional plot title
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract race
    def get_race(row):
        if pd.notna(row.get('buyer_race')):
            return str(row['buyer_race'])
        else:
            return str(row['seller_race'])
    
    df['race'] = df.apply(get_race, axis=1)
    
    # Calculate statistics by recommendation type and race
    buyers_df = df[df['recommendation_type'] == 'buyer'].copy()
    sellers_df = df[df['recommendation_type'] == 'seller'].copy()
    
    # Group and calculate stats
    buyer_stats = buyers_df.groupby('race')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    buyer_stats['se'] = buyer_stats['std'] / np.sqrt(buyer_stats['count'])  # Standard error
    
    seller_stats = sellers_df.groupby('race')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    seller_stats['se'] = seller_stats['std'] / np.sqrt(seller_stats['count'])  # Standard error
    
    # Define order: White, Black, Hispanic, Asian
    race_order = ['White', 'Black', 'Hispanic', 'Asian']
    
    # Reorder data
    buyer_ordered = buyer_stats.set_index('race').reindex(race_order).reset_index()
    seller_ordered = seller_stats.set_index('race').reindex(race_order).reset_index()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Set up x positions
    x = np.arange(len(race_order))
    width = 0.6
    
    # Subplot 1: Buyers
    buyer_bars = ax1.bar(
        x,
        buyer_ordered['mean'],
        width,
        yerr=buyer_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#2E86AB',
        edgecolor='black',
        linewidth=1
    )
    
    ax1.set_xlabel('Race', fontsize=12)
    ax1.set_ylabel('Recommended Price ($)', fontsize=12)
    ax1.set_title('Buyer Price Recommendations (by Race)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(race_order)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels
    for i, bar in enumerate(buyer_bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + buyer_ordered.iloc[i]['se'],
            f'${height:,.0f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Subplot 2: Sellers
    seller_bars = ax2.bar(
        x,
        seller_ordered['mean'],
        width,
        yerr=seller_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#A23B72',
        edgecolor='black',
        linewidth=1
    )
    
    ax2.set_xlabel('Race', fontsize=12)
    ax2.set_ylabel('Recommended Price ($)', fontsize=12)
    ax2.set_title('Seller Price Recommendations (by Race)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(race_order)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels
    for i, bar in enumerate(seller_bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + seller_ordered.iloc[i]['se'],
            f'${height:,.0f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Price Recommendations Aggregated by Race', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_by_race.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Only show if interactive backend is available
    import matplotlib
    backend = matplotlib.get_backend()
    if backend.lower() != 'agg':
        plt.show()
    else:
        plt.close(fig)  # Close to free memory in batch mode
    
    return fig, (ax1, ax2)


def plot_experiment1_by_gender(
    csv_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create a bar plot aggregated by gender only (ignoring race)
    
    Args:
        csv_path: Path to Experiment 1 CSV results file
        output_path: Optional path to save the plot (default: same directory as CSV)
        title: Optional plot title
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Extract gender
    def get_gender(row):
        if pd.notna(row.get('buyer_gender')):
            return str(row['buyer_gender'])
        else:
            return str(row['seller_gender'])
    
    df['gender'] = df.apply(get_gender, axis=1)
    
    # Calculate statistics by recommendation type and gender
    buyers_df = df[df['recommendation_type'] == 'buyer'].copy()
    sellers_df = df[df['recommendation_type'] == 'seller'].copy()
    
    # Group and calculate stats
    buyer_stats = buyers_df.groupby('gender')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    buyer_stats['se'] = buyer_stats['std'] / np.sqrt(buyer_stats['count'])  # Standard error
    
    seller_stats = sellers_df.groupby('gender')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    seller_stats['se'] = seller_stats['std'] / np.sqrt(seller_stats['count'])  # Standard error
    
    # Define order: Male, Female
    gender_order = ['Male', 'Female']
    
    # Reorder data
    buyer_ordered = buyer_stats.set_index('gender').reindex(gender_order).reset_index()
    seller_ordered = seller_stats.set_index('gender').reindex(gender_order).reset_index()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Set up x positions
    x = np.arange(len(gender_order))
    width = 0.6
    
    # Subplot 1: Buyers
    buyer_bars = ax1.bar(
        x,
        buyer_ordered['mean'],
        width,
        yerr=buyer_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#2E86AB',
        edgecolor='black',
        linewidth=1
    )
    
    ax1.set_xlabel('Gender', fontsize=12)
    ax1.set_ylabel('Recommended Price ($)', fontsize=12)
    ax1.set_title('Buyer Price Recommendations (by Gender)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gender_order)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels
    for i, bar in enumerate(buyer_bars):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + buyer_ordered.iloc[i]['se'],
            f'${height:,.0f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Subplot 2: Sellers
    seller_bars = ax2.bar(
        x,
        seller_ordered['mean'],
        width,
        yerr=seller_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#A23B72',
        edgecolor='black',
        linewidth=1
    )
    
    ax2.set_xlabel('Gender', fontsize=12)
    ax2.set_ylabel('Recommended Price ($)', fontsize=12)
    ax2.set_title('Seller Price Recommendations (by Gender)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gender_order)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels
    for i, bar in enumerate(seller_bars):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height + seller_ordered.iloc[i]['se'],
            f'${height:,.0f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Price Recommendations Aggregated by Gender', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_by_gender.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Only show if interactive backend is available
    import matplotlib
    backend = matplotlib.get_backend()
    if backend.lower() != 'agg':
        plt.show()
    else:
        plt.close(fig)  # Close to free memory in batch mode
    
    return fig, (ax1, ax2)


def plot_experiment1_by_house(
    csv_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Create a bar plot aggregated by house (for experiments with house details)
    
    Args:
        csv_path: Path to Experiment 1 CSV results file
        output_path: Optional path to save the plot (default: same directory as CSV)
        title: Optional plot title
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Check if house data is available
    if 'house_address' not in df.columns or df['house_address'].isna().all():
        print("Warning: No house data found in CSV. Skipping house aggregation plot.")
        return None, None
    
    # Filter out rows without house data
    df = df[df['house_address'].notna()].copy()
    
    # Create a house identifier (use address or city+state)
    df['house_id'] = df['house_address'].fillna(df['house_city'].astype(str) + ', ' + df['house_state'].astype(str))
    
    # Calculate statistics by recommendation type and house
    buyers_df = df[df['recommendation_type'] == 'buyer'].copy()
    sellers_df = df[df['recommendation_type'] == 'seller'].copy()
    
    # Group and calculate stats
    buyer_stats = buyers_df.groupby('house_id')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    buyer_stats['se'] = buyer_stats['std'] / np.sqrt(buyer_stats['count'])
    
    seller_stats = sellers_df.groupby('house_id')['recommended_price'].agg(['mean', 'std', 'count']).reset_index()
    seller_stats['se'] = seller_stats['std'] / np.sqrt(seller_stats['count'])
    
    # Sort by house price (if available) or address
    if 'house_price' in df.columns:
        house_order = df.groupby('house_id')['house_price'].first().sort_values().index.tolist()
    else:
        house_order = sorted(df['house_id'].unique())
    
    # Reorder data
    buyer_ordered = buyer_stats.set_index('house_id').reindex(house_order).reset_index()
    seller_ordered = seller_stats.set_index('house_id').reindex(house_order).reset_index()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Set up x positions
    x = np.arange(len(house_order))
    width = 0.6
    
    # Subplot 1: Buyers
    buyer_bars = ax1.bar(
        x,
        buyer_ordered['mean'],
        width,
        yerr=buyer_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#2E86AB',
        edgecolor='black',
        linewidth=1
    )
    
    ax1.set_xlabel('House', fontsize=12)
    ax1.set_ylabel('Recommended Price ($)', fontsize=12)
    ax1.set_title('Buyer Price Recommendations (by House)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    # Truncate house addresses for readability
    house_labels = [addr[:30] + '...' if len(addr) > 30 else addr for addr in house_order]
    ax1.set_xticklabels(house_labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels
    for i, bar in enumerate(buyer_bars):
        if pd.notna(buyer_ordered.iloc[i]['mean']):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2.,
                height + buyer_ordered.iloc[i]['se'],
                f'${height:,.0f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    # Subplot 2: Sellers
    seller_bars = ax2.bar(
        x,
        seller_ordered['mean'],
        width,
        yerr=seller_ordered['se'],
        capsize=5,
        alpha=0.8,
        color='#A23B72',
        edgecolor='black',
        linewidth=1
    )
    
    ax2.set_xlabel('House', fontsize=12)
    ax2.set_ylabel('Recommended Price ($)', fontsize=12)
    ax2.set_title('Seller Price Recommendations (by House)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(house_labels, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Add value labels
    for i, bar in enumerate(seller_bars):
        if pd.notna(seller_ordered.iloc[i]['mean']):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2.,
                height + seller_ordered.iloc[i]['se'],
                f'${height:,.0f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Price Recommendations Aggregated by House', fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_by_house.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Only show if interactive backend is available
    import matplotlib
    backend = matplotlib.get_backend()
    if backend.lower() != 'agg':
        plt.show()
    else:
        plt.close(fig)  # Close to free memory in batch mode
    
    return fig, (ax1, ax2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        plot_experiment1_results(csv_path)
    else:
        print("Usage: python -m src.analysis.visualization <path_to_csv>")

