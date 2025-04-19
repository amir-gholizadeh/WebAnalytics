# scripts/conversion_analysis.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

def analyze_traffic_conversion_relationship(df):
    """
    Analyze relationship between traffic volume and conversion rates.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        matplotlib Figure
    """
    # Group by date to analyze daily patterns
    daily_data = df.groupby('Date').agg({
        'Sessions': 'sum',
        'Users': 'sum',
        'Transactions': 'sum',
        'Conversion Rate (%)': 'mean'
    }).reset_index()

    # Calculate additional metrics
    daily_data['Sessions per User'] = daily_data['Sessions'] / daily_data['Users']

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Sessions vs Conversion Rate
    axes[0].scatter(daily_data['Sessions'], daily_data['Conversion Rate (%)'],
                   alpha=0.7, s=50, c='blue')

    # Add trendline
    z = np.polyfit(daily_data['Sessions'], daily_data['Conversion Rate (%)'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(daily_data['Sessions'].min(), daily_data['Sessions'].max(), 100)
    axes[0].plot(x_range, p(x_range), '--', color='red', alpha=0.8)

    # Calculate correlation
    r, p_value = pearsonr(daily_data['Sessions'], daily_data['Conversion Rate (%)'])
    axes[0].annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    axes[0].set_title('Traffic Volume vs. Conversion Rate')
    axes[0].set_xlabel('Sessions')
    axes[0].set_ylabel('Conversion Rate (%)')
    axes[0].grid(True, alpha=0.3)

    # Sessions per User vs Conversion Rate
    axes[1].scatter(daily_data['Sessions per User'], daily_data['Conversion Rate (%)'],
                   alpha=0.7, s=50, c='green')

    # Add trendline
    z = np.polyfit(daily_data['Sessions per User'], daily_data['Conversion Rate (%)'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(daily_data['Sessions per User'].min(), daily_data['Sessions per User'].max(), 100)
    axes[1].plot(x_range, p(x_range), '--', color='red', alpha=0.8)

    # Calculate correlation
    r, p_value = pearsonr(daily_data['Sessions per User'], daily_data['Conversion Rate (%)'])
    axes[1].annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    axes[1].set_title('Sessions per User vs. Conversion Rate')
    axes[1].set_xlabel('Sessions per User')
    axes[1].set_ylabel('Conversion Rate (%)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return plt

def source_revenue_analysis(df):
    """
    Analyze which sources lead to the highest revenue through various metrics.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        matplotlib Figure
    """
    # Group data by source
    source_data = df.groupby('Source / Medium').agg({
        'Sessions': 'sum',
        'Users': 'sum',
        'Transactions': 'sum',
        'Revenue': 'sum',
        'Conversion Rate (%)': 'mean'
    }).reset_index()

    # Calculate derived metrics
    source_data['Revenue per Transaction'] = source_data['Revenue'] / source_data['Transactions']
    source_data['Revenue per Session'] = source_data['Revenue'] / source_data['Sessions']
    source_data['Transactions per Session'] = source_data['Transactions'] / source_data['Sessions']

    # Sort by revenue
    top_sources = source_data.sort_values('Revenue', ascending=False).head(10)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Revenue vs Transactions per Session
    axes[0].scatter(top_sources['Transactions per Session'], top_sources['Revenue'],
                   s=top_sources['Sessions']/100, alpha=0.7)

    # Add source labels
    for i, row in top_sources.iterrows():
        axes[0].annotate(row['Source / Medium'],
                        (row['Transactions per Session'], row['Revenue']),
                        fontsize=9)

    axes[0].set_title('Revenue vs. Transactions per Session')
    axes[0].set_xlabel('Transactions per Session')
    axes[0].set_ylabel('Total Revenue')
    axes[0].grid(True, alpha=0.3)

    # Revenue per Transaction vs Conversion Rate
    axes[1].scatter(top_sources['Conversion Rate (%)'], top_sources['Revenue per Transaction'],
                   s=top_sources['Transactions']/10, alpha=0.7)

    # Add source labels
    for i, row in top_sources.iterrows():
        axes[1].annotate(row['Source / Medium'],
                        (row['Conversion Rate (%)'], row['Revenue per Transaction']),
                        fontsize=9)

    axes[1].set_title('Revenue per Transaction vs. Conversion Rate')
    axes[1].set_xlabel('Conversion Rate (%)')
    axes[1].set_ylabel('Revenue per Transaction')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return plt