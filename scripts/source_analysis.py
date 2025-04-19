# scripts/time_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def source_performance_metrics(df, metrics=None):
    """
    Calculate and compare key performance metrics across different traffic sources.

    Args:
        df: DataFrame with cleaned analytics data
        metrics: List of metrics to analyze (default: all available metrics)

    Returns:
        DataFrame with aggregated metrics by source
    """
    if metrics is None:
        metrics = ['Users', 'Sessions', 'Transactions', 'Revenue', 'Conversion Rate (%)']

    # Create a copy of the dataframe
    source_df = df.copy()

    # Group by Source/Medium and aggregate metrics
    source_metrics = source_df.groupby('Source / Medium').agg({
        'Users': 'sum',
        'Sessions': 'sum',
        'Transactions': 'sum',
        'Revenue': 'sum'
    }).reset_index()

    # Calculate derived metrics
    source_metrics['Revenue per User'] = source_metrics['Revenue'] / source_metrics['Users']
    source_metrics['Conversion Rate'] = source_metrics['Transactions'] / source_metrics['Users'] * 100
    source_metrics['Revenue per Transaction'] = source_metrics['Revenue'] / source_metrics['Transactions']

    # Sort by revenue (default) or other metrics
    source_metrics = source_metrics.sort_values('Revenue', ascending=False)

    return source_metrics

def plot_source_performance(source_metrics, metric='Revenue', top_n=10):
    """
    Create a bar chart showing the performance of top sources by a specific metric.

    Args:
        source_metrics: DataFrame with source metrics from source_performance_metrics
        metric: Metric to plot (default: Revenue)
        top_n: Number of top sources to display (default: 10)

    Returns:
        matplotlib Figure
    """
    # Get top N sources
    top_sources = source_metrics.head(top_n).copy()

    # Sort for display
    top_sources = top_sources.sort_values(metric)

    plt.figure(figsize=(12, 8))

    # Create horizontal bar chart
    bars = plt.barh(top_sources['Source / Medium'], top_sources[metric])

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if metric in ['Revenue', 'Revenue per Transaction', 'Revenue per User']:
            label = f'${width:,.2f}'
        elif metric in ['Conversion Rate']:
            label = f'{width:.2f}%'
        else:
            label = f'{width:,.0f}'
        plt.text(width, bar.get_y() + bar.get_height()/2, label, va='center')

    plt.title(f'Top {top_n} Sources by {metric}')
    plt.xlabel(metric)
    plt.ylabel('Source / Medium')
    plt.tight_layout()

    return plt

def source_metrics_heatmap(source_metrics, metrics=None, top_n=15):
    """
    Create a heatmap showing multiple metrics across top sources.

    Args:
        source_metrics: DataFrame with source metrics
        metrics: List of metrics to include in heatmap
        top_n: Number of top sources to include

    Returns:
        matplotlib Figure
    """
    if metrics is None:
        metrics = ['Users', 'Transactions', 'Revenue', 'Conversion Rate', 'Revenue per User']

    # Get top sources by revenue
    top_sources = source_metrics.nlargest(top_n, 'Revenue')

    # Select only the metrics we want to display
    heatmap_data = top_sources[['Source / Medium'] + metrics].set_index('Source / Medium')

    # Normalize data for better visualization
    normalized_data = heatmap_data.copy()
    for col in normalized_data.columns:
        normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / \
                              (normalized_data[col].max() - normalized_data[col].min())

    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_data, annot=heatmap_data, fmt='.2f', cmap='YlGnBu',
                linewidths=.5, cbar_kws={'label': 'Normalized Value'})

    plt.title('Performance Metrics by Source (normalized values with actual annotations)')
    plt.tight_layout()

    return plt

def source_trend_analysis(df, source, metric='Revenue'):
    """
    Analyze how a specific source's performance has changed over time.

    Args:
        df: DataFrame with cleaned analytics data
        source: Source/Medium to analyze
        metric: Metric to analyze

    Returns:
        matplotlib Figure
    """
    # Filter for the specific source
    source_data = df[df['Source / Medium'] == source].copy()

    # Group by month and calculate the metric
    monthly_trend = source_data.groupby('Date')[metric].sum().reset_index()

    plt.figure(figsize=(14, 6))
    plt.plot(monthly_trend['Date'], monthly_trend[metric], marker='o', linestyle='-')

    plt.title(f'{metric} Trend for {source}')
    plt.xlabel('Month')
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return plt