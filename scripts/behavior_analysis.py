# scripts/behavior_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def engagement_metrics_over_time(df):
    """
    Plot engagement metrics (Bounce Rate, Avg Session Duration, Pageviews) over time.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        matplotlib Figure
    """
    # Prepare data
    metrics_df = df.copy()

    # Group by date
    engagement_data = metrics_df.groupby('Date').agg({
        'Bounce Rate': 'mean',
        'Avg. Session Duration': 'mean',
        'Pageviews': 'mean'
    }).reset_index()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Plot Bounce Rate
    axes[0].plot(engagement_data['Date'], engagement_data['Bounce Rate'],
                marker='o', linestyle='-', color='crimson')
    axes[0].set_title('Bounce Rate Over Time')
    axes[0].set_ylabel('Bounce Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, min(1, engagement_data['Bounce Rate'].max() * 1.2))

    # Plot Avg Session Duration
    axes[1].plot(engagement_data['Date'], engagement_data['Avg. Session Duration'],
                marker='o', linestyle='-', color='forestgreen')
    axes[1].set_title('Average Session Duration Over Time')
    axes[1].set_ylabel('Duration (seconds)')
    axes[1].grid(True, alpha=0.3)

    # Plot Pageviews
    axes[2].plot(engagement_data['Date'], engagement_data['Pageviews'],
                marker='o', linestyle='-', color='darkblue')
    axes[2].set_title('Average Pageviews Per Session Over Time')
    axes[2].set_ylabel('Pageviews')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)

    # Format x-axis dates
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return plt

def source_engagement_comparison(df, top_n=8):
    """
    Compare engagement metrics across top traffic sources.

    Args:
        df: DataFrame with cleaned analytics data
        top_n: Number of top sources to analyze

    Returns:
        matplotlib Figure
    """
    metrics_df = df.copy()

    # Get top sources by number of sessions
    top_sources = metrics_df.groupby('Source / Medium')['Sessions'].sum().nlargest(top_n).index

    # Filter for top sources
    filtered_df = metrics_df[metrics_df['Source / Medium'].isin(top_sources)]

    # Calculate average metrics by source
    source_metrics = filtered_df.groupby('Source / Medium').agg({
        'Bounce Rate': 'mean',
        'Avg. Session Duration': 'mean',
        'Pageviews': 'mean',
        'Conversion Rate (%)': 'mean'
    }).reset_index()

    # Sort by bounce rate for the chart
    source_metrics = source_metrics.sort_values('Bounce Rate')

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Bounce Rate
    axes[0, 0].barh(source_metrics['Source / Medium'], source_metrics['Bounce Rate'], color='crimson')
    axes[0, 0].set_title('Average Bounce Rate by Source')
    axes[0, 0].set_xlabel('Bounce Rate')
    axes[0, 0].set_xlim(0, 1)

    # Session Duration
    source_metrics_sorted = source_metrics.sort_values('Avg. Session Duration')
    axes[0, 1].barh(source_metrics_sorted['Source / Medium'], source_metrics_sorted['Avg. Session Duration'], color='forestgreen')
    axes[0, 1].set_title('Average Session Duration by Source')
    axes[0, 1].set_xlabel('Duration (seconds)')

    # Pageviews
    source_metrics_sorted = source_metrics.sort_values('Pageviews')
    axes[1, 0].barh(source_metrics_sorted['Source / Medium'], source_metrics_sorted['Pageviews'], color='darkblue')
    axes[1, 0].set_title('Average Pageviews by Source')
    axes[1, 0].set_xlabel('Pageviews')

    # Conversion Rate
    source_metrics_sorted = source_metrics.sort_values('Conversion Rate (%)')
    axes[1, 1].barh(source_metrics_sorted['Source / Medium'], source_metrics_sorted['Conversion Rate (%)'], color='purple')
    axes[1, 1].set_title('Average Conversion Rate by Source')
    axes[1, 1].set_xlabel('Conversion Rate (%)')

    plt.tight_layout()
    return plt

def correlation_matrix(df):
    """
    Create a correlation matrix of key metrics.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        matplotlib Figure
    """
    # Select numeric columns for correlation
    numeric_cols = ['Users', 'New Users', 'Sessions', 'Bounce Rate',
                   'Pageviews', 'Avg. Session Duration', 'Conversion Rate (%)',
                   'Transactions', 'Revenue']

    # Aggregate by date to avoid source-level correlations
    corr_data = df.groupby('Date')[numeric_cols].mean().reset_index()

    # Calculate correlation matrix
    corr_matrix = corr_data[numeric_cols].corr()

    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5)

    plt.title('Correlation Matrix of Key Metrics')
    plt.tight_layout()
    return plt

def scatter_plots(df, x_metric, y_metric, color_by='Source / Medium', top_n=5):
    """
    Create scatter plots to visualize relationships between metrics.

    Args:
        df: DataFrame with cleaned analytics data
        x_metric: Metric for x-axis
        y_metric: Metric for y-axis
        color_by: Column to use for coloring points
        top_n: Number of top sources to highlight by color

    Returns:
        matplotlib Figure
    """
    scatter_df = df.copy()

    # Get top sources by users
    top_sources = scatter_df.groupby(color_by)['Users'].sum().nlargest(top_n).index.tolist()

    # Create a new column for coloring
    scatter_df['Color Category'] = scatter_df[color_by].apply(
        lambda x: x if x in top_sources else 'Others'
    )

    plt.figure(figsize=(12, 8))

    # Create a colormap with distinct colors for each source
    categories = top_sources + ['Others']
    colors = plt.cm.tab10(range(len(categories)))
    color_dict = dict(zip(categories, colors))

    # Plot each category
    for category in categories:
        subset = scatter_df[scatter_df['Color Category'] == category]

        # Skip if empty
        if len(subset) == 0:
            continue

        alpha = 0.8 if category != 'Others' else 0.4
        size = 80 if category != 'Others' else 30

        plt.scatter(
            subset[x_metric],
            subset[y_metric],
            color=color_dict[category],
            alpha=alpha,
            s=size,
            label=category
        )

    plt.title(f'Relationship Between {x_metric} and {y_metric}')
    plt.xlabel(x_metric)
    plt.ylabel(y_metric)
    plt.grid(True, alpha=0.3)
    plt.legend(title=color_by)

    # Add trendline for all data
    all_data = scatter_df[[x_metric, y_metric]].dropna()
    if len(all_data) > 1:  # Need at least 2 points for a line
        z = np.polyfit(all_data[x_metric], all_data[y_metric], 1)
        p = np.poly1d(z)

        # Add trendline to plot
        x_range = np.linspace(all_data[x_metric].min(), all_data[x_metric].max(), 100)
        plt.plot(x_range, p(x_range), '--', color='black', alpha=0.8)

        # Calculate correlation
        r, p_value = pearsonr(all_data[x_metric], all_data[y_metric])
        plt.annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    return plt


def bounce_rate_analysis(df):
    """
    Analyze bounce rate relationships with conversion and revenue.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        matplotlib Figure with 2 subplots
    """
    # Group data by date and source
    bounce_data = df.groupby(['Date', 'Source / Medium']).agg({
        'Bounce Rate': 'mean',
        'Conversion Rate (%)': 'mean',
        'Revenue': 'sum',
        'Sessions': 'sum'
    }).reset_index()

    # Calculate revenue per session
    bounce_data['Revenue per Session'] = bounce_data['Revenue'] / bounce_data['Sessions']

    # Create figure with 2 subplots with more space for colorbar
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Bounce Rate vs Conversion Rate
    axes[0].scatter(bounce_data['Bounce Rate'], bounce_data['Conversion Rate (%)'],
                    alpha=0.6, c=bounce_data['Sessions'], cmap='viridis', s=50)

    # Add trendline
    z = np.polyfit(bounce_data['Bounce Rate'], bounce_data['Conversion Rate (%)'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(bounce_data['Bounce Rate'].min(), bounce_data['Bounce Rate'].max(), 100)
    axes[0].plot(x_range, p(x_range), '--', color='red', alpha=0.8)

    # Calculate correlation
    r, p_value = pearsonr(bounce_data['Bounce Rate'], bounce_data['Conversion Rate (%)'])
    axes[0].annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    axes[0].set_title('Bounce Rate vs. Conversion Rate')
    axes[0].set_xlabel('Bounce Rate')
    axes[0].set_ylabel('Conversion Rate (%)')
    axes[0].grid(True, alpha=0.3)

    # Bounce Rate vs Revenue per Session
    im = axes[1].scatter(bounce_data['Bounce Rate'], bounce_data['Revenue per Session'],
                         alpha=0.6, c=bounce_data['Sessions'], cmap='viridis', s=50)

    # Add trendline
    z = np.polyfit(bounce_data['Bounce Rate'], bounce_data['Revenue per Session'], 1)
    p = np.poly1d(z)
    axes[1].plot(x_range, p(x_range), '--', color='red', alpha=0.8)

    # Calculate correlation
    r, p_value = pearsonr(bounce_data['Bounce Rate'], bounce_data['Revenue per Session'])
    axes[1].annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    axes[1].set_title('Bounce Rate vs. Revenue per Session')
    axes[1].set_xlabel('Bounce Rate')
    axes[1].set_ylabel('Revenue per Session')
    axes[1].grid(True, alpha=0.3)

    # Add colorbar with proper positioning
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Number of Sessions')

    # Use subplots_adjust instead of tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.25)

    return fig