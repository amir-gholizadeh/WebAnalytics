# scripts/segmentation_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.cm as cm

def prepare_data_for_clustering(df):
    """
    Prepare and normalize data for clustering.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        Tuple of (normalized DataFrame for clustering, original DataFrame with segment)
    """
    # Group by Source/Medium to create source-level profiles
    segments_df = df.groupby('Source / Medium').agg({
        'Sessions': 'sum',
        'Users': 'sum',
        'New Users': 'sum',
        'Bounce Rate': 'mean',
        'Pageviews': 'sum',
        'Avg. Session Duration': 'mean',
        'Transactions': 'sum',
        'Revenue': 'sum',
        'Conversion Rate (%)': 'mean'
    }).reset_index()

    # Create derived metrics
    segments_df['Pages per Session'] = segments_df['Pageviews'] / segments_df['Sessions']
    segments_df['Revenue per User'] = segments_df['Revenue'] / segments_df['Users']
    segments_df['New User Ratio'] = segments_df['New Users'] / segments_df['Users']
    segments_df['Revenue per Transaction'] = segments_df['Revenue'] / segments_df['Transactions']

    # Select features for clustering
    features = [
        'Bounce Rate', 'Pages per Session', 'Avg. Session Duration',
        'Conversion Rate (%)', 'Revenue per User', 'New User Ratio',
        'Revenue per Transaction'
    ]

    # Create a copy for clustering
    cluster_df = segments_df.copy()

    # Handle NaN and infinite values
    for feature in features:
        cluster_df[feature] = cluster_df[feature].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN values
    cluster_df = cluster_df.dropna(subset=features)

    # Standardize the features
    scaler = StandardScaler()
    cluster_df[features] = scaler.fit_transform(cluster_df[features])

    return cluster_df, segments_df, features

def kmeans_clustering(cluster_df, segments_df, features, n_clusters=4):
    """
    Perform K-means clustering on traffic sources.

    Args:
        cluster_df: Normalized DataFrame for clustering
        segments_df: Original DataFrame
        features: List of features used for clustering
        n_clusters: Number of clusters

    Returns:
        DataFrame with cluster assignments
    """
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    # Get standardized features only
    X = cluster_df[features].values

    # Fit the model
    kmeans.fit(X)

    # Get cluster assignments
    cluster_df['Cluster'] = kmeans.labels_

    # Merge cluster assignments back to the original data
    result_df = segments_df.copy()
    result_df = result_df.merge(cluster_df[['Source / Medium', 'Cluster']],
                                on='Source / Medium', how='left')

    return result_df, kmeans

def visualize_clusters_2d(df, features, kmeans, title="2D Visualization of Clusters"):
    """
    Create a 2D visualization of clusters using PCA.

    Args:
        df: DataFrame with cluster assignments
        features: Features used for clustering
        kmeans: Fitted KMeans model
        title: Plot title

    Returns:
        matplotlib Figure
    """
    # Create a copy for PCA
    pca_df = df.copy()

    # Select only rows with cluster assignments
    pca_df = pca_df.dropna(subset=['Cluster'])

    # Get standardized features
    X = pca_df[features].values

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)

    # Create a DataFrame with principal components
    pca_df = pd.DataFrame(data=principal_components,
                          columns=['PC1', 'PC2'])

    # Add cluster information
    pca_df['Cluster'] = pca_df.index.map(
        lambda i: df.iloc[i]['Cluster'] if i < len(df) else None
    )

    # Add source information for labeling points
    pca_df['Source'] = pca_df.index.map(
        lambda i: df.iloc[i]['Source / Medium'] if i < len(df) else None
    )

    # Create a scatter plot
    plt.figure(figsize=(12, 8))

    # Plot each cluster with a different color
    for cluster in sorted(pca_df['Cluster'].unique()):
        cluster_data = pca_df[pca_df['Cluster'] == cluster]
        plt.scatter(
            cluster_data['PC1'],
            cluster_data['PC2'],
            label=f'Cluster {cluster}',
            alpha=0.7,
            s=80
        )

    # Add labels for important points (top sources by revenue)
    top_sources = df.sort_values('Revenue', ascending=False).head(10)['Source / Medium'].tolist()
    for i, row in pca_df.iterrows():
        if row['Source'] in top_sources:
            plt.annotate(
                row['Source'],
                (row['PC1'], row['PC2']),
                fontsize=9,
                alpha=0.8
            )

    # Add cluster centers if available
    if hasattr(kmeans, 'cluster_centers_'):
        # Transform cluster centers
        centers = pca.transform(kmeans.cluster_centers_)
        plt.scatter(
            centers[:, 0],
            centers[:, 1],
            s=200,
            c='black',
            marker='X',
            alpha=0.8,
            label='Cluster Centers'
        )

    # Add axis labels with explained variance
    explained_variance = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')

    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt

def analyze_cluster_profiles(df):
    """
    Create a heatmap of cluster profiles based on key metrics.

    Args:
        df: DataFrame with cluster assignments

    Returns:
        matplotlib Figure
    """
    # Calculate mean values for each cluster
    profile_metrics = [
        'Bounce Rate', 'Avg. Session Duration', 'Pages per Session',
        'Conversion Rate (%)', 'Revenue per User', 'New User Ratio',
        'Revenue per Transaction'
    ]

    # Group by cluster and calculate mean for each metric
    cluster_profiles = df.groupby('Cluster')[profile_metrics].mean()

    # Create a heatmap
    plt.figure(figsize=(14, 8))

    # Normalize each column for better visualization
    normalized_profiles = cluster_profiles.copy()
    for col in normalized_profiles.columns:
        normalized_profiles[col] = (normalized_profiles[col] - normalized_profiles[col].min()) / \
                                  (normalized_profiles[col].max() - normalized_profiles[col].min())

    # Plot heatmap with original values as annotations
    sns.heatmap(normalized_profiles, annot=cluster_profiles, fmt='.2f', cmap='YlGnBu',
                linewidths=.5, cbar_kws={'label': 'Normalized Value'})

    plt.title('Cluster Profiles (Normalized Heat with Actual Values)')
    plt.tight_layout()

    return plt

def cluster_composition(df):
    """
    Visualize the composition of each cluster.

    Args:
        df: DataFrame with cluster assignments

    Returns:
        matplotlib Figure
    """
    # Count sources in each cluster
    cluster_counts = df['Cluster'].value_counts().sort_index()

    # Calculate percentage of total revenue by cluster
    revenue_by_cluster = df.groupby('Cluster')['Revenue'].sum()
    revenue_by_cluster = revenue_by_cluster / revenue_by_cluster.sum() * 100

    # Calculate percentage of total sessions by cluster
    sessions_by_cluster = df.groupby('Cluster')['Sessions'].sum()
    sessions_by_cluster = sessions_by_cluster / sessions_by_cluster.sum() * 100

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Number of sources per cluster
    axes[0].bar(
        cluster_counts.index.astype(str),
        cluster_counts.values,
        color='skyblue'
    )
    for i, count in enumerate(cluster_counts):
        axes[0].text(i, count + 0.5, str(count), ha='center')

    axes[0].set_title('Number of Sources per Cluster')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Count')

    # Percentage of revenue by cluster
    axes[1].bar(
        revenue_by_cluster.index.astype(str),
        revenue_by_cluster.values,
        color='salmon'
    )
    for i, pct in enumerate(revenue_by_cluster):
        axes[1].text(i, pct + 0.5, f'{pct:.1f}%', ha='center')

    axes[1].set_title('Percentage of Total Revenue by Cluster')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Percentage')

    # Percentage of sessions by cluster
    axes[2].bar(
        sessions_by_cluster.index.astype(str),
        sessions_by_cluster.values,
        color='lightgreen'
    )
    for i, pct in enumerate(sessions_by_cluster):
        axes[2].text(i, pct + 0.5, f'{pct:.1f}%', ha='center')

    axes[2].set_title('Percentage of Total Sessions by Cluster')
    axes[2].set_xlabel('Cluster')
    axes[2].set_ylabel('Percentage')

    plt.tight_layout()
    return plt

def top_sources_by_cluster(df, top_n=5):
    """
    Show top sources in each cluster.

    Args:
        df: DataFrame with cluster assignments
        top_n: Number of top sources to show per cluster

    Returns:
        matplotlib Figure
    """
    # Get unique clusters sorted
    clusters = sorted(df['Cluster'].dropna().unique())

    # Calculate number of rows and columns for subplots
    n_clusters = len(clusters)
    n_cols = min(2, n_clusters)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))

    # Convert to 2D array of axes if only one row
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = np.array([axes]).reshape(n_rows, n_cols)

    # Flatten for easier iteration
    axes_flat = axes.flatten()

    # For each cluster, show top sources by revenue
    for i, cluster in enumerate(clusters):
        if i < len(axes_flat):
            ax = axes_flat[i]

            # Filter for cluster
            cluster_df = df[df['Cluster'] == cluster]

            # Sort by revenue
            top_sources = cluster_df.sort_values('Revenue', ascending=False).head(top_n)

            # Plot horizontal bar chart
            bars = ax.barh(
                top_sources['Source / Medium'],
                top_sources['Revenue'],
                color='lightblue'
            )

            # Add revenue values
            for bar in bars:
                width = bar.get_width()
                ax.text(
                    width * 1.01,
                    bar.get_y() + bar.get_height()/2,
                    f'${width:,.0f}',
                    va='center'
                )

            ax.set_title(f'Cluster {cluster}: Top {top_n} Sources by Revenue')
            ax.set_xlabel('Revenue')

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    return plt