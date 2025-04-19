# main.py
import os
from scripts.data_loader import load_data, clean_data
from scripts.time_analysis import (prepare_data_for_time_analysis,
                                 monthly_trend, seasonal_pattern,
                                 source_comparison, year_over_year_comparison)
from scripts.source_analysis import (source_performance_metrics, plot_source_performance,
                                    source_metrics_heatmap, source_trend_analysis)
from scripts.behavior_analysis import (engagement_metrics_over_time, source_engagement_comparison,
                                     correlation_matrix, scatter_plots, bounce_rate_analysis)
from scripts.conversion_analysis import (analyze_traffic_conversion_relationship,
                                        source_revenue_analysis)
from scripts.segmentation_analysis import (prepare_data_for_clustering, kmeans_clustering,
                                          visualize_clusters_2d, analyze_cluster_profiles,
                                          cluster_composition, top_sources_by_cluster)
from scripts.predictive_modeling import (prepare_time_series_data, time_series_forecast,
                                        seasonal_analysis, conversion_regression_model,
                                        predict_bounce_classification)
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Define file path
    file_path = r"C:\Users\Amir\Desktop\Web Analytic_Dataset.csv"

    # Load the data
    print("Loading data...")
    df = load_data(file_path)

    # Clean the data
    print("Cleaning data...")
    df_clean = clean_data(df)

    # Display information
    print("\nData Overview:")
    print(f"Shape: {df_clean.shape}")
    print("\nFirst 5 rows:")
    print(df_clean.head())

    print("\nData Types:")
    print(df_clean.dtypes)

    print("\nSummary Statistics:")
    print(df_clean.describe())

    # Create data folder and subdirectories if they don't exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/time_analysis", exist_ok=True)
    os.makedirs("data/source_analysis", exist_ok=True)
    os.makedirs("data/behavior_analysis", exist_ok=True)
    os.makedirs("data/conversion_analysis", exist_ok=True)
    os.makedirs("data/segmentation_analysis", exist_ok=True)
    os.makedirs("data/predictive_modeling", exist_ok=True)



    # Prepare data for analysis
    df_analysis = prepare_data_for_time_analysis(df_clean)

    # Calculate conversion rate for the source trend analysis
    df_analysis['Conversion Rate'] = df_analysis['Transactions'] / df_analysis['Users'] * 100

    # Define metrics to analyze
    metrics = ['Revenue', 'Transactions', 'Users', 'Conversion Rate (%)']

    # Time-based analysis
    print("\nPerforming time-based analysis...")

    # Generate visualizations for each metric
    for metric in metrics:
        print(f"Analyzing {metric}...")
        metric_file = metric.replace(' ', '_').replace('(%)', '')

        # Monthly trend
        fig = monthly_trend(df_analysis, metric)
        fig.savefig(os.path.join("data", "time_analysis", f"monthly_trend_{metric_file}.png"))
        plt.close()

        # Seasonal pattern
        fig = seasonal_pattern(df_analysis, metric)
        fig.savefig(os.path.join("data", "time_analysis", f"seasonal_pattern_{metric_file}.png"))
        plt.close()

        # Source comparison (top 5 + others)
        fig = source_comparison(df_analysis, metric, top_n=5)
        fig.savefig(os.path.join("data", "time_analysis", f"source_comparison_top5_{metric_file}.png"))
        plt.close()

        # Year-over-year comparison
        fig = year_over_year_comparison(df_analysis, metric)
        fig.savefig(os.path.join("data", "time_analysis", f"yoy_comparison_{metric_file}.png"))
        plt.close()

    print("Time-based analysis complete. Visualizations saved to data/time_analysis folder.")

    # Source/Medium Analysis
    print("\nPerforming source/medium analysis...")
    source_metrics = source_performance_metrics(df_analysis)

    # Save source metrics to CSV
    source_metrics.to_csv(os.path.join("data", "source_analysis", "source_performance_metrics.csv"), index=False)
    print(f"Source metrics saved to {os.path.join('data', 'source_analysis', 'source_performance_metrics.csv')}")

    # Top sources by revenue
    fig = plot_source_performance(source_metrics, metric='Revenue', top_n=10)
    fig.savefig(os.path.join("data", "source_analysis", "top_sources_by_revenue.png"))
    plt.close()

    # Top sources by conversion rate
    fig = plot_source_performance(source_metrics, metric='Conversion Rate', top_n=10)
    fig.savefig(os.path.join("data", "source_analysis", "top_sources_by_conversion_rate.png"))
    plt.close()

    # Performance metrics heatmap
    fig = source_metrics_heatmap(source_metrics, top_n=15)
    fig.savefig(os.path.join("data", "source_analysis", "source_metrics_heatmap.png"))
    plt.close()

    # Trend analysis for top source
    top_source = source_metrics.iloc[0]['Source / Medium']
    fig = source_trend_analysis(df_analysis, top_source, metric='Revenue')
    fig.savefig(os.path.join("data", "source_analysis", f"trend_analysis_{top_source.replace('/', '_').replace(' ', '_')}.png"))
    plt.close()

    # Trend analysis for top source by conversion rate
    top_converting_source = source_metrics.sort_values('Conversion Rate', ascending=False).iloc[0]['Source / Medium']
    fig = source_trend_analysis(df_analysis, top_converting_source, metric='Revenue')  # Use Revenue instead of Conversion Rate
    fig.savefig(os.path.join("data", "source_analysis", f"revenue_trend_{top_converting_source.replace('/', '_').replace(' ', '_')}.png"))
    plt.close()

    print("Source/Medium analysis complete. Visualizations saved to data/source_analysis folder.")

    # User Behavior Metrics Analysis
    print("\nAnalyzing user behavior metrics...")

    # Engagement metrics over time
    fig = engagement_metrics_over_time(df_analysis)
    fig.savefig(os.path.join("data", "behavior_analysis", "engagement_metrics_over_time.png"))
    plt.close()

    # Compare engagement metrics across top sources
    fig = source_engagement_comparison(df_analysis, top_n=8)
    fig.savefig(os.path.join("data", "behavior_analysis", "source_engagement_comparison.png"))
    plt.close()

    # Create correlation matrix
    fig = correlation_matrix(df_analysis)
    fig.savefig(os.path.join("data", "behavior_analysis", "metrics_correlation_matrix.png"))
    plt.close()

    # Scatter plots for key relationships
    key_relationships = [
        ('Bounce Rate', 'Conversion Rate (%)'),
        ('Avg. Session Duration', 'Conversion Rate (%)'),
        ('Pageviews', 'Revenue'),
        ('Bounce Rate', 'Revenue')
    ]

    for x_metric, y_metric in key_relationships:
        x_metric_file = x_metric.replace(' ', '_').replace('.', '')
        y_metric_file = y_metric.replace(' ', '_').replace('(%)', '')
        fig = scatter_plots(df_analysis, x_metric, y_metric, top_n=5)
        fig.savefig(os.path.join("data", "behavior_analysis", f"scatter_{x_metric_file}_{y_metric_file}.png"))
        plt.close()

    # Bounce rate analysis
    fig = bounce_rate_analysis(df_analysis)
    fig.savefig(os.path.join("data", "behavior_analysis", "bounce_rate_analysis.png"))
    plt.close()

    print("User behavior analysis complete. Visualizations saved to data/behavior_analysis folder.")

    # Conversion Analysis
    print("\nPerforming conversion analysis...")

    # Traffic and conversion relationship
    fig = analyze_traffic_conversion_relationship(df_analysis)
    fig.savefig(os.path.join("data", "conversion_analysis", "traffic_conversion_relationship.png"))
    plt.close()

    # Source revenue analysis
    fig = source_revenue_analysis(df_analysis)
    fig.savefig(os.path.join("data", "conversion_analysis", "source_revenue_analysis.png"))
    plt.close()

    print("Conversion analysis complete. Visualizations saved to data/conversion_analysis folder.")

    # User Segmentation Analysis
    print("\nPerforming user segmentation analysis...")

    # Prepare data for clustering
    cluster_df, segments_df, features = prepare_data_for_clustering(df_analysis)

    # Perform K-means clustering with 4 clusters
    result_df, kmeans = kmeans_clustering(cluster_df, segments_df, features, n_clusters=4)

    # Save the segmentation results
    result_df.to_csv(os.path.join("data", "segmentation_analysis", "source_segments.csv"), index=False)
    print(f"Segmentation results saved to {os.path.join('data', 'segmentation_analysis', 'source_segments.csv')}")

    # Visualize clusters in 2D
    fig = visualize_clusters_2d(result_df, features, kmeans)
    fig.savefig(os.path.join("data", "segmentation_analysis", "cluster_visualization_2d.png"))
    plt.close()

    # Analyze cluster profiles
    fig = analyze_cluster_profiles(result_df)
    fig.savefig(os.path.join("data", "segmentation_analysis", "cluster_profiles.png"))
    plt.close()

    # Visualize cluster composition
    fig = cluster_composition(result_df)
    fig.savefig(os.path.join("data", "segmentation_analysis", "cluster_composition.png"))
    plt.close()

    # Show top sources by cluster
    fig = top_sources_by_cluster(result_df, top_n=5)
    fig.savefig(os.path.join("data", "segmentation_analysis", "top_sources_by_cluster.png"))
    plt.close()

    print("User segmentation analysis complete. Visualizations saved to data/segmentation_analysis folder.")

    # Predictive Modeling
    print("\nPerforming predictive modeling...")

    # Time Series Forecasting for Sessions
    print("Building time series forecast model for web traffic...")
    ts_data = prepare_time_series_data(df_analysis, metric='Sessions')
    fig, forecast_values = time_series_forecast(ts_data, periods=30)
    fig.savefig(os.path.join("data", "predictive_modeling", "sessions_forecast.png"))
    plt.close()

    # Save forecast values
    forecast_df = pd.DataFrame({'Date': forecast_values.index, 'Forecast Sessions': forecast_values.values})
    forecast_df.to_csv(os.path.join("data", "predictive_modeling", "sessions_forecast_values.csv"), index=False)

    # Seasonal decomposition
    fig = seasonal_analysis(ts_data)
    if fig:
        fig.savefig(os.path.join("data", "predictive_modeling", "sessions_seasonal_decomposition.png"))
        plt.close()

    # Also forecast Revenue
    print("Building time series forecast model for revenue...")
    revenue_ts = prepare_time_series_data(df_analysis, metric='Revenue')
    fig, forecast_values = time_series_forecast(revenue_ts, periods=30)
    fig.savefig(os.path.join("data", "predictive_modeling", "revenue_forecast.png"))
    plt.close()

    # Conversion Rate Regression Model
    print("Building regression model for conversion rate prediction...")
    fig, model, importance = conversion_regression_model(df_analysis)
    fig.savefig(os.path.join("data", "predictive_modeling", "conversion_regression_model.png"))
    plt.close()

    # Save feature importance
    importance.to_csv(os.path.join("data", "predictive_modeling", "conversion_feature_importance.csv"), index=False)

    # Bounce Rate Classification
    print("Building classification model to predict high/low bounce rates...")
    fig, clf, report = predict_bounce_classification(df_analysis)
    fig.savefig(os.path.join("data", "predictive_modeling", "bounce_classification_model.png"))
    plt.close()

    # Save classification report
    with open(os.path.join("data", "predictive_modeling", "bounce_classification_report.txt"), 'w') as f:
        f.write(report)

    print("Predictive modeling complete. Results saved to data/predictive_modeling folder.")

    print("\nAll analyses complete!")


if __name__ == "__main__":
    main()