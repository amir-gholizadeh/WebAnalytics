# scripts/time_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data_for_time_analysis(df):
    """Prepare data for time analysis by converting data types."""
    df = df.copy()

    # Convert numeric columns from strings to numbers
    numeric_cols = ['Users', 'New Users', 'Sessions', 'Pageviews',
                    'Transactions', 'Revenue', 'Quantity Sold']

    for col in numeric_cols:
        if col in df.columns:
            # Remove commas and convert to numeric
            df[col] = df[col].str.replace(',', '').astype(float)

    # Convert Bounce Rate
    if 'Bounce Rate' in df.columns:
        df['Bounce Rate'] = df['Bounce Rate'].str.rstrip('%').astype(float) / 100

    # Convert Conversion Rate with special handling for '<0.01'
    if 'Conversion Rate (%)' in df.columns:
        # Replace '<0.01' with '0.005' (half of 0.01)
        df['Conversion Rate (%)'] = df['Conversion Rate (%)'].replace('<0.01', '0.005')
        df['Conversion Rate (%)'] = df['Conversion Rate (%)'].str.rstrip('%').astype(float) / 100

    # Convert time column
    if 'Avg. Session Duration' in df.columns:
        # If the format is MM:SS or HH:MM:SS, convert to seconds
        df['Avg. Session Duration'] = df['Avg. Session Duration'].apply(
            lambda x: sum(int(x) * 60**i for i, x in enumerate(reversed(x.split(':'))))
            if ':' in str(x) else x
        )

    # Create date column from Year and Month
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' +
                               df['Month of the year'].astype(str) + '-01')

    return df

def monthly_trend(df, metric_column, source_filter=None):
    """Plot monthly trend of a specific metric."""
    data = df.copy()

    if source_filter:
        data = data[data['Source / Medium'] == source_filter]

    # Group by date and calculate sum
    monthly_data = data.groupby('Date')[metric_column].sum().reset_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data['Date'], monthly_data[metric_column], marker='o', linestyle='-')

    title = f'Monthly {metric_column}'
    if source_filter:
        title += f' for {source_filter}'

    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel(metric_column)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def seasonal_pattern(df, metric_column):
    """Identify seasonal patterns in the data."""
    # Group by month across all years
    monthly_avg = df.groupby('Month of the year')[metric_column].mean().reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(monthly_avg['Month of the year'], monthly_avg[metric_column])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if metric_column == 'Conversion Rate (%)':
            # Use more decimal places for conversion rates
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom')
        else:
            # Use regular formatting for other metrics
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}', ha='center', va='bottom')

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(range(1, 13), months)
    plt.title(f'Monthly Pattern of {metric_column}')
    plt.xlabel('Month')
    plt.ylabel(f'Average {metric_column}')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return plt


def source_comparison(df, metric_column, top_n=5):
    """Compare performance of top N sources over time, grouping others."""
    # Group by source and date
    source_data = df.groupby(['Source / Medium', 'Date'])[metric_column].sum().reset_index()

    # Identify top N sources by total metric value
    top_sources = source_data.groupby('Source / Medium')[metric_column].sum().nlargest(top_n).index.tolist()

    # Create a modified dataframe with top sources and "Others"
    modified_data = source_data.copy()
    modified_data['Source Category'] = modified_data['Source / Medium'].apply(
        lambda x: x if x in top_sources else 'Others'
    )

    # Aggregate by the new source category
    agg_data = modified_data.groupby(['Source Category', 'Date'])[metric_column].sum().reset_index()

    # Plot
    plt.figure(figsize=(14, 8))

    # Plot Others first (so it appears in the background)
    others_data = agg_data[agg_data['Source Category'] == 'Others']
    if not others_data.empty:
        plt.plot(others_data['Date'], others_data[metric_column],
                 marker='o', linestyle='--', color='gray', linewidth=1.5, label='Others')

    # Plot top sources with distinct colors
    for source in top_sources:
        source_subset = agg_data[agg_data['Source Category'] == source]
        plt.plot(source_subset['Date'], source_subset[metric_column],
                 marker='o', linewidth=2, label=source)

    plt.title(f'Top {top_n} Sources - {metric_column}')
    plt.xlabel('Date')
    plt.ylabel(metric_column)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(
        fontsize=10,
        title='Source / Medium'
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return plt


def year_over_year_comparison(df, metric_column):
    """Compare the same months across different years in chronological order."""
    # Create a copy of data
    temp_df = df.copy()

    # Ensure Date column is datetime
    temp_df['Date'] = pd.to_datetime(temp_df['Date'])

    # Sort by date to ensure chronological order
    temp_df = temp_df.sort_values('Date')

    # Group by year and month to get the sum of metrics
    monthly_data = temp_df.groupby([
        temp_df['Date'].dt.year.rename('Year'),
        temp_df['Date'].dt.month.rename('Month')
    ])[metric_column].sum().reset_index()

    # Create a single chronological index for plotting
    monthly_data['YearMonth'] = monthly_data['Year'].astype(str) + '-' + monthly_data['Month'].astype(str).str.zfill(2)
    monthly_data['YearMonthLabel'] = monthly_data['Month'].apply(
        lambda m: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][m - 1]
    ) + '\n' + monthly_data['Year'].astype(str)

    # Sort chronologically
    monthly_data = monthly_data.sort_values(['Year', 'Month'])

    # Plot
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(monthly_data)), monthly_data[metric_column], width=0.7)

    # Color the bars by year
    years = sorted(monthly_data['Year'].unique())
    colors = plt.cm.tab10(range(len(years)))
    year_color_map = dict(zip(years, colors))

    for i, bar in enumerate(bars):
        year = monthly_data.iloc[i]['Year']
        bar.set_color(year_color_map[year])

    # Add a legend for years
    handles = [plt.Rectangle((0, 0), 1, 1, color=year_color_map[y]) for y in years]
    plt.legend(handles, [str(y) for y in years], title='Year')

    plt.title(f'Year over Year Comparison - {metric_column}')
    plt.xlabel('Year-Month')
    plt.ylabel(metric_column)
    plt.xticks(range(len(monthly_data)), monthly_data['YearMonthLabel'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return plt