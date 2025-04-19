# dashboard.py
import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from scripts.data_loader import load_data, clean_data
from scripts.time_analysis import prepare_data_for_time_analysis
from scripts.predictive_modeling import prepare_time_series_data, time_series_forecast

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define file path - use the actual path from your main.py
file_path = r"C:\Users\Amir\Desktop\Web Analytic_Dataset.csv"

# Load and prepare data
print("Loading data...")
df = load_data(file_path)
df_clean = clean_data(df)
df_analysis = prepare_data_for_time_analysis(df_clean)

# Calculate conversion rate for consistency with your main.py
df_analysis['Conversion Rate'] = df_analysis['Transactions'] / df_analysis['Users'] * 100

# Create app layout
app.layout = html.Div([
    html.H1("Web Analytics Dashboard", style={'textAlign': 'center', 'margin-bottom': '30px'}),

    # Tabs for different analysis sections
    dcc.Tabs([
        # Overview Tab
        dcc.Tab(label='Overview', children=[
            html.Div([
                html.H3("Key Metrics Overview", style={'textAlign': 'center', 'margin': '20px'}),

                # Date range filter
                html.Div([
                    html.Label("Select Date Range:"),
                    dcc.DatePickerRange(
                        id='overview-date-range',
                        min_date_allowed=df_analysis['Date'].min(),
                        max_date_allowed=df_analysis['Date'].max(),
                        start_date=df_analysis['Date'].min(),
                        end_date=df_analysis['Date'].max()
                    )
                ], style={'margin': '20px'}),

                # Key metrics cards
                html.Div(id='key-metrics-cards', className='row', style={'display': 'flex', 'justifyContent': 'space-around'}),

                # Time series chart for main metrics
                html.Div([
                    html.Label("Select Metric:"),
                    dcc.Dropdown(
                        id='overview-metric-dropdown',
                        options=[
                            {'label': 'Users', 'value': 'Users'},
                            {'label': 'Sessions', 'value': 'Sessions'},
                            {'label': 'Revenue', 'value': 'Revenue'},
                            {'label': 'Transactions', 'value': 'Transactions'},
                            {'label': 'Conversion Rate', 'value': 'Conversion Rate'}
                        ],
                        value='Users'
                    )
                ], style={'width': '30%', 'margin': '20px'}),

                dcc.Graph(id='overview-time-series')
            ])
        ]),

        # Source Analysis Tab
        dcc.Tab(label='Source Analysis', children=[
            html.Div([
                html.H3("Traffic Source Analysis", style={'textAlign': 'center', 'margin': '20px'}),

                # Source performance chart
                html.Div([
                    html.Label("Select Metric:"),
                    dcc.Dropdown(
                        id='source-metric-dropdown',
                        options=[
                            {'label': 'Revenue', 'value': 'Revenue'},
                            {'label': 'Users', 'value': 'Users'},
                            {'label': 'Transactions', 'value': 'Transactions'},
                            {'label': 'Conversion Rate', 'value': 'Conversion Rate'}
                        ],
                        value='Revenue'
                    ),
                    html.Label("Number of Sources:"),
                    dcc.Slider(
                        id='source-count-slider',
                        min=5,
                        max=20,
                        step=5,
                        value=10,
                        marks={i: str(i) for i in range(5, 25, 5)}
                    )
                ], style={'width': '50%', 'margin': '20px'}),

                dcc.Graph(id='source-performance-chart'),

                # Source comparison over time
                html.Div([
                    html.Label("Select Top Sources to Compare:"),
                    dcc.Dropdown(
                        id='source-selection-dropdown',
                        options=[{'label': source, 'value': source}
                                for source in df_analysis['Source / Medium'].unique()],
                        multi=True,
                        value=df_analysis.groupby('Source / Medium')['Revenue'].sum().nlargest(5).index.tolist()
                    )
                ], style={'width': '70%', 'margin': '20px'}),

                dcc.Graph(id='source-comparison-chart')
            ])
        ]),

        # Behavior Analysis Tab
        dcc.Tab(label='Behavior Analysis', children=[
            html.Div([
                html.H3("User Behavior Analysis", style={'textAlign': 'center', 'margin': '20px'}),

                # Behavior metrics visualization
                html.Div([
                    dcc.Graph(id='engagement-metrics-chart')
                ]),

                # Correlation analysis
                html.Div([
                    html.Label("Select Metrics to Correlate:"),
                    html.Div([
                        dcc.Dropdown(
                            id='x-axis-metric',
                            options=[
                                {'label': 'Bounce Rate', 'value': 'Bounce Rate'},
                                {'label': 'Avg. Session Duration', 'value': 'Avg. Session Duration'},
                                {'label': 'Pages per Session', 'value': 'Pageviews'},
                                {'label': 'New User Ratio', 'value': 'New Users'}
                            ],
                            value='Bounce Rate'
                        ),
                        dcc.Dropdown(
                            id='y-axis-metric',
                            options=[
                                {'label': 'Conversion Rate', 'value': 'Conversion Rate'},
                                {'label': 'Revenue', 'value': 'Revenue'},
                                {'label': 'Transactions', 'value': 'Transactions'}
                            ],
                            value='Conversion Rate'
                        )
                    ], style={'display': 'flex', 'gap': '20px'})
                ], style={'width': '50%', 'margin': '20px'}),

                dcc.Graph(id='correlation-scatter-plot')
            ])
        ]),

        # Forecasting Tab
        dcc.Tab(label='Forecasting', children=[
            html.Div([
                html.H3("Predictive Analytics", style={'textAlign': 'center', 'margin': '20px'}),

                # Forecasting options
                html.Div([
                    html.Label("Select Metric to Forecast:"),
                    dcc.Dropdown(
                        id='forecast-metric-dropdown',
                        options=[
                            {'label': 'Sessions', 'value': 'Sessions'},
                            {'label': 'Revenue', 'value': 'Revenue'},
                            {'label': 'Transactions', 'value': 'Transactions'}
                        ],
                        value='Sessions'
                    ),
                    html.Label("Forecast Period (Days):"),
                    dcc.Slider(
                        id='forecast-period-slider',
                        min=7,
                        max=90,
                        step=7,
                        value=30,
                        marks={i: f'{i}d' for i in range(7, 91, 7)}
                    )
                ], style={'width': '50%', 'margin': '20px'}),

                # Forecast chart
                dcc.Graph(id='forecast-chart'),

                # Forecast data table
                html.Div(id='forecast-table-container', style={'margin': '20px'})
            ])
        ])
    ])
])

# Callback for key metrics cards
@app.callback(
    Output('key-metrics-cards', 'children'),
    [Input('overview-date-range', 'start_date'),
     Input('overview-date-range', 'end_date')]
)
def update_key_metrics(start_date, end_date):
    # Filter data by date range
    filtered_df = df_analysis[(df_analysis['Date'] >= start_date) & (df_analysis['Date'] <= end_date)]

    # Calculate total metrics
    total_users = filtered_df['Users'].sum()
    total_sessions = filtered_df['Sessions'].sum()
    total_revenue = filtered_df['Revenue'].sum()
    total_transactions = filtered_df['Transactions'].sum()
    avg_conversion_rate = (total_transactions / total_users * 100) if total_users > 0 else 0

    # Create metric cards
    metrics = [
        {'name': 'Total Users', 'value': f"{total_users:,.0f}"},
        {'name': 'Total Sessions', 'value': f"{total_sessions:,.0f}"},
        {'name': 'Total Revenue', 'value': f"${total_revenue:,.2f}"},
        {'name': 'Transactions', 'value': f"{total_transactions:,.0f}"},
        {'name': 'Conversion Rate', 'value': f"{avg_conversion_rate:.2f}%"}
    ]

    # Return HTML div with metric cards
    return [
        html.Div([
            html.H4(metric['name']),
            html.H2(metric['value'])
        ], style={
            'width': '18%',
            'textAlign': 'center',
            'padding': '10px',
            'boxShadow': '0px 0px 5px 1px rgba(0,0,0,0.1)',
            'borderRadius': '5px'
        }) for metric in metrics
    ]

# Callback for overview time series
@app.callback(
    Output('overview-time-series', 'figure'),
    [Input('overview-metric-dropdown', 'value'),
     Input('overview-date-range', 'start_date'),
     Input('overview-date-range', 'end_date')]
)
def update_overview_time_series(metric, start_date, end_date):
    # Filter data by date range
    filtered_df = df_analysis[(df_analysis['Date'] >= start_date) & (df_analysis['Date'] <= end_date)]

    # Group by date
    time_data = filtered_df.groupby('Date')[metric].sum().reset_index()

    # Create figure
    fig = px.line(time_data, x='Date', y=metric, title=f'{metric} Over Time')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=metric,
        template='plotly_white'
    )

    return fig

# Callback for source performance chart
@app.callback(
    Output('source-performance-chart', 'figure'),
    [Input('source-metric-dropdown', 'value'),
     Input('source-count-slider', 'value'),
     Input('overview-date-range', 'start_date'),
     Input('overview-date-range', 'end_date')]
)
def update_source_performance(metric, n_sources, start_date, end_date):
    # Filter data by date range
    filtered_df = df_analysis[(df_analysis['Date'] >= start_date) & (df_analysis['Date'] <= end_date)]

    # Group by source
    source_data = filtered_df.groupby('Source / Medium')[metric].sum().reset_index()

    # Get top N sources
    top_sources = source_data.nlargest(n_sources, metric)

    # Sort in ascending order for horizontal bar chart
    top_sources = top_sources.sort_values(metric)

    # Create figure
    fig = px.bar(
        top_sources,
        y='Source / Medium',
        x=metric,
        orientation='h',
        title=f'Top {n_sources} Sources by {metric}'
    )

    fig.update_layout(
        yaxis_title='',
        xaxis_title=metric,
        template='plotly_white',
        height=600
    )

    return fig

# Callback for source comparison chart
@app.callback(
    Output('source-comparison-chart', 'figure'),
    [Input('source-selection-dropdown', 'value'),
     Input('source-metric-dropdown', 'value'),
     Input('overview-date-range', 'start_date'),
     Input('overview-date-range', 'end_date')]
)
def update_source_comparison(selected_sources, metric, start_date, end_date):
    # Filter data by date range
    filtered_df = df_analysis[(df_analysis['Date'] >= start_date) & (df_analysis['Date'] <= end_date)]

    # Filter for selected sources
    source_df = filtered_df[filtered_df['Source / Medium'].isin(selected_sources)]

    # Group by source and date
    source_time_data = source_df.groupby(['Source / Medium', 'Date'])[metric].sum().reset_index()

    # Create figure
    fig = px.line(
        source_time_data,
        x='Date',
        y=metric,
        color='Source / Medium',
        title=f'{metric} Comparison by Source'
    )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=metric,
        template='plotly_white'
    )

    return fig

# Callback for engagement metrics chart
@app.callback(
    Output('engagement-metrics-chart', 'figure'),
    [Input('overview-date-range', 'start_date'),
     Input('overview-date-range', 'end_date')]
)
def update_engagement_metrics(start_date, end_date):
    # Filter data by date range
    filtered_df = df_analysis[(df_analysis['Date'] >= start_date) & (df_analysis['Date'] <= end_date)]

    # Group by date
    engagement_data = filtered_df.groupby('Date').agg({
        'Bounce Rate': 'mean',
        'Avg. Session Duration': 'mean',
        'Pageviews': 'mean'
    }).reset_index()

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Add bounce rate
    fig.add_trace(
        go.Scatter(
            x=engagement_data['Date'],
            y=engagement_data['Bounce Rate'],
            name='Bounce Rate',
            line=dict(color='crimson', width=2)
        )
    )

    # Add session duration
    fig.add_trace(
        go.Scatter(
            x=engagement_data['Date'],
            y=engagement_data['Avg. Session Duration'],
            name='Avg. Session Duration',
            yaxis='y2',
            line=dict(color='forestgreen', width=2)
        )
    )

    # Add pageviews
    fig.add_trace(
        go.Scatter(
            x=engagement_data['Date'],
            y=engagement_data['Pageviews'],
            name='Pageviews per Session',
            yaxis='y3',
            line=dict(color='royalblue', width=2)
        )
    )

    # Update layout with three y-axes
    fig.update_layout(
        title='Engagement Metrics Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(
            title='Bounce Rate',
            titlefont=dict(color='crimson'),
            tickfont=dict(color='crimson'),
            range=[0, 1]
        ),
        yaxis2=dict(
            title='Session Duration (sec)',
            titlefont=dict(color='forestgreen'),
            tickfont=dict(color='forestgreen'),
            anchor='free',
            overlaying='y',
            side='right',
            position=0.9
        ),
        yaxis3=dict(
            title='Pageviews',
            titlefont=dict(color='royalblue'),
            tickfont=dict(color='royalblue'),
            anchor='free',
            overlaying='y',
            side='right',
            position=1.0
        ),
        template='plotly_white',
        legend=dict(x=0.01, y=0.99)
    )

    return fig

# Callback for correlation scatter plot
@app.callback(
    Output('correlation-scatter-plot', 'figure'),
    [Input('x-axis-metric', 'value'),
     Input('y-axis-metric', 'value'),
     Input('overview-date-range', 'start_date'),
     Input('overview-date-range', 'end_date')]
)
def update_correlation_plot(x_metric, y_metric, start_date, end_date):
    # Filter data by date range
    filtered_df = df_analysis[(df_analysis['Date'] >= start_date) & (df_analysis['Date'] <= end_date)]

    # Group by source to get aggregate metrics
    source_metrics = filtered_df.groupby('Source / Medium').agg({
        x_metric: 'mean',
        y_metric: 'mean',
        'Sessions': 'sum'
    }).reset_index()

    # Create figure
    fig = px.scatter(
        source_metrics,
        x=x_metric,
        y=y_metric,
        size='Sessions',
        color='Sessions',
        hover_data=['Source / Medium'],
        title=f'Correlation between {x_metric} and {y_metric}',
        trendline='ols'
    )

    fig.update_layout(
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        template='plotly_white'
    )

    return fig

# Callback for forecast chart
@app.callback(
    [Output('forecast-chart', 'figure'),
     Output('forecast-table-container', 'children')],
    [Input('forecast-metric-dropdown', 'value'),
     Input('forecast-period-slider', 'value')]
)
def update_forecast(metric, periods):
    # Prepare time series data
    ts_data = prepare_time_series_data(df_analysis, metric=metric)

    # Get forecast (use matplotlib figure and convert to plotly)
    _, forecast_values = time_series_forecast(ts_data, periods=periods)

    # Create a DataFrame with historical and forecast values
    historical = pd.DataFrame({
        'Date': ts_data.index,
        'Value': ts_data.values,
        'Type': 'Historical'
    })

    forecast = pd.DataFrame({
        'Date': forecast_values.index,
        'Value': forecast_values.values,
        'Type': 'Forecast'
    })

    plot_data = pd.concat([historical, forecast])

    # Create figure
    fig = px.line(
        plot_data,
        x='Date',
        y='Value',
        color='Type',
        title=f'{metric} Forecast',
        color_discrete_map={'Historical': 'blue', 'Forecast': 'red'}
    )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=metric,
        template='plotly_white'
    )

    # Create forecast table
    forecast_table = dash_table.DataTable(
        columns=[
            {'name': 'Date', 'id': 'Date'},
            {'name': f'Forecasted {metric}', 'id': 'Value'}
        ],
        data=forecast.assign(
            Date=forecast['Date'].dt.strftime('%Y-%m-%d'),
            Value=forecast['Value'].round(2)
        ).to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'padding': '10px'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        page_size=10
    )

    return fig, html.Div([
        html.H4(f"Forecasted {metric} for Next {periods} Days"),
        forecast_table
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)