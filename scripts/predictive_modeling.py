# scripts/predictive_modeling.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, TimeSeriesSplit

def prepare_time_series_data(df, metric='Sessions'):
    """
    Prepare time series data for forecasting.

    Args:
        df: DataFrame with cleaned analytics data
        metric: Metric to forecast

    Returns:
        Series with datetime index for time series analysis
    """
    # Group by date to get daily totals
    ts_data = df.groupby('Date')[metric].sum()

    # Convert index to datetime
    ts_data.index = pd.to_datetime(ts_data.index)

    # Sort by date
    ts_data = ts_data.sort_index()

    return ts_data


def time_series_forecast(ts_data, periods=30):
    """
    Perform time series forecasting using SARIMA model.

    Args:
        ts_data: Time series data with datetime index
        periods: Number of periods to forecast

    Returns:
        matplotlib Figure with forecast results
    """
    # Create training set
    train_data = ts_data

    # Check if data is stationary
    # Using simpler model parameters for more stability
    model = SARIMAX(train_data,
                    order=(1, 1, 0),  # (p,d,q) - trend component
                    seasonal_order=(0, 0, 0, 0),  # No seasonal component
                    enforce_stationarity=False,
                    enforce_invertibility=False)

    results = model.fit(disp=False)

    # Create forecast
    forecast = results.get_forecast(steps=periods)
    forecast_ci = forecast.conf_int()

    # Ensure forecast is positive (can't have negative sessions or revenue)
    forecast_mean = forecast.predicted_mean
    forecast_mean = forecast_mean.clip(lower=0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual data
    ax.plot(ts_data.index, ts_data, label='Observed')

    # Plot forecast
    ax.plot(forecast_mean.index,
            forecast_mean,
            color='red',
            label='Forecast')

    # Plot confidence intervals
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0].clip(lower=0),
                    forecast_ci.iloc[:, 1],
                    color='pink', alpha=0.3)

    ax.set_title(f'Time Series Forecast - {ts_data.name}')
    ax.set_xlabel('Date')
    ax.set_ylabel(ts_data.name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt, forecast_mean

def seasonal_analysis(ts_data):
    """
    Perform seasonal decomposition of time series data.

    Args:
        ts_data: Time series data with datetime index

    Returns:
        matplotlib Figure with seasonal decomposition
    """
    # Ensure we have enough data for decomposition
    if len(ts_data) < 14:
        print("Warning: Not enough data for seasonal decomposition")
        return None

    # Handle frequency
    if not ts_data.index.is_monotonic:
        ts_data = ts_data.sort_index()

    # Determine frequency - daily, weekly, monthly
    if len(ts_data) >= 2*365:  # If we have at least 2 years of data
        period = 365  # Annual seasonality
    elif len(ts_data) >= 2*30:  # If we have at least 2 months of data
        period = 30   # Monthly seasonality
    else:
        period = 7    # Weekly seasonality

    # Perform decomposition
    decomposition = seasonal_decompose(ts_data, model='additive', period=period)

    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
    decomposition.resid.plot(ax=axes[3], title='Residuals')

    plt.tight_layout()
    return plt

def conversion_regression_model(df):
    """
    Build a regression model to predict conversion rates.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        Tuple of (figure with results, model, feature importances)
    """
    # Group by date and source
    model_data = df.groupby(['Date', 'Source / Medium']).agg({
        'Sessions': 'sum',
        'Pageviews': 'sum',
        'Bounce Rate': 'mean',
        'Avg. Session Duration': 'mean',
        'Conversion Rate (%)': 'mean',
        'New Users': 'sum',
        'Users': 'sum'
    }).reset_index()

    # Create derived features
    model_data['Pages per Session'] = model_data['Pageviews'] / model_data['Sessions']
    model_data['New User Ratio'] = model_data['New Users'] / model_data['Users']

    # Select features and target
    features = ['Pages per Session', 'Bounce Rate', 'Avg. Session Duration', 'New User Ratio']
    X = model_data[features]
    y = model_data['Conversion Rate (%)']

    # Handle missing values
    X = X.fillna(X.mean())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    axes[0].set_xlabel('Actual Conversion Rate (%)')
    axes[0].set_ylabel('Predicted Conversion Rate (%)')
    axes[0].set_title(f'Actual vs Predicted Conversion Rate\nMSE: {mse:.4f}, R²: {r2:.4f}')
    axes[0].grid(True, alpha=0.3)

    # Feature Importance
    sns.barplot(x='Importance', y='Feature', data=importance, ax=axes[1])
    axes[1].set_title('Feature Importance')
    axes[1].set_xlabel('Importance')

    plt.tight_layout()

    return plt, rf_model, importance

def predict_bounce_classification(df):
    """
    Build a classification model to predict high/low bounce rate.

    Args:
        df: DataFrame with cleaned analytics data

    Returns:
        Tuple of (figure with results, model, classification report)
    """
    # Group by date and source
    bounce_data = df.groupby(['Date', 'Source / Medium']).agg({
        'Sessions': 'sum',
        'Pageviews': 'sum',
        'Users': 'sum',
        'New Users': 'sum',
        'Avg. Session Duration': 'mean',
        'Bounce Rate': 'mean'
    }).reset_index()

    # Create features
    bounce_data['Pages per Session'] = bounce_data['Pageviews'] / bounce_data['Sessions']
    bounce_data['New User Ratio'] = bounce_data['New Users'] / bounce_data['Users']

    # Create target: high bounce (1) or low bounce (0)
    median_bounce = bounce_data['Bounce Rate'].median()
    bounce_data['High Bounce'] = (bounce_data['Bounce Rate'] > median_bounce).astype(int)

    # Select features and target
    features = ['Pages per Session', 'Avg. Session Duration', 'New User Ratio']
    X = bounce_data[features]
    y = bounce_data['High Bounce']

    # Handle missing values
    X = X.fillna(X.mean())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = clf.predict(X_test_scaled)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)

    # Create visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Confusion Matrix
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'Confusion Matrix (Accuracy: {accuracy:.4f})')

    # Feature Importance
    sns.barplot(x='Importance', y='Feature', data=importance, ax=axes[1])
    axes[1].set_title('Feature Importance')
    axes[1].set_xlabel('Importance')

    plt.tight_layout()

    return plt, clf, report