import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from utils import load_data, load_models, format_number
import time

# Page configuration
st.set_page_config(
    page_title="Sales Forecast",
    page_icon="🔮",
    layout="wide"
)

# Load data and models
@st.cache_resource
def load_resources():
    df, _ = load_data()
    xgb_model, _, _, _ = load_models()
    return df, xgb_model

df, xgb_model = load_resources()

# Header
st.title("🔮 Sales Forecast")
st.markdown("Interactive forecasting tool powered by XGBoost to predict future sales trends")

# Create daily sales dataframe for forecasting
@st.cache_data
def prepare_forecast_data(df):
    # Group by date and sum the total price for each day
    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales['InvoiceDate'])
    daily_sales = daily_sales.sort_values('Date')
    
    # Feature Engineering for Time Series
    daily_sales['dayofweek'] = daily_sales['Date'].dt.dayofweek
    daily_sales['month'] = daily_sales['Date'].dt.month
    daily_sales['year'] = daily_sales['Date'].dt.year
    daily_sales['day'] = daily_sales['Date'].dt.day
    daily_sales['is_weekend'] = (daily_sales['dayofweek'] >= 5).astype(int)
    
    # Create lag features (previous days' sales)
    for i in range(1, 8):
        daily_sales[f'lag_{i}'] = daily_sales['TotalPrice'].shift(i)
    
    # Create rolling window features
    daily_sales['rolling_mean_3'] = daily_sales['TotalPrice'].rolling(window=3).mean()
    daily_sales['rolling_mean_7'] = daily_sales['TotalPrice'].rolling(window=7).mean()
    daily_sales['rolling_std_7'] = daily_sales['TotalPrice'].rolling(window=7).std()
    
    # Drop NA values created by lag and rolling features
    daily_sales = daily_sales.dropna()
    
    return daily_sales

daily_sales = prepare_forecast_data(df)

# Forecast Settings
with st.sidebar:
    st.header("Forecast Settings")
    forecast_days = st.slider(
        "Forecast Horizon (days)", 
        min_value=7, 
        max_value=90, 
        value=30, 
        step=7,
        key="forecast_horizon"
    )
    
    confidence_level = st.slider(
        "Confidence Level", 
        min_value=0.8, 
        max_value=0.99, 
        value=0.95, 
        step=0.01,
        key="confidence_level"
    )
    
    historical_days = st.slider(
        "Historical Days to Show",
        min_value=7,
        max_value=90,
        value=30,
        step=7,
        key="historical_days"
    )
    
    run_forecast = st.button("Generate Forecast", use_container_width=True)

# Sales Forecast
st.header("Sales Forecast")
forecast_container = st.container()

# Prepare forecast input data
last_date = daily_sales['Date'].max()
forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)

def generate_forecast(daily_sales, forecast_dates, confidence_level):
    """Generate sales forecast with confidence intervals"""
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Generate features for forecast dates
    forecast_df = pd.DataFrame()
    forecast_df['Date'] = forecast_dates
    forecast_df['dayofweek'] = forecast_df['Date'].dt.dayofweek
    forecast_df['month'] = forecast_df['Date'].dt.month
    forecast_df['year'] = forecast_df['Date'].dt.year
    forecast_df['day'] = forecast_df['Date'].dt.day
    forecast_df['is_weekend'] = (forecast_df['dayofweek'] >= 5).astype(int)
    
    # Initialize with the last known sales values
    last_known_sales = daily_sales.iloc[-7:]['TotalPrice'].values
    
    # Generate forecasts iteratively for each day
    forecasts = []
    lower_bounds = []
    upper_bounds = []
    
    total_steps = len(forecast_df)
    
    for i in range(total_steps):
        # Update progress
        progress = (i + 1) / total_steps
        progress_bar.progress(progress)
        status_text.text(f"Generating forecast: {i+1}/{total_steps} days processed")
        
        # Use the model to predict the next day
        if i < 7:
            # For the first 7 days, we use the last known sales
            lag_values = list(last_known_sales[-(7-i):]) + list(np.array(forecasts)[:i])
        else:
            # After that, we use previous predictions
            lag_values = forecasts[i-7:i]
        
        # Calculate rolling metrics from available values
        if i < 3:
            rolling_mean_3 = np.mean(list(last_known_sales[-3:]) + forecasts[:i])
        else:
            rolling_mean_3 = np.mean(forecasts[i-3:i])
            
        if i < 7:
            known_plus_forecast = list(last_known_sales[-(7-i):]) + forecasts[:i]
            rolling_mean_7 = np.mean(known_plus_forecast)
            rolling_std_7 = np.std(known_plus_forecast) if len(known_plus_forecast) > 1 else 0
        else:
            rolling_mean_7 = np.mean(forecasts[i-7:i])
            rolling_std_7 = np.std(forecasts[i-7:i])
        
        # Create the feature row for prediction
        X_pred = pd.DataFrame({
            'dayofweek': [forecast_df.iloc[i]['dayofweek']],
            'month': [forecast_df.iloc[i]['month']],
            'year': [forecast_df.iloc[i]['year']],
            'day': [forecast_df.iloc[i]['day']],
            'is_weekend': [forecast_df.iloc[i]['is_weekend']],
        })
        
        # Add lag features
        for j in range(1, 8):
            X_pred[f'lag_{j}'] = lag_values[7-j] if 7-j < len(lag_values) else last_known_sales[-(j-i)]
        
        # Add rolling features
        X_pred['rolling_mean_3'] = rolling_mean_3
        X_pred['rolling_mean_7'] = rolling_mean_7
        X_pred['rolling_std_7'] = rolling_std_7
        
        # Make prediction
        pred = xgb_model.predict(X_pred)[0]
        # Ensure predictions are non-negative
        pred = max(0, pred)
        forecasts.append(pred)
        
        # Calculate confidence intervals based on recent volatility
        if i < 7:
            std_dev = daily_sales['TotalPrice'].rolling(window=30).std().iloc[-1]
        else:
            std_dev = np.std(forecasts[max(0, i-30):i]) if i > 0 else daily_sales['TotalPrice'].std()
        
        # Get z-score based on confidence level
        if confidence_level >= 0.99:
            z_score = 2.58
        elif confidence_level >= 0.95:
            z_score = 1.96
        elif confidence_level >= 0.90:
            z_score = 1.64
        else:
            z_score = 1.28
            
        # Increase uncertainty over time
        margin = z_score * std_dev * np.sqrt(1 + i/10)
        
        lower_bounds.append(max(0, pred - margin))
        upper_bounds.append(pred + margin)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    forecast_df['Forecast'] = forecasts
    forecast_df['Lower Bound'] = lower_bounds
    forecast_df['Upper Bound'] = upper_bounds
    
    return forecast_df

# Check if we should generate a new forecast
if "forecast_df" not in st.session_state or run_forecast:
    with forecast_container:
        forecast_df = generate_forecast(daily_sales, forecast_dates, confidence_level)
        st.session_state["forecast_df"] = forecast_df
        st.session_state["last_params"] = {
            "forecast_days": forecast_days,
            "confidence_level": confidence_level
        }
# If parameters changed without clicking button, note it but don't regenerate
elif (st.session_state.get("last_params", {}).get("forecast_days") != forecast_days or 
      st.session_state.get("last_params", {}).get("confidence_level") != confidence_level):
    with forecast_container:
        st.warning("Forecast settings changed. Click 'Generate Forecast' to update.")
        forecast_df = st.session_state["forecast_df"]
else:
    forecast_df = st.session_state["forecast_df"]

with forecast_container:
    # Combine historical data with forecast for visualization
    hist_data = daily_sales[['Date', 'TotalPrice']].rename(columns={'TotalPrice': 'Historical'})
    
    # Only show the specified number of historical days
    if historical_days > 0:
        cutoff_date = last_date - timedelta(days=historical_days)
        hist_data = hist_data[hist_data['Date'] >= cutoff_date]
        
    combined_df = pd.merge(hist_data, forecast_df, on='Date', how='outer')
    
    # Create a line chart with confidence intervals
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=combined_df['Date'],
        y=combined_df['Historical'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='royalblue', width=2)
    ))
    
    # Forecast data
    fig.add_trace(go.Scatter(
        x=combined_df['Date'],
        y=combined_df['Forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='firebrick', width=2)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=combined_df['Date'].tolist() + combined_df['Date'].tolist()[::-1],
        y=combined_df['Upper Bound'].tolist() + combined_df['Lower Bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(231, 107, 243, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name=f'{int(confidence_level*100)}% Confidence Interval',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'{forecast_days}-Day Sales Forecast with {int(confidence_level*100)}% Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast metrics
    total_forecast = forecast_df['Forecast'].sum()
    avg_daily_forecast = forecast_df['Forecast'].mean()
    
    # Calculate growth compared to similar historical period
    try:
        historical_comparison = daily_sales['TotalPrice'].tail(min(forecast_days, len(daily_sales))).mean()
        forecast_growth = ((avg_daily_forecast / historical_comparison) - 1) * 100
    except:
        forecast_growth = 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Forecasted Sales", f"${format_number(total_forecast)}")
    with col2:
        st.metric("Avg. Daily Forecast", f"${format_number(avg_daily_forecast)}")
    with col3:
        st.metric("Forecast Growth", f"{forecast_growth:.2f}%", 
                delta=f"{forecast_growth:.2f}%",
                delta_color="normal" if forecast_growth >= 0 else "inverse")

# Feature importance for the forecast
st.header("Forecast Drivers")
st.markdown("Key factors influencing the sales forecast")

# Get feature importance from the model
@st.cache_data
def get_feature_importance(_model):
    feature_importance = pd.DataFrame({
        'Feature': _model.feature_names_in_,
        'Importance': _model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Clean up feature names for better display
    feature_importance['Feature'] = feature_importance['Feature'].str.replace('_', ' ').str.title()
    
    return feature_importance

feature_importance = get_feature_importance(xgb_model)

# Plot feature importance
fig = px.bar(
    feature_importance.head(10), 
    x='Importance', 
    y='Feature',
    orientation='h',
    title='Top 10 Features Driving the Forecast',
    labels={'Importance': 'Relative Importance', 'Feature': 'Feature'},
    color='Importance',
    color_continuous_scale='Viridis'
)
fig.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig, use_container_width=True)

# Scenario Testing
st.header("Forecast Scenario Testing")
st.markdown("Adjust factors to see how they might affect future sales")

# Scenario inputs
col1, col2, col3 = st.columns(3)

with col1:
    weekend_boost = st.slider(
        "Weekend Sales Boost (%)",
        min_value=-50,
        max_value=100,
        value=0,
        help="Adjust weekend sales performance",
        key="weekend_boost"
    )
    
with col2:
    month_options = [
        "Normal", 
        "Holiday Season (+20%)", 
        "Summer Sale (+15%)", 
        "Slow Season (-10%)"
    ]
    month_factor = st.selectbox(
        "Seasonal Month Adjustment",
        options=month_options,
        index=0,
        help="Adjust for seasonal effects",
        key="month_factor"
    )
    
with col3:
    trend_adjustment = st.slider(
        "Overall Trend Adjustment (%)",
        min_value=-30,
        max_value=50,
        value=0,
        help="Adjust the overall sales trend",
        key="trend_adjustment"
    )

# Create scenario container to show results
scenario_container = st.container()

# Apply scenarios interactively
if "forecast_df" in st.session_state:
    apply_scenario = st.button("Apply Scenario", use_container_width=True, key="apply_scenario")
    
    if apply_scenario or "scenario_df" in st.session_state:
        # If we need to generate a new scenario
        if apply_scenario or "last_scenario_params" not in st.session_state or (
            st.session_state.get("last_scenario_params", {}).get("weekend_boost") != weekend_boost or
            st.session_state.get("last_scenario_params", {}).get("month_factor") != month_factor or
            st.session_state.get("last_scenario_params", {}).get("trend_adjustment") != trend_adjustment
        ):
            with scenario_container:
                with st.spinner("Applying scenario adjustments..."):
                    scenario_df = st.session_state["forecast_df"].copy()
                    
                    # Weekend boost
                    if weekend_boost != 0:
                        weekend_mask = scenario_df['dayofweek'] >= 5
                        scenario_df.loc[weekend_mask, 'Forecast'] *= (1 + weekend_boost/100)
                    
                    # Month factor
                    if month_factor != "Normal":
                        if month_factor == "Holiday Season (+20%)":
                            scenario_df['Forecast'] *= 1.2
                        elif month_factor == "Summer Sale (+15%)":
                            scenario_df['Forecast'] *= 1.15
                        elif month_factor == "Slow Season (-10%)":
                            scenario_df['Forecast'] *= 0.9
                    
                    # Trend adjustment
                    if trend_adjustment != 0:
                        # Apply gradually increasing adjustment
                        days = np.arange(len(scenario_df))
                        adjustment_factors = 1 + (days / max(1, len(days)-1)) * (trend_adjustment/100)
                        scenario_df['Forecast'] *= adjustment_factors
                    
                    # Recalculate bounds
                    z_score = 1.96 if confidence_level >= 0.95 else 2.58 if confidence_level >= 0.99 else 1.64
                    
                    for i in range(len(scenario_df)):
                        # Get volatility estimate
                        if i < 7:
                            std_dev = daily_sales['TotalPrice'].rolling(window=30).std().iloc[-1]
                        else:
                            prev_forecasts = scenario_df['Forecast'].iloc[:i].values
                            std_dev = np.std(prev_forecasts) if len(prev_forecasts) > 1 else daily_sales['TotalPrice'].std()
                        
                        # Calculate margin
                        margin = z_score * std_dev * np.sqrt(1 + i/10)
                        
                        # Update bounds
                        scenario_df.loc[i, 'Lower Bound'] = max(0, scenario_df.loc[i, 'Forecast'] - margin)
                        scenario_df.loc[i, 'Upper Bound'] = scenario_df.loc[i, 'Forecast'] + margin
                
                # Store the updated scenario
                st.session_state["scenario_df"] = scenario_df
                st.session_state["last_scenario_params"] = {
                    "weekend_boost": weekend_boost,
                    "month_factor": month_factor,
                    "trend_adjustment": trend_adjustment
                }
        else:
            # Use existing scenario
            scenario_df = st.session_state["scenario_df"]
        
        with scenario_container:
            # Create updated visualization
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=combined_df['Date'],
                y=combined_df['Historical'],
                mode='lines',
                name='Historical Sales',
                line=dict(color='royalblue', width=2)
            ))
            
            # Original forecast
            fig.add_trace(go.Scatter(
                x=st.session_state["forecast_df"]['Date'],
                y=st.session_state["forecast_df"]['Forecast'],
                mode='lines',
                name='Original Forecast',
                line=dict(color='firebrick', width=2, dash='dash')
            ))
            
            # Scenario forecast
            fig.add_trace(go.Scatter(
                x=scenario_df['Date'],
                y=scenario_df['Forecast'],
                mode='lines',
                name='Scenario Forecast',
                line=dict(color='green', width=2)
            ))
            
            # Scenario confidence interval
            fig.add_trace(go.Scatter(
                x=scenario_df['Date'].tolist() + scenario_df['Date'].tolist()[::-1],
                y=scenario_df['Upper Bound'].tolist() + scenario_df['Lower Bound'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 176, 80, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'Scenario {int(confidence_level*100)}% CI',
                showlegend=True
            ))
            
            fig.update_layout(
                title='Scenario Forecast Comparison',
                xaxis_title='Date',
                yaxis_title='Sales ($)',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario metrics
            orig_total = st.session_state["forecast_df"]['Forecast'].sum()
            scenario_total = scenario_df['Forecast'].sum()
            difference = ((scenario_total / orig_total) - 1) * 100
            
            # Show scenario summary metrics
            scenario_summary = pd.DataFrame({
                'Scenario': ['Original Forecast', 'Modified Scenario'],
                'Total Sales': [orig_total, scenario_total],
                'Avg Daily Sales': [
                    st.session_state["forecast_df"]['Forecast'].mean(),
                    scenario_df['Forecast'].mean()
                ],
                'Change (%)': [0, difference]
            })
            
            st.dataframe(
                scenario_summary.style.format({
                    'Total Sales': '${:,.2f}',
                    'Avg Daily Sales': '${:,.2f}',
                    'Change (%)': '{:+.2f}%'
                }),
                hide_index=True,
                use_container_width=True
            )
            
            st.metric(
                "Scenario Impact",
                f"${format_number(scenario_total)}",
                f"{difference:+.2f}% vs. original forecast",
                delta_color="normal" if difference >= 0 else "inverse"
            )
            
            # Describe the scenario
            scenario_description = []
            if weekend_boost != 0:
                scenario_description.append(f"Weekend sales adjusted by {weekend_boost:+}%")
            if month_factor != "Normal":
                scenario_description.append(f"Applied {month_factor}")
            if trend_adjustment != 0:
                scenario_description.append(f"Overall trend adjusted by {trend_adjustment:+}%")
                
            if scenario_description:
                st.info("Applied scenario: " + ", ".join(scenario_description))

# Download forecast data
st.header("Export Forecast Data")
st.markdown("Download the forecast data for further analysis")

col1, col2 = st.columns(2)

with col1:
    if "forecast_df" in st.session_state:
        forecast_csv = st.session_state["forecast_df"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Base Forecast CSV",
            data=forecast_csv,
            file_name="sales_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

with col2:
    if "scenario_df" in st.session_state:
        scenario_csv = st.session_state["scenario_df"].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scenario Forecast CSV",
            data=scenario_csv,
            file_name="scenario_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("📊 **Retail Analytics Dashboard** | Data sourced from UCI Machine Learning Repository")