import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, format_number
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📈",
    layout="wide"
)

# Load data
df, _ = load_data()

# Header
st.title("📈 Sales Analysis Dashboard")
st.markdown("Detailed analysis of sales trends, patterns, and key drivers")

# Date filter
min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
with col2:
    end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Filter data based on date range
filtered_df = df[(df['InvoiceDate'].dt.date >= start_date) & (df['InvoiceDate'].dt.date <= end_date)]

# Daily Sales Trend
st.subheader("Daily Sales Trend")

# Group by date
daily_sales = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['TotalPrice'].sum().reset_index()
daily_sales['InvoiceDate'] = pd.to_datetime(daily_sales['InvoiceDate'])

# Create figure with dual y-axis
fig = go.Figure()

# Add daily sales line
fig.add_trace(
    go.Scatter(
        x=daily_sales['InvoiceDate'],
        y=daily_sales['TotalPrice'],
        mode='lines',
        name='Daily Sales',
        line=dict(color='royalblue')
    )
)

# Add 7-day moving average
daily_sales['7_Day_MA'] = daily_sales['TotalPrice'].rolling(window=7).mean()
fig.add_trace(
    go.Scatter(
        x=daily_sales['InvoiceDate'],
        y=daily_sales['7_Day_MA'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='firebrick', dash='dash')
    )
)

# Set layout
fig.update_layout(
    title='Daily Sales with 7-Day Moving Average',
    xaxis_title='Date',
    yaxis_title='Total Sales ($)',
    hovermode='x unified',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Sales Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_daily_sales = daily_sales['TotalPrice'].mean()
    st.metric("Avg. Daily Sales", f"${format_number(avg_daily_sales)}")

with col2:
    max_day_sales = daily_sales['TotalPrice'].max()
    max_day = daily_sales.loc[daily_sales['TotalPrice'].idxmax(), 'InvoiceDate'].strftime('%Y-%m-%d')
    st.metric("Peak Sales Day", f"${format_number(max_day_sales)}", f"{max_day}")
    
with col3:
    avg_order_value = filtered_df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    st.metric("Avg. Order Value", f"${format_number(avg_order_value)}")

with col4:
    avg_items_per_order = filtered_df.groupby('InvoiceNo')['Quantity'].sum().mean()
    st.metric("Avg. Items per Order", f"{avg_items_per_order:.2f}")

# Sales by Hour and Day of Week
st.subheader("Sales Patterns")

col1, col2 = st.columns(2)

with col1:
    # Sales by Hour
    hourly_sales = filtered_df.groupby(filtered_df['InvoiceDate'].dt.hour)['TotalPrice'].sum().reset_index()
    
    fig = px.bar(
        hourly_sales,
        x='InvoiceDate',
        y='TotalPrice',
        title='Sales by Hour of Day',
        labels={'InvoiceDate': 'Hour of Day', 'TotalPrice': 'Total Sales ($)'},
        color_discrete_sequence=['#3366CC']
    )
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Sales by Day of Week
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_sales = filtered_df.groupby(filtered_df['InvoiceDate'].dt.dayofweek)['TotalPrice'].sum().reset_index()
    weekday_sales['Day'] = weekday_sales['InvoiceDate'].map(lambda x: weekday_names[x])
    
    fig = px.bar(
        weekday_sales,
        x='Day',
        y='TotalPrice',
        title='Sales by Day of Week',
        labels={'Day': 'Day of Week', 'TotalPrice': 'Total Sales ($)'},
        color_discrete_sequence=['#3366CC'],
        category_orders={"Day": weekday_names}
    )
    st.plotly_chart(fig, use_container_width=True)

# Monthly Analysis
st.subheader("Monthly Sales Analysis")

# Group by year and month
monthly_sales = filtered_df.groupby([filtered_df['Year'], filtered_df['Month']])[['TotalPrice', 'Quantity']].agg({
    'TotalPrice': 'sum',
    'Quantity': 'sum',
}).reset_index()

monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)

# Use tabs for different monthly metrics
tab1, tab2, tab3 = st.tabs(["Sales", "Items Sold", "Average Price"])

with tab1:
    fig = px.bar(
        monthly_sales, 
        x='YearMonth', 
        y='TotalPrice',
        title='Monthly Sales',
        labels={'YearMonth': 'Month', 'TotalPrice': 'Total Sales ($)'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.bar(
        monthly_sales, 
        x='YearMonth', 
        y='Quantity',
        title='Monthly Items Sold',
        labels={'YearMonth': 'Month', 'Quantity': 'Items Sold'},
        color_discrete_sequence=['#FF9900']
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    monthly_sales['AvgPrice'] = monthly_sales['TotalPrice'] / monthly_sales['Quantity']
    fig = px.line(
        monthly_sales, 
        x='YearMonth', 
        y='AvgPrice',
        title='Monthly Average Price per Item',
        labels={'YearMonth': 'Month', 'AvgPrice': 'Avg Price per Item ($)'},
        markers=True,
        color_discrete_sequence=['#33CC99']
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)