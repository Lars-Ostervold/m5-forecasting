import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from utils import load_data, format_number

# Page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
df, rfm = load_data()

# App title and description
st.title("📊 Retail Analytics Dashboard")
st.markdown("""
This dashboard provides insights from online retail transaction data, including:
- Sales trends and forecasting
- Customer segmentation analysis
- Product performance insights
- Market and geographic distribution
""")

# Display last update time
st.sidebar.info(f"Data last updated: {datetime.now().strftime('%Y-%m-%d')}")

# Main KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = df['TotalPrice'].sum()
    st.metric("Total Sales", f"${format_number(total_sales)}")

with col2:
    total_customers = df['CustomerID'].nunique()
    st.metric("Total Customers", format_number(total_customers))
    
with col3:
    total_products = df['StockCode'].nunique()
    st.metric("Total Products", format_number(total_products))
    
with col4:
    total_transactions = df['InvoiceNo'].nunique()
    st.metric("Total Transactions", format_number(total_transactions))

# Sales Overview
st.header("Sales Overview")

# Monthly Sales Trend
st.subheader("Monthly Sales Trend")
monthly_sales = df.groupby([df['Year'], df['Month']])['TotalPrice'].sum().reset_index()
monthly_sales['YearMonth'] = monthly_sales['Year'].astype(str) + '-' + monthly_sales['Month'].astype(str).str.zfill(2)

fig = px.line(
    monthly_sales, 
    x='YearMonth', 
    y='TotalPrice',
    title='Monthly Sales Trend',
    labels={'YearMonth': 'Month', 'TotalPrice': 'Total Sales ($)'},
    markers=True
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Split view for additional insights
col1, col2 = st.columns(2)

# Top 5 Countries
with col1:
    st.subheader("Top Countries by Sales")
    country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(5)
    
    fig = px.bar(
        country_sales,
        labels={'value': 'Total Sales ($)', 'Country': 'Country'},
        title="Top 5 Countries by Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

# Sales Distribution by Day of Week
with col2:
    st.subheader("Sales by Day of Week")
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_sales = df.groupby(df['InvoiceDate'].dt.dayofweek)['TotalPrice'].sum().reindex(range(7))
    
    fig = px.bar(
        x=weekday_names, 
        y=weekday_sales.values,
        labels={'x': 'Day of Week', 'y': 'Total Sales ($)'},
        title="Sales Distribution by Day of Week"
    )
    st.plotly_chart(fig, use_container_width=True)

# Customer RFM Distribution
st.header("Customer RFM Snapshot")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Recency Distribution")
    fig = px.histogram(
        rfm, 
        x='Recency',
        nbins=30,
        labels={'Recency': 'Days Since Last Purchase'},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Frequency Distribution") 
    fig = px.histogram(
        rfm,
        x='Frequency',
        nbins=30,
        labels={'Frequency': 'Number of Orders'},
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 50])
    st.plotly_chart(fig, use_container_width=True)
    
with col3:
    st.subheader("Monetary Distribution")
    fig = px.histogram(
        rfm,
        x='MonetaryValue',
        nbins=30,
        labels={'MonetaryValue': 'Total Customer Spend ($)'},
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0, 10000])
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("📊 **Retail Analytics Dashboard** | Data sourced from UCI Machine Learning Repository")