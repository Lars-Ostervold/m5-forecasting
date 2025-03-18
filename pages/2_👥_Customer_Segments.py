import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, load_models, create_radar_chart

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

# Load data and models
df, rfm = load_data()
_, encoder_model, kmeans_model, scaler = load_models()

# Header
st.title("👥 Customer Segmentation Analysis")
st.markdown("Interactive analysis of customer segments using RFM (Recency, Frequency, Monetary) metrics")

# Add encoder predictions to RFM data
@st.cache_data
def get_customer_segments():
    # Prepare RFM data
    rfm_features = rfm[['Recency', 'Frequency', 'MonetaryValue']]
    
    # Scale the features
    rfm_scaled = scaler.transform(rfm_features)
    
    # Encode features
    encoded_features = encoder_model.predict(rfm_scaled)
    
    # Cluster customers
    rfm_with_clusters = rfm.copy()
    rfm_with_clusters['Cluster'] = kmeans_model.predict(encoded_features)
    
    # Map cluster numbers to segment names
    segment_names = {
        0: "Champions",
        1: "Loyal Customers",
        2: "Potential Loyalists",
        3: "At Risk/Lost"
    }
    # Add segment names
    rfm_with_clusters['Segment'] = rfm_with_clusters['Cluster'].map(segment_names)
    
    # Create segment summary
    segment_summary = rfm_with_clusters.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean'
    }).reset_index()
    
    segment_summary['Count'] = rfm_with_clusters['Segment'].value_counts().values
    segment_summary['Percentage'] = 100 * segment_summary['Count'] / segment_summary['Count'].sum()
    return rfm_with_clusters, segment_summary, encoded_features

rfm_with_clusters, segment_summary, encoded_features = get_customer_segments()

# Display segment distribution
st.subheader("Customer Segment Distribution")

col1, col2 = st.columns([2, 3])

with col1:
    # Pie chart of segment distribution
    fig = px.pie(
        segment_summary,
        values='Count',
        names='Segment',
        title="Customer Segment Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Segment metrics table
    st.dataframe(
        segment_summary.style.format({
            'Recency': '{:.1f} days',
            'Frequency': '{:.1f} orders',
            'MonetaryValue': '${:.2f}',
            'Count': '{:,.0f}',
            'Percentage': '{:.1f}%'
        }),
        hide_index=True,
        use_container_width=True
    )

# Segment Comparison
st.subheader("Segment Comparisons")

# Normalize segment metrics for radar chart
radar_df = segment_summary.copy()
for col in ['Recency', 'Frequency', 'MonetaryValue']:
    if col == 'Recency':
        # For recency, lower is better so we invert the normalization
        max_val = radar_df[col].max()
        radar_df[col] = 1 - (radar_df[col] / max_val)
    else:
        max_val = radar_df[col].max()
        radar_df[col] = radar_df[col] / max_val

# Create radar chart
radar_fig = create_radar_chart(
    radar_df, 
    ['Recency', 'Frequency', 'MonetaryValue'], 
    radar_df[['Recency', 'Frequency', 'MonetaryValue']].values, 
    "Customer Segment Profiles"
)
st.pyplot(radar_fig)

# 2D Visualization of Customer Clusters
st.subheader("Customer Segment Visualization")

# Get 2D coordinates from encoder
X_embedded = encoded_features[:, :2]

# Create plot
fig = px.scatter(
    x=X_embedded[:, 0], 
    y=X_embedded[:, 1],
    color=rfm_with_clusters['Segment'],
    hover_data={
        'Recency': rfm_with_clusters['Recency'],
        'Frequency': rfm_with_clusters['Frequency'],
        'Monetary Value': rfm_with_clusters['MonetaryValue'],
    },
    title="Customer Segments in 2D Space",
    labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
    color_discrete_map={
        'Champions': '#36A2EB',
        'Loyal Customers': '#4BC0C0',
        'Potential Loyalists': '#FFCD56',
        'At Risk/Lost': '#FF6384'
    }
)
fig.update_layout(legend_title="Segment")
st.plotly_chart(fig, use_container_width=True)

# Segment Explorer
st.subheader("Customer Segment Explorer")

# Select segment to explore
selected_segment = st.selectbox(
    "Select customer segment to explore:",
    options=rfm_with_clusters['Segment'].unique()
)

# Filter data for selected segment
segment_data = rfm_with_clusters[rfm_with_clusters['Segment'] == selected_segment]

# Display segment statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Customers in Segment", f"{segment_data.shape[0]:,}")
    
with col2:
    avg_recency = segment_data['Recency'].mean()
    st.metric("Avg. Recency", f"{avg_recency:.1f} days")
    
with col3:
    avg_frequency = segment_data['Frequency'].mean()
    st.metric("Avg. Frequency", f"{avg_frequency:.1f} orders")
    
with col4:
    avg_monetary = segment_data['MonetaryValue'].mean()
    st.metric("Avg. Monetary Value", f"${avg_monetary:.2f}")

# Display 3D RFM plot for the segment
fig = px.scatter_3d(
    segment_data,
    x='Recency',
    y='Frequency',
    z='MonetaryValue',
    color='MonetaryValue',
    color_continuous_scale=px.colors.sequential.Viridis,
    labels={
        'Recency': 'Recency (days)',
        'Frequency': 'Frequency (orders)',
        'MonetaryValue': 'Monetary Value ($)'
    },
    title=f"3D RFM Plot for {selected_segment} Segment"
)
fig.update_layout(height=700)
st.plotly_chart(fig, use_container_width=True)

# Customer details
st.subheader(f"Top Customers in {selected_segment} Segment")
top_customers = segment_data.sort_values(by='MonetaryValue', ascending=False).head(10).reset_index(drop=True)
st.dataframe(
    top_customers[['Recency', 'Frequency', 'MonetaryValue']].style.format({
        'Recency': '{:.0f} days',
        'Frequency': '{:.0f} orders',
        'MonetaryValue': '${:.2f}'
    }),
    hide_index=False,
    use_container_width=True
)