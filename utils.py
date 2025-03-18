import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
import streamlit as st

@st.cache_data
def load_data():
    """Load and prepare the data files"""
    # Load cleaned retail data
    df = pd.read_csv('data/cleaned_retail_data.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Load RFM data
    rfm = pd.read_csv('data/rfm_data.csv')
    
    return df, rfm

@st.cache_resource
def load_models():
    """Load the trained models"""
    # Load XGBoost model
    xgb_model = joblib.load('models/xgb_sales_forecast_model.pkl')
    
    # Load encoder model
    encoder_model = tf.keras.models.load_model('models/customer_encoder_model.h5')
    
    # Load K-means model
    kmeans_model = joblib.load('models/customer_kmeans_model.pkl')
    
    # Load scaler
    scaler = joblib.load('models/rfm_scaler.pkl')
    
    return xgb_model, encoder_model, kmeans_model, scaler

def format_number(num):
    """Format numbers for display with K, M suffixes"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def predict_sales(model, input_data):
    """Generate sales forecast using the trained model"""
    return model.predict(input_data)

def get_customer_segment(encoder, kmeans, scaler, rfm_data):
    """Predict customer segment using autoencoder and kmeans"""
    # Scale the RFM data
    rfm_scaled = scaler.transform(rfm_data[['Recency', 'Frequency', 'MonetaryValue']])
    
    # Get encoded features
    encoded_features = encoder.predict(rfm_scaled)
    
    # Predict cluster
    cluster = kmeans.predict(encoded_features)
    
    return cluster

def create_radar_chart(df, categories, values, title):
    """Create a radar chart for customer segments"""
    import matplotlib.pyplot as plt
    from math import pi
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw y labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], size=10)
    plt.ylim(0, 1)
    
    # Plot each segment
    for i, row in df.iterrows():
        values = [row[cat] for cat in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Segment'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, size=15, y=1.1)
    
    return fig