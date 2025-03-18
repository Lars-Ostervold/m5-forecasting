import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras

def precompute_embeddings():
    """Pre-compute embeddings from the encoder model for all customers"""
    # Load RFM data
    rfm = pd.read_csv('data/rfm_data.csv')
    
    # Generate unique customer IDs if not present
    if 'CustomerID' not in rfm.columns:
        rfm['CustomerID'] = range(1, len(rfm) + 1)
    
    # Load scaler
    scaler = joblib.load('models/rfm_scaler.pkl')
    
    # Scale the RFM data
    rfm_scaled = scaler.transform(rfm[['Recency', 'Frequency', 'MonetaryValue']])
    
    # Load encoder model
    encoder_model = tf.keras.models.load_model('models/customer_encoder_model.h5')
    
    # Get the encoded features from the autoencoder
    encoded_features = encoder_model.predict(rfm_scaled)
    
    # Create a dictionary mapping customer IDs to embeddings
    embeddings_dict = {}
    for idx, customer_id in enumerate(rfm['CustomerID']):
        embeddings_dict[customer_id] = encoded_features[idx]
        
    # Save the embeddings dictionary
    joblib.dump(embeddings_dict, 'models/customer_embeddings.pkl')
    
    print("Embeddings pre-computed and saved successfully!")

if __name__ == "__main__":
    precompute_embeddings()