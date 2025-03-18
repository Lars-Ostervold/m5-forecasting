# ⚠️⚠️⚠️ Deployment is too big for the Streamlit community cloud. Will trim up a future version so it stops crashing :)
# Retail Analytics Dashboard

## Overview
This project is a fun exploration I built to enhance my skills in data science, machine learning, and interactive dashboard development. I wanted to create something that demonstrates end-to-end analytics capabilities using publicly available retail data, while learning modern tools and techniques used in the industry.

The requirements had to be split into two files since Streamlit gets angry when it has to install large packages like Tensorflow 😉

![Dashboard Preview](https://retailforecasting.streamlit.app/)

## Key Features

### 🔮 Advanced Sales Forecasting
- Time series forecasting using XGBoost with custom feature engineering
- Interactive forecasting parameters (confidence intervals, horizon length)
- What-if scenario testing to simulate market conditions
- Visualization of forecast drivers and feature importance

### 🛍️ Product Performance Analysis
- Comprehensive product metrics and rankings
- Market basket analysis with configurable parameters
- Interactive association rules visualization
- Network graph of product relationships

### 📈 Customer Segmentation
- RFM (Recency, Frequency, Monetary) analysis
- Customer lifetime value calculation
- Segment visualization and profiling
- Neural network-based customer clustering

### 🌎 Market Opportunity Analysis
- Geographic performance visualization
- Growth metrics by region
- Market opportunity scoring
- Interactive expansion recommendation engine

## Technologies Used
- **Python**: Core language for data processing and analysis
- **Pandas/NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Traditional machine learning algorithms
- **XGBoost**: Gradient boosting framework for forecasting
- **TensorFlow/PyTorch**: Neural network implementations
- **Streamlit**: Interactive web dashboard
- **Plotly/Matplotlib/Seaborn**: Data visualization
- **NetworkX/Pyvis**: Network visualization for association rules
- **MLxtend**: Market basket analysis

## What I Learned

This project was an amazing learning journey where I developed skills in:

1. **End-to-end data science workflow** - From data cleaning to model deployment in a cohesive process
2. **Time series forecasting** - Implementing advanced forecasting techniques with XGBoost, handling seasonality, and interpreting results
3. **Market basket analysis** - Discovering product associations and visualizing complex relationships
4. **Interactive dashboard design** - Creating intuitive, responsive user interfaces with Streamlit
5. **Data storytelling** - Presenting complex analytical findings in a clear, actionable format
6. **Machine learning deployment** - Taking models from notebooks to production-ready applications
7. **Performance optimization** - Handling larger datasets efficiently using caching and parallel processing
