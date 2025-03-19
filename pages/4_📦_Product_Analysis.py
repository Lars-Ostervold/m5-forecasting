import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, format_number
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# Page configuration
st.set_page_config(
    page_title="Product Analysis",
    page_icon="📦",
    layout="wide"
)

# Load data
df, _ = load_data()

# Header
st.title("📦 Product Analysis")
st.markdown("Detailed insights into product performance, clustering and market basket analysis")

# Prepare product data
@st.cache_data
def prepare_product_data():
    # Create product metrics
    product_metrics = df.groupby('StockCode').agg({
        'Description': 'first',
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique'
    }).reset_index()
    
    product_metrics = product_metrics.rename(columns={
        'InvoiceNo': 'OrderCount',
        'CustomerID': 'CustomerCount'
    })
    
    # Calculate average price, revenue per customer, etc.
    product_metrics['AvgPrice'] = product_metrics['TotalPrice'] / product_metrics['Quantity']
    product_metrics['RevenuePerCustomer'] = product_metrics['TotalPrice'] / product_metrics['CustomerCount']
    product_metrics['OrderFrequency'] = product_metrics['OrderCount'] / product_metrics['CustomerCount']
    
    # Filter out unusual products (outliers, non-product items)
    product_metrics = product_metrics[
        (product_metrics['Quantity'] > 0) & 
        (product_metrics['AvgPrice'] > 0) &
        (product_metrics['AvgPrice'] < 500)
    ]
    
    return product_metrics

# Prepare market basket data
@st.cache_data
def prepare_basket_data():
    # Create a pivot table: one row per order, one column per product, values are quantities
    basket = df.pivot_table(
        index='InvoiceNo',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    
    # Convert to binary: 1 if product was purchased, 0 otherwise
    basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)
    
    return basket_sets

# Get product data
product_metrics = prepare_product_data()

# Product Overview
st.header("Product Overview")
st.markdown("Key metrics about product performance")

# Filters
col1, col2 = st.columns([1, 3])
with col1:
    min_orders = st.slider("Min Orders", 1, 1000, 10)
    product_filtered = product_metrics[product_metrics['OrderCount'] >= min_orders]

# Top metrics
total_products = product_filtered.shape[0]
total_revenue = product_filtered['TotalPrice'].sum()
avg_price = product_filtered['TotalPrice'].sum() / product_filtered['Quantity'].sum()
top_product = product_filtered.loc[product_filtered['TotalPrice'].idxmax()]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Products", f"{total_products:,}")
with col2:
    st.metric("Total Revenue", f"${format_number(total_revenue)}")
with col3:
    st.metric("Avg. Product Price", f"${avg_price:.2f}")
with col4:
    st.metric("Top Product", f"{top_product['Description'][:20]}...")

# Product Analytics
st.subheader("Product Analytics")

# Product volume vs revenue chart
fig = px.scatter(
    product_filtered,
    x='Quantity',
    y='TotalPrice',
    size='CustomerCount',
    color='AvgPrice',
    hover_name='Description',
    log_x=True,
    log_y=True,
    title='Product Performance: Volume vs Revenue',
    labels={
        'Quantity': 'Total Quantity Sold (log scale)',
        'TotalPrice': 'Total Revenue (log scale)',
        'CustomerCount': 'Number of Customers',
        'AvgPrice': 'Average Price ($)'
    }
)

st.plotly_chart(fig, use_container_width=True)

# Product ranking
tab1, tab2, tab3 = st.tabs(["Top Products by Revenue", "Top Products by Volume", "Top Products by Customer Reach"])

with tab1:
    top_revenue = product_filtered.sort_values('TotalPrice', ascending=False).head(10)
    fig = px.bar(
        top_revenue,
        x='TotalPrice',
        y='Description',
        orientation='h',
        title='Top 10 Products by Revenue',
        color='TotalPrice',
        labels={'TotalPrice': 'Revenue ($)', 'Description': 'Product'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    top_volume = product_filtered.sort_values('Quantity', ascending=False).head(10)
    fig = px.bar(
        top_volume,
        x='Quantity',
        y='Description',
        orientation='h',
        title='Top 10 Products by Volume',
        color='Quantity',
        labels={'Quantity': 'Quantity Sold', 'Description': 'Product'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    top_reach = product_filtered.sort_values('CustomerCount', ascending=False).head(10)
    fig = px.bar(
        top_reach,
        x='CustomerCount',
        y='Description',
        orientation='h',
        title='Top 10 Products by Customer Reach',
        color='CustomerCount',
        labels={'CustomerCount': 'Number of Customers', 'Description': 'Product'},
        color_continuous_scale='Viridis'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Product Clustering
st.header("Product Clustering")
st.markdown("Automatic grouping of products based on their characteristics")

# Clustering
@st.cache_data
def cluster_products():
    # Select features for clustering
    features = ['AvgPrice', 'Quantity', 'CustomerCount', 'OrderFrequency']
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(product_metrics[features])
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    product_metrics['Cluster'] = kmeans.fit_predict(X)
    
    # Get cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Create a summary of each cluster
    cluster_summary = []
    for i in range(5):
        cluster_data = product_metrics[product_metrics['Cluster'] == i]
        summary = {
            'Cluster': i,
            'Size': len(cluster_data),
            'AvgPrice': cluster_data['AvgPrice'].mean(),
            'AvgQuantity': cluster_data['Quantity'].mean() / len(cluster_data),
            'AvgCustomerCount': cluster_data['CustomerCount'].mean(),
            'AvgOrderFrequency': cluster_data['OrderFrequency'].mean(),
            'TotalRevenue': cluster_data['TotalPrice'].sum()
        }
        cluster_summary.append(summary)
    
    cluster_df = pd.DataFrame(cluster_summary)
    
    # Assign cluster names based on characteristics
    def assign_cluster_name(row):
        if row['AvgPrice'] > 20 and row['AvgCustomerCount'] < 100:
            return "Premium Niche Products"
        elif row['AvgPrice'] > 10 and row['AvgCustomerCount'] > 200:
            return "Premium Popular Products"
        elif row['AvgPrice'] < 5 and row['AvgQuantity'] > 500:
            return "Budget Volume Sellers"
        elif row['AvgOrderFrequency'] > 1.5:
            return "Frequent Repurchase Items"
        else:
            return "Standard Catalog Products"
    
    cluster_df['ClusterName'] = cluster_df.apply(assign_cluster_name, axis=1)
    
    # Map cluster names back to products
    cluster_name_map = dict(zip(cluster_df['Cluster'], cluster_df['ClusterName']))
    product_metrics['ClusterName'] = product_metrics['Cluster'].map(cluster_name_map)
    
    return product_metrics, cluster_df

product_clustered, cluster_summary = cluster_products()

# Display cluster summary
st.subheader("Product Cluster Summary")
st.dataframe(
    cluster_summary[['ClusterName', 'Size', 'AvgPrice', 'AvgQuantity', 'AvgCustomerCount', 'TotalRevenue']].style.format({
        'AvgPrice': '${:.2f}',
        'AvgQuantity': '{:.1f}',
        'AvgCustomerCount': '{:.1f}',
        'TotalRevenue': '${:,.2f}'
    }),
    hide_index=True,
    use_container_width=True
)

# Visualize clusters
fig = px.scatter(
    product_clustered,
    x='AvgPrice',
    y='OrderFrequency',
    size='Quantity',
    color='ClusterName',
    hover_name='Description',
    title='Product Clusters: Price vs Order Frequency',
    labels={
        'AvgPrice': 'Average Price ($)',
        'OrderFrequency': 'Orders per Customer',
        'Quantity': 'Total Quantity',
        'ClusterName': 'Product Cluster'
    }
)

# Limit x-axis to better visualize clusters
fig.update_xaxes(range=[0, min(100, product_clustered['AvgPrice'].max())])
st.plotly_chart(fig, use_container_width=True)

st.warning("⚠️ In the GitHub repo there is a 'Market Basket Analysis' commented out here, but it uses too much memory to deploy on the Streamlit Community Cloud.")


# # Market Basket Analysis
# st.header("Market Basket Analysis")
# st.warning("⚠️ This analysis is not updating with the sliders currently. Might fix in future updates.")
# st.markdown("Discover which products are frequently purchased together")

# col1, col2 = st.columns(2)
# with col1:
#     min_support = st.slider("Minimum Support", 0.01, 0.1, 0.02, step=0.01,
#                           help="Minimum frequency of itemsets (higher value = more common patterns)")
# with col2:
#     min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.5, step=0.1,
#                              help="Minimum confidence for rules (higher value = stronger relationships)")

# @st.cache_data
# def get_basket_rules(_min_support, _min_confidence):
#     # Prepare basket data
#     basket_sets = prepare_basket_data()
    
#     # Run the Apriori algorithm
#     frequent_itemsets = apriori(basket_sets, min_support=_min_support, use_colnames=True)
    
#     # Generate association rules
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=_min_confidence)
    
#     # Add product descriptions
#     stock_to_desc = dict(zip(df['StockCode'], df['Description']))
    
#     def get_product_names(itemset):
#         return ', '.join([stock_to_desc.get(item, item) for item in itemset])
    
#     # Format rules for display
#     if len(rules) > 0:
#         rules['antecedents_str'] = rules['antecedents'].apply(lambda x: get_product_names(x))
#         rules['consequents_str'] = rules['consequents'].apply(lambda x: get_product_names(x))
#         return rules
#     else:
#         # Return empty dataframe with expected columns if no rules found
#         return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 
#                                     'lift', 'antecedents_str', 'consequents_str'])

# try:
#     # Get market basket rules
#     basket_rules = get_basket_rules(min_support, min_confidence)
    
#     if len(basket_rules) > 0:
#         # Display top association rules
#         st.subheader(f"Product Association Rules (Found: {len(basket_rules)})")
        
#         # Display table of top rules
#         st.dataframe(
#             basket_rules.sort_values('lift', ascending=False).head(10)[
#                 ['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']
#             ].rename(columns={
#                 'antecedents_str': 'Product(s) Purchased',
#                 'consequents_str': 'Often Purchased With',
#                 'support': 'Support',
#                 'confidence': 'Confidence',
#                 'lift': 'Lift'
#             }).style.format({
#                 'Support': '{:.3f}',
#                 'Confidence': '{:.3f}',
#                 'Lift': '{:.2f}'
#             }),
#             hide_index=True,
#             use_container_width=True
#         )
        
#         # Visualize top rules
#         top_rules = basket_rules.sort_values('lift', ascending=False).head(10)
        
#         fig = px.scatter(
#             top_rules,
#             x='support',
#             y='confidence',
#             size='lift',
#             hover_name='antecedents_str',
#             hover_data=['consequents_str', 'lift'],
#             title='Top Product Association Rules',
#             labels={
#                 'support': 'Support (frequency of itemset)',
#                 'confidence': 'Confidence (reliability of rule)',
#                 'lift': 'Lift (strength of relationship)'
#             }
#         )
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Network graph for rules
#         st.subheader("Product Association Network")
#         st.markdown("Network visualization of product relationships")
        
#         # Create network graph using NetworkX and Pyvis
#         import networkx as nx
#         from pyvis.network import Network
        
#         # Create network from association rules
#         G = nx.DiGraph()
        
#         # Add edges for top rules
#         for _, row in top_rules.iterrows():
#             source = row['antecedents_str']
#             target = row['consequents_str']
            
#             # Add nodes
#             if source not in G:
#                 G.add_node(source)
#             if target not in G:
#                 G.add_node(target)
                
#             # Add edge with lift as weight
#             G.add_edge(source, target, weight=row['lift'], title=f"Lift: {row['lift']:.2f}")
        
#         # Convert to pyvis network for interactive visualization
#         net = Network(height="600px", width="100%", directed=True)
        
#         # Copy nodes from networkx
#         for node in G.nodes():
#             # Truncate long labels
#             label = node[:30] + '...' if len(node) > 30 else node
#             net.add_node(node, label=label, title=node)
        
#         # Copy edges with weights
#         for edge in G.edges(data=True):
#             source, target, data = edge
#             weight = data.get('weight', 1.0)
#             title = data.get('title', '')
#             net.add_edge(source, target, value=weight, title=title)
        
#         # Set physics options for better visualization
#         net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=150)
        
#         # Save and display
#         net.save_graph("product_network.html")
#         with open("product_network.html", "r", encoding="utf-8") as f:
#             html_string = f.read()
            
#         st.components.v1.html(html_string, height=600)
#     else:
#         st.info("No strong association rules found with the current settings. Try lowering the support or confidence thresholds.")
# except Exception as e:
#     st.error(f"Error in market basket analysis: {str(e)}")
#     st.info("Try adjusting the support and confidence parameters or refreshing the page.")

# Footer
st.markdown("---")
st.markdown("📊 **Retail Analytics Dashboard** | Data sourced from UCI Machine Learning Repository")