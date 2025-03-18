import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data, format_number
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Market Analysis",
    page_icon="🌎",
    layout="wide"
)

# Load data
df, _ = load_data()

# Header
st.title("🌎 Market Analysis")
st.markdown("Geographical and temporal analysis of market performance")

# Prepare market data
@st.cache_data
def prepare_market_data():
    # Group by country
    country_data = df.groupby('Country').agg({
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    # Rename for clarity
    country_data = country_data.rename(columns={
        'InvoiceNo': 'OrderCount',
        'CustomerID': 'CustomerCount'
    })
    
    # Calculate metrics
    country_data['AvgOrderValue'] = country_data['TotalPrice'] / country_data['OrderCount']
    country_data['RevenuePerCustomer'] = country_data['TotalPrice'] / country_data['CustomerCount']
    
    # Time based metrics - monthly sales by country
    monthly_by_country = df.groupby([
        df['Year'], 
        df['Month'], 
        'Country'
    ])['TotalPrice'].sum().reset_index()
    
    monthly_by_country['YearMonth'] = monthly_by_country['Year'].astype(str) + "-" + monthly_by_country['Month'].astype(str).str.zfill(2)
    
    # Compute growth metrics
    growth_data = monthly_by_country.pivot_table(
        index='Country',
        columns='YearMonth',
        values='TotalPrice',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Get first and last periods
    columns = [col for col in growth_data.columns if col != 'Country']
    first_period = columns[0]
    last_period = columns[-1]
    
    # Calculate growth percentage
    growth_data['GrowthRate'] = ((growth_data[last_period] / growth_data[first_period]) - 1) * 100
    growth_data['GrowthRate'] = growth_data['GrowthRate'].replace([np.inf, -np.inf], np.nan)
    growth_data['GrowthRate'] = growth_data['GrowthRate'].fillna(0)
    
    # Merge growth data with country_data
    country_data = pd.merge(country_data, growth_data[['Country', 'GrowthRate']], on='Country', how='left')
    
    return country_data, monthly_by_country

# Get market data
country_data, monthly_by_country = prepare_market_data()

# Market Overview
st.header("Market Overview")
st.markdown("Key metrics about geographical market performance")

# Top markets
col1, col2, col3 = st.columns(3)

with col1:
    total_countries = country_data.shape[0]
    st.metric("Total Markets", f"{total_countries}")
    
with col2:
    top_market = country_data.loc[country_data['TotalPrice'].idxmax()]['Country']
    top_market_rev = country_data['TotalPrice'].max()
    st.metric("Top Market", f"{top_market}", f"${format_number(top_market_rev)}")
    
with col3:
    avg_growth = country_data['GrowthRate'].mean()
    st.metric("Avg. Market Growth", f"{avg_growth:.1f}%")

# Country map visualization
fig = px.choropleth(
    country_data, 
    locations="Country", 
    locationmode="country names",
    color="TotalPrice", 
    hover_name="Country",
    hover_data=["CustomerCount", "OrderCount", "AvgOrderValue", "GrowthRate"],
    color_continuous_scale=px.colors.sequential.Plasma,
    title="Global Sales Distribution",
    labels={
        "TotalPrice": "Total Sales ($)",
        "CustomerCount": "Customers",
        "OrderCount": "Orders",
        "AvgOrderValue": "Avg Order ($)",
        "GrowthRate": "Growth (%)"
    }
)

fig.update_layout(geo=dict(showframe=False, showcoastlines=True))
st.plotly_chart(fig, use_container_width=True)

# Top markets breakdown
st.subheader("Top Markets Analysis")

tab1, tab2, tab3 = st.tabs(["By Revenue", "By Customer Count", "By Growth"])

with tab1:
    top_revenue_countries = country_data.sort_values("TotalPrice", ascending=False).head(10)
    fig = px.bar(
        top_revenue_countries, 
        x="TotalPrice", 
        y="Country", 
        orientation="h",
        title="Top 10 Countries by Revenue",
        labels={"TotalPrice": "Total Revenue ($)", "Country": "Country"},
        color="TotalPrice",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    top_customer_countries = country_data.sort_values("CustomerCount", ascending=False).head(10)
    fig = px.bar(
        top_customer_countries, 
        x="CustomerCount", 
        y="Country", 
        orientation="h",
        title="Top 10 Countries by Customer Count",
        labels={"CustomerCount": "Number of Customers", "Country": "Country"},
        color="CustomerCount",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Filter out countries with few orders for meaningful growth rates
    growth_countries = country_data[country_data['OrderCount'] > 10].sort_values("GrowthRate", ascending=False).head(10)
    fig = px.bar(
        growth_countries, 
        x="GrowthRate", 
        y="Country", 
        orientation="h",
        title="Top 10 Countries by Growth Rate",
        labels={"GrowthRate": "Growth Rate (%)", "Country": "Country"},
        color="GrowthRate",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# Market Comparison
st.header("Market Comparison")
st.markdown("Comparing key metrics across different markets")

# Scatter plot with multiple dimensions
fig = px.scatter(
    country_data,
    x="CustomerCount",
    y="TotalPrice",
    size="OrderCount",
    color="AvgOrderValue",
    hover_name="Country",
    log_x=True,
    log_y=True,
    title="Market Comparison: Customer Base vs Revenue",
    labels={
        "CustomerCount": "Number of Customers (log scale)",
        "TotalPrice": "Total Revenue (log scale)",
        "OrderCount": "Number of Orders",
        "AvgOrderValue": "Average Order Value ($)"
    }
)

# Add trend line
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Market Growth Analysis
st.header("Market Growth Analysis")
st.markdown("Analyzing growth patterns across different markets")

# Select top markets for detailed analysis
top_countries = country_data.sort_values('TotalPrice', ascending=False)['Country'].head(10).tolist()
selected_countries = st.multiselect(
    "Select countries to analyze:",
    options=sorted(country_data['Country'].unique()),
    default=top_countries[:5]
)

if selected_countries:
    # Filter data for selected countries
    filtered_monthly = monthly_by_country[monthly_by_country['Country'].isin(selected_countries)]
    
    # Plot monthly trend by country
    fig = px.line(
        filtered_monthly,
        x="YearMonth",
        y="TotalPrice",
        color="Country",
        title="Monthly Sales Trend by Country",
        labels={"YearMonth": "Month", "TotalPrice": "Total Sales ($)", "Country": "Country"},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate comparison
    growth_comparison = country_data[country_data['Country'].isin(selected_countries)].sort_values('GrowthRate', ascending=False)
    
    fig = px.bar(
        growth_comparison,
        x="Country",
        y="GrowthRate",
        title="Growth Rate Comparison",
        labels={"GrowthRate": "Growth Rate (%)", "Country": "Country"},
        color="GrowthRate",
        color_continuous_scale=px.colors.diverging.RdYlGn
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select at least one country to analyze growth patterns.")

# Seasonality Analysis
st.header("Seasonal Market Patterns")
st.markdown("Exploring seasonal patterns across different markets")

# Group data by month across all countries
monthly_all = df.groupby(df['InvoiceDate'].dt.month)['TotalPrice'].sum().reset_index()
monthly_all['Month'] = monthly_all['InvoiceDate'].apply(lambda x: datetime(2000, x, 1).strftime('%B'))
monthly_all['MonthNum'] = monthly_all['InvoiceDate']

# Order by month
monthly_all = monthly_all.sort_values('MonthNum')

# Plot seasonal pattern
fig = px.line(
    monthly_all,
    x="Month",
    y="TotalPrice",
    title="Seasonal Sales Pattern (All Markets)",
    labels={"Month": "Month", "TotalPrice": "Total Sales ($)"},
    markers=True
)

# Add average line
avg_sales = monthly_all['TotalPrice'].mean()
fig.add_hline(y=avg_sales, line_dash="dash", line_color="red", 
              annotation_text="Average", annotation_position="top left")

st.plotly_chart(fig, use_container_width=True)

# Country comparison by weekday
st.subheader("Market Activity Patterns")

if selected_countries:
    # Filter data for selected countries
    df_selected = df[df['Country'].isin(selected_countries)]
    
    # Group by country and weekday
    weekday_data = df_selected.groupby(['Country', df_selected['InvoiceDate'].dt.dayofweek])['TotalPrice'].sum().reset_index()
    weekday_data['Day'] = weekday_data['InvoiceDate'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
        4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    fig = px.line(
        weekday_data,
        x="Day",
        y="TotalPrice",
        color="Country",
        title="Sales by Day of Week Across Markets",
        labels={"Day": "Day of Week", "TotalPrice": "Total Sales ($)", "Country": "Country"},
        markers=True,
        category_orders={"Day": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select countries above to analyze market activity patterns.")

# Market Opportunity Analysis section
st.header("Market Opportunity Analysis")
st.markdown("Identifying high-potential markets for business expansion")

# Create opportunity score based on growth, customer count, and average order value
opportunity_data = country_data.copy()

# Filter to countries with sufficient orders for reliable analysis
opportunity_data = opportunity_data[opportunity_data['OrderCount'] > 5]

# Create normalized metrics (0-100 scale)
opportunity_data['Growth_Score'] = 100 * (opportunity_data['GrowthRate'] - opportunity_data['GrowthRate'].min()) / (opportunity_data['GrowthRate'].max() - opportunity_data['GrowthRate'].min())
opportunity_data['AOV_Score'] = 100 * (opportunity_data['AvgOrderValue'] - opportunity_data['AvgOrderValue'].min()) / (opportunity_data['AvgOrderValue'].max() - opportunity_data['AvgOrderValue'].min())
opportunity_data['Size_Score'] = 100 * (opportunity_data['TotalPrice'] - opportunity_data['TotalPrice'].min()) / (opportunity_data['TotalPrice'].max() - opportunity_data['TotalPrice'].min())

# Calculate composite opportunity score - weighted average
opportunity_data['OpportunityScore'] = (
    0.5 * opportunity_data['Growth_Score'] + 
    0.3 * opportunity_data['AOV_Score'] + 
    0.2 * opportunity_data['Size_Score']
)

# Top opportunities
col1, col2 = st.columns([3, 2])

with col1:
    # Top opportunity markets
    top_opportunities = opportunity_data.sort_values('OpportunityScore', ascending=False).head(10)
    
    fig = px.bar(
        top_opportunities,
        x='OpportunityScore',
        y='Country',
        orientation='h',
        title='Top 10 Market Opportunities',
        color='OpportunityScore',
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={'OpportunityScore': 'Opportunity Score', 'Country': 'Country'},
        hover_data=['GrowthRate', 'AvgOrderValue', 'TotalPrice']
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Explanation of scoring
    st.subheader("Opportunity Score Components")
    st.markdown("""
    The Opportunity Score is calculated using:
    - 50% Growth Rate
    - 30% Average Order Value
    - 20% Market Size
    
    Higher scores indicate better expansion opportunities.
    """)
    
    # Show top market details
    if len(top_opportunities) > 0:
        top_market = top_opportunities.iloc[0]
        st.metric("Top Opportunity Market", top_market['Country'])
        st.write(f"**Growth Rate:** {top_market['GrowthRate']:.1f}%")
        st.write(f"**Avg. Order Value:** ${top_market['AvgOrderValue']:.2f}")
        st.write(f"**Total Revenue:** ${format_number(top_market['TotalPrice'])}")

# Opportunity quadrant analysis
st.subheader("Market Opportunity Matrix")

fig = px.scatter(
    opportunity_data,
    x="GrowthRate", 
    y="AvgOrderValue",
    size="TotalPrice",
    color="OpportunityScore",
    hover_name="Country",
    title="Market Opportunity Matrix: Growth vs. Value",
    labels={
        "GrowthRate": "Growth Rate (%)", 
        "AvgOrderValue": "Average Order Value ($)",
        "TotalPrice": "Total Revenue",
        "OpportunityScore": "Opportunity Score"
    },
    color_continuous_scale=px.colors.sequential.Viridis
)

# Add quadrant lines (using median values)
median_growth = opportunity_data['GrowthRate'].median()
median_aov = opportunity_data['AvgOrderValue'].median()

fig.add_hline(y=median_aov, line_dash="dash", line_color="gray")
fig.add_vline(x=median_growth, line_dash="dash", line_color="gray")

# Add quadrant labels
fig.add_annotation(x=opportunity_data['GrowthRate'].max()*0.75, y=opportunity_data['AvgOrderValue'].max()*0.75, 
                  text="High Growth, High Value", showarrow=False, font=dict(size=12))
fig.add_annotation(x=opportunity_data['GrowthRate'].min()*0.75, y=opportunity_data['AvgOrderValue'].max()*0.75, 
                  text="Low Growth, High Value", showarrow=False, font=dict(size=12))
fig.add_annotation(x=opportunity_data['GrowthRate'].max()*0.75, y=opportunity_data['AvgOrderValue'].min()*0.75, 
                  text="High Growth, Low Value", showarrow=False, font=dict(size=12))
fig.add_annotation(x=opportunity_data['GrowthRate'].min()*0.75, y=opportunity_data['AvgOrderValue'].min()*0.75, 
                  text="Low Growth, Low Value", showarrow=False, font=dict(size=12))

st.plotly_chart(fig, use_container_width=True)

# Recommendations section
st.subheader("Market Expansion Recommendations")

# Get top markets from different categories
top_growth = opportunity_data.sort_values('GrowthRate', ascending=False).head(3)['Country'].tolist()
top_value = opportunity_data.sort_values('AvgOrderValue', ascending=False).head(3)['Country'].tolist()
top_overall = opportunity_data.sort_values('OpportunityScore', ascending=False).head(3)['Country'].tolist()

st.markdown(f"""
### Recommended Action Plan:

1. **Primary Focus Markets** (Highest Overall Opportunity):
   - {', '.join(top_overall)}
   
2. **Growth-Focused Markets** (Highest Growth Rate):
   - {', '.join(top_growth)}
   
3. **Value-Focused Markets** (Highest Average Order):
   - {', '.join(top_value)}

These recommendations are based on historical sales data, growth trends, and customer value metrics.
""")

# Footer
st.markdown("---")
st.markdown("📊 **Retail Analytics Dashboard** | Data sourced from UCI Machine Learning Repository")