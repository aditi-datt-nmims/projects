```python
# Retail Sales Forecasting & Strategic Optimization
# Advanced time series forecasting with revenue optimization recommendations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Fetch real retail sales data
print("Fetching real retail sales data...")
url = "https://raw.githubusercontent.com/plotly/datasets/master/superstore_sales.csv"
df = pd.read_csv(url)

print(f"Dataset loaded: {len(df)} records")
print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")

# Data preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Quarter'] = df['Order Date'].dt.quarter

# Aggregate sales data
daily_sales = df.groupby('Order Date').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Order ID': 'count'
}).rename(columns={'Order ID': 'Orders'}).reset_index()

monthly_sales = df.groupby([df['Order Date'].dt.to_period('M')]).agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Order ID': 'count'
}).rename(columns={'Order ID': 'Orders'}).reset_index()
monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()

# Regional analysis
regional_performance = df.groupby('Region').agg({
    'Sales': ['sum', 'mean'],
    'Profit': ['sum', 'mean'],
    'Quantity': 'sum',
    'Order ID': 'count'
}).round(2)

# Category analysis
category_performance = df.groupby('Category').agg({
    'Sales': ['sum', 'mean'],
    'Profit': ['sum', 'mean'],
    'Discount': 'mean'
}).round(2)

# Sub-category top performers
subcategory_perf = df.groupby('Sub-Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values('Sales', ascending=False).head(10)

print("\n" + "="*60)
print("BASELINE MODEL - Simple Moving Average")
print("="*60)

# Baseline: Simple moving average (7-day)
daily_sales['SMA_Baseline'] = daily_sales['Sales'].rolling(window=7).mean()
daily_sales['Baseline_Error'] = abs(daily_sales['Sales'] - daily_sales['SMA_Baseline'])
baseline_mae = daily_sales['Baseline_Error'].mean()
print(f"Baseline MAE: ${baseline_mae:,.2f}")

print("\n" + "="*60)
print("IMPROVED MODEL - Triple Exponential Smoothing (Holt-Winters)")
print("="*60)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prepare data for Holt-Winters
monthly_sales_sorted = monthly_sales.sort_values('Order Date')
monthly_sales_sorted.set_index('Order Date', inplace=True)

# Split into train and test
train_size = int(len(monthly_sales_sorted) * 0.8)
train, test = monthly_sales_sorted[:train_size], monthly_sales_sorted[train_size:]

# Fit Holt-Winters model
model = ExponentialSmoothing(
    train['Sales'],
    seasonal_periods=12,
    trend='add',
    seasonal='add',
    initialization_method='estimated'
)
fitted_model = model.fit()

# Forecast
forecast_steps = len(test)
forecast = fitted_model.forecast(steps=forecast_steps)

# Calculate improved MAE
improved_mae = abs(test['Sales'] - forecast).mean()
print(f"Improved Model MAE: ${improved_mae:,.2f}")

# Calculate improvement
improvement = ((baseline_mae - improved_mae) / baseline_mae) * 100
print(f"\nFORECAST ACCURACY IMPROVEMENT: {improvement:.1f}%")

# Future forecast (next 6 months)
future_forecast = fitted_model.forecast(steps=6)
future_dates = pd.date_range(start=monthly_sales_sorted.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')

print("\n" + "="*60)
print("6-MONTH SALES FORECAST")
print("="*60)
for date, value in zip(future_dates, future_forecast):
    print(f"{date.strftime('%B %Y')}: ${value:,.2f}")

print("\n" + "="*60)
print("STRATEGIC RECOMMENDATIONS")
print("="*60)

# 1. Revenue Optimization by Region
print("\n1. REVENUE OPTIMIZATION BY REGION")
print("-" * 60)
regional_revenue = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
total_revenue = regional_revenue.sum()

for region, sales in regional_revenue.items():
    share = (sales / total_revenue) * 100
    regional_avg = df[df['Region'] == region]['Sales'].mean()
    print(f"\n{region}:")
    print(f"  - Total Revenue: ${sales:,.2f} ({share:.1f}% of total)")
    print(f"  - Avg Order Value: ${regional_avg:.2f}")
    
    if share < 20:
        print(f"  → RECOMMENDATION: Increase marketing spend by 15-20% to capture market share")
    else:
        print(f"  → RECOMMENDATION: Optimize operations and introduce premium products")

# 2. Product Mix Optimization
print("\n\n2. PRODUCT MIX OPTIMIZATION")
print("-" * 60)
category_profit_margin = df.groupby('Category').apply(
    lambda x: (x['Profit'].sum() / x['Sales'].sum() * 100)
).sort_values(ascending=False)

for category, margin in category_profit_margin.items():
    cat_sales = df[df['Category'] == category]['Sales'].sum()
    print(f"\n{category}:")
    print(f"  - Profit Margin: {margin:.1f}%")
    print(f"  - Total Sales: ${cat_sales:,.2f}")
    
    if margin > 15:
        print(f"  → RECOMMENDATION: Expand product line and increase inventory by 25%")
    elif margin > 10:
        print(f"  → RECOMMENDATION: Maintain current strategy, optimize pricing")
    else:
        print(f"  → RECOMMENDATION: Review discount strategy, reduce low-margin SKUs")

# 3. Market Expansion Opportunities
print("\n\n3. MARKET EXPANSION OPPORTUNITIES")
print("-" * 60)

# Analyze growth by segment
segment_growth = df.groupby('Segment').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Order ID': 'count'
}).sort_values('Sales', ascending=False)

for segment in segment_growth.index:
    seg_data = segment_growth.loc[segment]
    profit_margin = (seg_data['Profit'] / seg_data['Sales']) * 100
    
    print(f"\n{segment} Segment:")
    print(f"  - Revenue: ${seg_data['Sales']:,.2f}")
    print(f"  - Orders: {seg_data['Order ID']:,.0f}")
    print(f"  - Profit Margin: {profit_margin:.1f}%")
    
    if segment == 'Consumer':
        print(f"  → RECOMMENDATION: Launch e-commerce platform, target 30% online sales")
    elif segment == 'Corporate':
        print(f"  → RECOMMENDATION: Develop B2B subscription model for recurring revenue")
    else:
        print(f"  → RECOMMENDATION: Create educational discount program, partner with institutions")

# 4. Seasonal Strategy
print("\n\n4. SEASONAL REVENUE STRATEGY")
print("-" * 60)
quarterly_sales = df.groupby('Quarter')['Sales'].sum().sort_values(ascending=False)

for quarter, sales in quarterly_sales.items():
    print(f"\nQ{quarter}: ${sales:,.2f}")
    if quarter == quarterly_sales.idxmax():
        print(f"  → RECOMMENDATION: Increase inventory 40% ahead of peak season")
        print(f"  → Launch promotional campaigns 6 weeks in advance")
    else:
        print(f"  → RECOMMENDATION: Implement flash sales and clearance events")

# 5. Top Growth Sub-Categories
print("\n\n5. HIGH-GROWTH PRODUCT CATEGORIES")
print("-" * 60)
top_subcats = subcategory_perf.head(5)

for subcat, row in top_subcats.iterrows():
    profit_margin = (row['Profit'] / row['Sales']) * 100
    print(f"\n{subcat}:")
    print(f"  - Revenue: ${row['Sales']:,.2f}")
    print(f"  - Profit Margin: {profit_margin:.1f}%")
    print(f"  → RECOMMENDATION: Expand SKUs by 35%, introduce premium tier")

# Summary metrics
print("\n\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)
print(f"\nTotal Revenue: ${df['Sales'].sum():,.2f}")
print(f"Total Profit: ${df['Profit'].sum():,.2f}")
print(f"Overall Margin: {(df['Profit'].sum() / df['Sales'].sum() * 100):.1f}%")
print(f"Total Orders: {df['Order ID'].nunique():,}")
print(f"Average Order Value: ${df.groupby('Order ID')['Sales'].sum().mean():.2f}")

print(f"\n✓ Forecast Accuracy Improved by: {improvement:.1f}%")
print(f"✓ Projected 6-Month Revenue: ${future_forecast.sum():,.2f}")
print(f"✓ Expected Growth: {((future_forecast.sum() / monthly_sales_sorted['Sales'].tail(6).sum() - 1) * 100):.1f}%")

print("\n" + "="*60)

# Export results
results_summary = {
    'Baseline_MAE': baseline_mae,
    'Improved_MAE': improved_mae,
    'Accuracy_Improvement': improvement,
    'Next_6_Months_Forecast': future_forecast.sum(),
    'Top_Region': regional_revenue.idxmax(),
    'Top_Category': category_profit_margin.idxmax(),
    'Avg_Profit_Margin': (df['Profit'].sum() / df['Sales'].sum() * 100)
}

results_df = pd.DataFrame([results_summary])
results_df.to_csv('/mnt/user-data/outputs/forecast_results.csv', index=False)

# Detailed forecast output
forecast_output = pd.DataFrame({
    'Date': future_dates,
    'Forecasted_Sales': future_forecast
})
forecast_output.to_csv('/mnt/user-data/outputs/6_month_forecast.csv', index=False)

print("\nResults exported to:")
print("  - forecast_results.csv")
print("  - 6_month_forecast.csv")
```