import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# This code should be inserted after the existing code in your Streamlit app
# Add it at the end of the file

# Pre vs Post Test Analysis Section
st.write("## 🔄 Pre vs Post Test Analysis (Test Groups Only)")
st.write("This analysis compares test group performance before and after the test implementation, ignoring control groups.")

# Function to calculate pre vs post metrics for a test group
def calculate_pre_post_metrics(df, cohort, pre_days=7):
    """Calculate metrics for test groups before and after test implementation"""
    if cohort != "All Cohorts":
        cohort_df = df[df['cohort'] == cohort]
        start_date = test_start_dates.get(cohort, df['date'].min())
    else:
        cohort_df = df.copy()
        # For all cohorts, use the earliest test start date
        start_date = min([date for date in test_start_dates.values()])
    
    # Only include test groups, exclude control
    test_df = cohort_df[cohort_df['data_set'] != control_group]
    
    # Determine pre-period (n days before test start)
    pre_start_date = start_date - timedelta(days=pre_days)
    pre_end_date = start_date - timedelta(days=1)
    
    # Filter data for pre and post periods
    pre_df = test_df[(test_df['date'] >= pre_start_date) & (test_df['date'] <= pre_end_date)]
    post_df = test_df[test_df['date'] >= start_date]
    
    pre_post_metrics = {}
    
    # Calculate metrics for each test group
    for test_group in test_df['data_set'].unique():
        test_group_pre = pre_df[pre_df['data_set'] == test_group]
        test_group_post = post_df[post_df['data_set'] == test_group]
        
        # Skip if we don't have data for either period
        if len(test_group_pre) == 0 or len(test_group_post) == 0:
            continue
        
        # Calculate average daily metrics for both periods
        pre_days_count = (test_group_pre['date'].max() - test_group_pre['date'].min()).days + 1
        post_days_count = (test_group_post['date'].max() - test_group_post['date'].min()).days + 1
        
        # Avoid division by zero
        pre_days_count = max(pre_days_count, 1)
        post_days_count = max(post_days_count, 1)
        
        # Pre-period metrics (daily average)
        pre_gmv = test_group_pre['gmv'].sum() / pre_days_count
        pre_app_opens = test_group_pre['app_opens'].sum() / pre_days_count
        pre_orders = test_group_pre['orders'].sum() / pre_days_count
        pre_transactors = test_group_pre['transactors'].sum() / pre_days_count
        
        # Post-period metrics (daily average)
        post_gmv = test_group_post['gmv'].sum() / post_days_count
        post_app_opens = test_group_post['app_opens'].sum() / post_days_count
        post_orders = test_group_post['orders'].sum() / post_days_count
        post_transactors = test_group_post['transactors'].sum() / post_days_count
        
        # Calculate percentage changes
        gmv_pct_change = ((post_gmv - pre_gmv) / pre_gmv * 100) if pre_gmv > 0 else float('inf')
        app_opens_pct_change = ((post_app_opens - pre_app_opens) / pre_app_opens * 100) if pre_app_opens > 0 else float('inf')
        orders_pct_change = ((post_orders - pre_orders) / pre_orders * 100) if pre_orders > 0 else float('inf')
        transactors_pct_change = ((post_transactors - pre_transactors) / pre_transactors * 100) if pre_transactors > 0 else float('inf')
        
        # Calculate absolute increases
        gmv_increase = post_gmv - pre_gmv
        app_opens_increase = post_app_opens - pre_app_opens
        orders_increase = post_orders - pre_orders
        transactors_increase = post_transactors - pre_transactors
        
        # Store metrics
        pre_post_metrics[test_group] = {
            'pre_gmv': pre_gmv,
            'post_gmv': post_gmv,
            'gmv_increase': gmv_increase,
            'gmv_pct_change': gmv_pct_change,
            
            'pre_app_opens': pre_app_opens,
            'post_app_opens': post_app_opens,
            'app_opens_increase': app_opens_increase,
            'app_opens_pct_change': app_opens_pct_change,
            
            'pre_orders': pre_orders,
            'post_orders': post_orders,
            'orders_increase': orders_increase,
            'orders_pct_change': orders_pct_change,
            
            'pre_transactors': pre_transactors,
            'post_transactors': post_transactors,
            'transactors_increase': transactors_increase,
            'transactors_pct_change': transactors_pct_change,
            
            'pre_days': pre_days_count,
            'post_days': post_days_count
        }
    
    return pre_post_metrics

# Set the number of days to look back for pre-test analysis
pre_test_days = st.slider("Number of days before test to analyze (Pre-period)", min_value=3, max_value=30, value=7)

# Calculate pre-post metrics based on selections
pre_post_summary = {}

if selected_cohort == "All Cohorts":
    # Calculate for each cohort individually
    for cohort in df['cohort'].unique():
        cohort_df = filtered_df[filtered_df['cohort'] == cohort]
        if len(cohort_df) > 0:
            pre_post_summary[cohort] = calculate_pre_post_metrics(cohort_df, cohort, pre_test_days)
else:
    # Calculate for selected cohort
    pre_post_summary[selected_cohort] = calculate_pre_post_metrics(filtered_df, selected_cohort, pre_test_days)

# Display results in a nice format
st.write("### 📊 Pre vs Post Test Metrics Summary (Daily Average)")

# Summary table
pre_post_rows = []

for cohort, test_groups_metrics in pre_post_summary.items():
    for test_group, metrics in test_groups_metrics.items():
        # Handle infinite percentage changes
        for metric in ['gmv_pct_change', 'app_opens_pct_change', 'orders_pct_change', 'transactors_pct_change']:
            if metrics[metric] == float('inf'):
                metrics[metric] = "N/A (zero in pre-period)"
            elif isinstance(metrics[metric], (int, float)):
                # Keep the value as is
                pass
    
        pre_post_rows.append({
            'Cohort': cohort,
            'Test Group': test_group,
            'Pre-Period (Days)': metrics['pre_days'],
            'Post-Period (Days)': metrics['post_days'],
            'Pre GMV (Daily Avg ₹)': round(metrics['pre_gmv'], 2),
            'Post GMV (Daily Avg ₹)': round(metrics['post_gmv'], 2),
            'GMV Change (%)': metrics['gmv_pct_change'] if isinstance(metrics['gmv_pct_change'], str) else round(metrics['gmv_pct_change'], 2),
            'Pre App Opens (Daily Avg)': round(metrics['pre_app_opens'], 1),
            'Post App Opens (Daily Avg)': round(metrics['post_app_opens'], 1),
            'App Opens Change (%)': metrics['app_opens_pct_change'] if isinstance(metrics['app_opens_pct_change'], str) else round(metrics['app_opens_pct_change'], 2),
            'Pre Orders (Daily Avg)': round(metrics['pre_orders'], 1),
            'Post Orders (Daily Avg)': round(metrics['post_orders'], 1),
            'Orders Change (%)': metrics['orders_pct_change'] if isinstance(metrics['orders_pct_change'], str) else round(metrics['orders_pct_change'], 2),
            'Pre Transactors (Daily Avg)': round(metrics['pre_transactors'], 1),
            'Post Transactors (Daily Avg)': round(metrics['post_transactors'], 1),
            'Transactors Change (%)': metrics['transactors_pct_change'] if isinstance(metrics['transactors_pct_change'], str) else round(metrics['transactors_pct_change'], 2)
        })

if pre_post_rows:
    pre_post_df = pd.DataFrame(pre_post_rows)
    
    # Just display the dataframe
    st.dataframe(pre_post_df)
    
    # Create visualizations
    st.write("### 📈 Pre vs Post Test Visual Comparison")
    
    # Prepare data for visualization - melt to long format
    metrics_to_plot = [
        ('GMV (Daily Avg ₹)', 'Pre GMV (Daily Avg ₹)', 'Post GMV (Daily Avg ₹)'),
        ('App Opens (Daily Avg)', 'Pre App Opens (Daily Avg)', 'Post App Opens (Daily Avg)'),
        ('Orders (Daily Avg)', 'Pre Orders (Daily Avg)', 'Post Orders (Daily Avg)'),
        ('Transactors (Daily Avg)', 'Pre Transactors (Daily Avg)', 'Post Transactors (Daily Avg)')
    ]
    
    for metric_name, pre_col, post_col in metrics_to_plot:
        # Create a visualization for each metric
        viz_data = []
        
        for _, row in pre_post_df.iterrows():
            viz_data.append({
                'Cohort': row['Cohort'],
                'Test Group': row['Test Group'],
                'Period': 'Pre-Test',
                metric_name: row[pre_col]
            })
            viz_data.append({
                'Cohort': row['Cohort'],
                'Test Group': row['Test Group'],
                'Period': 'Post-Test',
                metric_name: row[post_col]
            })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create grouped bar chart
        fig = px.bar(
            viz_df, 
            x='Test Group', 
            y=metric_name,
            color='Period',
            barmode='group',
            facet_col='Cohort' if len(pre_post_df['Cohort'].unique()) > 1 else None,
            title=f"Pre vs Post {metric_name} by Test Group",
            color_discrete_map={'Pre-Test': '#636EFA', 'Post-Test': '#EF553B'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Percentage change visualization
    change_metrics = [
        ('GMV Change (%)', 'GMV'),
        ('App Opens Change (%)', 'App Opens'),
        ('Orders Change (%)', 'Orders'),
        ('Transactors Change (%)', 'Transactors')
    ]
    
    # Prepare data for visualization - remove non-numeric values
    change_data = []
    for _, row in pre_post_df.iterrows():
        for col, label in change_metrics:
            if isinstance(row[col], (int, float)):
                change_data.append({
                    'Cohort': row['Cohort'],
                    'Test Group': row['Test Group'],
                    'Metric': label,
                    'Percentage Change': row[col]
                })
    
    if change_data:
        change_df = pd.DataFrame(change_data)
        
        # Create grouped bar chart for percentage changes
        fig_change = px.bar(
            change_df, 
            x='Metric', 
            y='Percentage Change',
            color='Test Group',
            barmode='group',
            facet_col='Cohort' if len(pre_post_df['Cohort'].unique()) > 1 else None,
            title="Percentage Change in Metrics (Pre vs Post)",
            text_auto='.1f'
        )
        
        # Add a horizontal line at y=0
        fig_change.add_shape(
            type="line",
            x0=-0.5,
            x1=3.5,
            y0=0,
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
        )
        
        st.plotly_chart(fig_change, use_container_width=True)
    
    # Download button for CSV export
    pre_post_csv = pre_post_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Pre vs Post Analysis as CSV",
        data=pre_post_csv,
        file_name="pre_post_test_analysis.csv",
        mime="text/csv",
    )
    
    # Summary insights
    st.write("### 💡 Key Insights from Pre vs Post Analysis")
    
    # Extract key insights
    insights = []
    
    for cohort, test_groups_metrics in pre_post_summary.items():
        for test_group, metrics in test_groups_metrics.items():
            # Check for significant changes (>10% change)
            significant_metrics = []
            
            if isinstance(metrics['gmv_pct_change'], (int, float)) and abs(metrics['gmv_pct_change']) > 10:
                direction = "increase" if metrics['gmv_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"GMV showed a {abs(metrics['gmv_pct_change']):.1f}% {direction}")
            
            if isinstance(metrics['app_opens_pct_change'], (int, float)) and abs(metrics['app_opens_pct_change']) > 10:
                direction = "increase" if metrics['app_opens_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"App Opens showed a {abs(metrics['app_opens_pct_change']):.1f}% {direction}")
            
            if isinstance(metrics['orders_pct_change'], (int, float)) and abs(metrics['orders_pct_change']) > 10:
                direction = "increase" if metrics['orders_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"Orders showed a {abs(metrics['orders_pct_change']):.1f}% {direction}")
            
            if isinstance(metrics['transactors_pct_change'], (int, float)) and abs(metrics['transactors_pct_change']) > 10:
                direction = "increase" if metrics['transactors_pct_change'] > 0 else "decrease"
                significant_metrics.append(f"Transactors showed a {abs(metrics['transactors_pct_change']):.1f}% {direction}")
            
            if significant_metrics:
                insights.append(f"**{cohort} - {test_group}**: {'; '.join(significant_metrics)}.")
    
    if insights:
        for insight in insights:
            st.markdown(f"- {insight}")
    else:
        st.write("No significant changes detected between pre and post periods.")
else:
    st.write("No data available for pre vs post analysis with the selected filters.")
