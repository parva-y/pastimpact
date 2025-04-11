import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", page_title="Pre vs Post Test Analysis", page_icon="📊")

# App title and description
st.title("Pre vs Post Incremental Analysis")
st.markdown("""
This application analyzes test group performance before and after test implementation, focusing on key business metrics.
Upload your experiment data to see the incremental impact of your tests without requiring control group comparisons.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your experiment data CSV", type=['csv'])

# Function to process the uploaded file
def process_uploaded_file(file):
    # Read the CSV
    df = pd.read_csv(file, parse_dates=['date'])
    return df

# Main app logic
if uploaded_file is not None:
    # Process the file
    df = process_uploaded_file(uploaded_file)
    
    # Display DataFrame overview
    st.write("### Data Overview")
    st.write(f"**Total rows:** {len(df)} | **Date range:** {df['date'].min()} to {df['date'].max()}")
    st.dataframe(df.head(), use_container_width=True)
    
    # Check required columns
    required_columns = {'date', 'data_set', 'audience_size', 'app_opens', 'transactors', 'orders', 'gmv', 'cohort'}
    missing_columns = required_columns - set(df.columns)
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()
    
    # Check if Recency data is available
    has_recency_data = 'Recency' in df.columns
    
    # Get unique values for filters
    cohorts = sorted(df['cohort'].unique())
    test_groups = sorted([g for g in df['data_set'].unique() if g != "Control Set"])
    control_group = "Control Set" if "Control Set" in df['data_set'].unique() else None
    
    # Sidebar for test configuration
    st.sidebar.header("Test Configuration")
    
    # Create dictionary to store test start dates
    test_start_dates = {}
    
    # Setup for each cohort's test date
    st.sidebar.subheader("Test Start Dates")
    for cohort in cohorts:
        # Default date is the minimum date in the dataset for this cohort
        cohort_df = df[df['cohort'] == cohort]
        default_date = cohort_df['date'].min() if not cohort_df.empty else df['date'].min()
        
        # Allow user to set the test start date for each cohort
        test_start_date = st.sidebar.date_input(
            f"Start date for {cohort}", 
            value=pd.to_datetime(default_date),
            min_value=pd.to_datetime(df['date'].min()),
            max_value=pd.to_datetime(df['date'].max())
        )
        test_start_dates[cohort] = pd.Timestamp(test_start_date)
    
    # Cohort selection with default "All Cohorts" option
    st.sidebar.subheader("Data Filters")
    cohort_options = ["All Cohorts"] + list(cohorts)
    selected_cohort = st.sidebar.selectbox("Select Cohort", cohort_options)
    
    # Recency selector with default "All Recency" option
    if has_recency_data:
        recency_values = sorted(df['Recency'].unique())
        recency_options = ["All Recency"] + list(recency_values)
        selected_recency = st.sidebar.selectbox("Select Recency", recency_options)
    else:
        selected_recency = "All Recency"
    
    # Data selection options
    st.sidebar.subheader("Analysis Settings")
    pre_test_days = st.sidebar.slider("Days before test to analyze (Pre-period)", 
                                     min_value=3, max_value=30, value=7)
    
    exclude_control = st.sidebar.checkbox("Exclude Control Group", value=True)
    normalize_by_audience = st.sidebar.checkbox("Normalize by Audience Size", value=True)
    
    # Filter data based on selections
    def filter_data(df, cohort, recency):
        # Start with base filtered data
        if cohort != "All Cohorts":
            filtered_df = df[df['cohort'] == cohort]
            start_date = test_start_dates.get(cohort, df['date'].min())
        else:
            filtered_df = df.copy()
            # For all cohorts, use the earliest test start date
            start_date = min([date for date in test_start_dates.values()])
        
        # Apply recency filter if needed
        if recency != "All Recency" and has_recency_data:
            filtered_df = filtered_df[filtered_df['Recency'] == recency]
        
        # Exclude control group if requested
        if exclude_control and control_group:
            filtered_df = filtered_df[filtered_df['data_set'] != control_group]
        
        return filtered_df, start_date
    
    filtered_df, selected_start_date = filter_data(df, selected_cohort, selected_recency)
    
    # Function to calculate pre vs post metrics for a test group
    def calculate_pre_post_metrics(df, cohort, pre_days=7, normalize=True):
        """Calculate metrics for test groups before and after test implementation"""
        if cohort != "All Cohorts":
            cohort_df = df[df['cohort'] == cohort]
            start_date = test_start_dates.get(cohort, df['date'].min())
        else:
            cohort_df = df.copy()
            # For all cohorts, use the earliest test start date
            start_date = min([date for date in test_start_dates.values()])
        
        # Exclude control group if requested
        if exclude_control and control_group:
            test_df = cohort_df[cohort_df['data_set'] != control_group]
        else:
            test_df = cohort_df
        
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
            
            # Get audience sizes
            pre_audience = test_group_pre['audience_size'].sum()
            post_audience = test_group_post['audience_size'].sum()
            
            # Calculate metrics (can be normalized by audience size)
            if normalize and pre_audience > 0 and post_audience > 0:
                # Normalized metrics
                pre_gmv = (test_group_pre['gmv'].sum() / pre_audience) * 1000  # per 1000 users
                pre_app_opens = (test_group_pre['app_opens'].sum() / pre_audience) * 1000
                pre_orders = (test_group_pre['orders'].sum() / pre_audience) * 1000
                pre_transactors = (test_group_pre['transactors'].sum() / pre_audience) * 1000
                
                post_gmv = (test_group_post['gmv'].sum() / post_audience) * 1000
                post_app_opens = (test_group_post['app_opens'].sum() / post_audience) * 1000
                post_orders = (test_group_post['orders'].sum() / post_audience) * 1000
                post_transactors = (test_group_post['transactors'].sum() / post_audience) * 1000
            else:
                # Raw daily averages
                pre_gmv = test_group_pre['gmv'].sum() / pre_days_count
                pre_app_opens = test_group_pre['app_opens'].sum() / pre_days_count
                pre_orders = test_group_pre['orders'].sum() / pre_days_count
                pre_transactors = test_group_pre['transactors'].sum() / pre_days_count
                
                post_gmv = test_group_post['gmv'].sum() / post_days_count
                post_app_opens = test_group_post['app_opens'].sum() / post_days_count
                post_orders = test_group_post['orders'].sum() / post_days_count
                post_transactors = test_group_post['transactors'].sum() / post_days_count
            
            # Calculate percentage changes
            gmv_pct_change = ((post_gmv - pre_gmv) / pre_gmv * 100) if pre_gmv > 0 else np.nan
            app_opens_pct_change = ((post_app_opens - pre_app_opens) / pre_app_opens * 100) if pre_app_opens > 0 else np.nan
            orders_pct_change = ((post_orders - pre_orders) / pre_orders * 100) if pre_orders > 0 else np.nan
            transactors_pct_change = ((post_transactors - pre_transactors) / pre_transactors * 100) if pre_transactors > 0 else np.nan
            
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
                'post_days': post_days_count,
                'pre_audience': pre_audience,
                'post_audience': post_audience
            }
        
        return pre_post_metrics
    
    # Calculate pre-post metrics based on selections
    pre_post_summary = {}
    
    if selected_cohort == "All Cohorts":
        # Calculate for each cohort individually
        for cohort in cohorts:
            cohort_df = filtered_df[filtered_df['cohort'] == cohort]
            if len(cohort_df) > 0:
                pre_post_summary[cohort] = calculate_pre_post_metrics(
                    cohort_df, cohort, pre_test_days, normalize_by_audience
                )
    else:
        # Calculate for selected cohort
        pre_post_summary[selected_cohort] = calculate_pre_post_metrics(
            filtered_df, selected_cohort, pre_test_days, normalize_by_audience
        )
    
    # Display results
    metric_label = " (per 1000 users)" if normalize_by_audience else " (Daily Avg)"
    
    st.write(f"## 📊 Pre vs Post Test Analysis")
    st.write(f"Comparing metrics before and after test implementation for {selected_cohort if selected_cohort != 'All Cohorts' else 'all cohorts'}.")
    
    if selected_recency != "All Recency":
        st.write(f"### Recency: {selected_recency}")
    
    # Summary table
    pre_post_rows = []
    
    for cohort, test_groups_metrics in pre_post_summary.items():
        for test_group, metrics in test_groups_metrics.items():
            # Format percentage changes
            gmv_pct = "N/A" if np.isnan(metrics['gmv_pct_change']) else f"{metrics['gmv_pct_change']:.2f}%"
            app_opens_pct = "N/A" if np.isnan(metrics['app_opens_pct_change']) else f"{metrics['app_opens_pct_change']:.2f}%"
            orders_pct = "N/A" if np.isnan(metrics['orders_pct_change']) else f"{metrics['orders_pct_change']:.2f}%"
            transactors_pct = "N/A" if np.isnan(metrics['transactors_pct_change']) else f"{metrics['transactors_pct_change']:.2f}%"
            
            pre_post_rows.append({
                'Cohort': cohort,
                'Test Group': test_group,
                'Pre-Period (Days)': metrics['pre_days'],
                'Post-Period (Days)': metrics['post_days'],
                'Pre-Audience Size': metrics['pre_audience'],
                'Post-Audience Size': metrics['post_audience'],
                f'Pre GMV{metric_label}': round(metrics['pre_gmv'], 2),
                f'Post GMV{metric_label}': round(metrics['post_gmv'], 2),
                'GMV Change (%)': gmv_pct,
                f'Pre App Opens{metric_label}': round(metrics['pre_app_opens'], 1),
                f'Post App Opens{metric_label}': round(metrics['post_app_opens'], 1),
                'App Opens Change (%)': app_opens_pct,
                f'Pre Orders{metric_label}': round(metrics['pre_orders'], 1),
                f'Post Orders{metric_label}': round(metrics['post_orders'], 1),
                'Orders Change (%)': orders_pct,
                f'Pre Transactors{metric_label}': round(metrics['pre_transactors'], 1),
                f'Post Transactors{metric_label}': round(metrics['post_transactors'], 1),
                'Transactors Change (%)': transactors_pct
            })
    
    if pre_post_rows:
        pre_post_df = pd.DataFrame(pre_post_rows)
        
        # Display the dataframe
        st.dataframe(pre_post_df, use_container_width=True)
        
        # Create visualizations
        st.write("## 📈 Pre vs Post Test Visual Comparison")
        
        # Create tabs for different visualization types
        tab1, tab2 = st.tabs(["Metrics Comparison", "Percentage Changes"])
        
        with tab1:
            # Prepare data for visualization - melt to long format
            metrics_to_plot = [
                (f'GMV{metric_label}', f'Pre GMV{metric_label}', f'Post GMV{metric_label}'),
                (f'App Opens{metric_label}', f'Pre App Opens{metric_label}', f'Post App Opens{metric_label}'),
                (f'Orders{metric_label}', f'Pre Orders{metric_label}', f'Post Orders{metric_label}'),
                (f'Transactors{metric_label}', f'Pre Transactors{metric_label}', f'Post Transactors{metric_label}')
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
        
        with tab2:
            # Percentage change visualization
            change_metrics = [
                ('GMV Change (%)', 'GMV'),
                ('App Opens Change (%)', 'App Opens'),
                ('Orders Change (%)', 'Orders'),
                ('Transactors Change (%)', 'Transactors')
            ]
            
            # Prepare data for visualization - handle non-numeric values
            change_data = []
            for _, row in pre_post_df.iterrows():
                for col, label in change_metrics:
                    value = row[col]
                    try:
                        # Try to extract the numeric part from the string (e.g. "12.34%" -> 12.34)
                        if isinstance(value, str) and "%" in value:
                            value = float(value.replace('%', ''))
                        elif value == "N/A":
                            continue
                    except:
                        continue
                    
                    change_data.append({
                        'Cohort': row['Cohort'],
                        'Test Group': row['Test Group'],
                        'Metric': label,
                        'Percentage Change': value
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
        
        # Create a time series visualization
        st.write("## 📉 Trend Analysis")
        st.write("Daily metrics over time with test implementation date marked")
        
        # Choose metric for time series
        metric_options = ["gmv", "app_opens", "orders", "transactors"]
        selected_metric = st.selectbox("Select Metric for Time Series", metric_options, index=0)
        
        # Prepare data for time series visualization
        time_series_df = filtered_df.copy()
        
        # Calculate per user metrics if normalization is enabled
        if normalize_by_audience:
            time_series_df[selected_metric] = time_series_df[selected_metric] / time_series_df['audience_size'] * 1000
        
        # Create time series plot
        fig_time = px.line(
            time_series_df, 
            x="date", 
            y=selected_metric,
            color="data_set",
            title=f"{selected_metric.upper()}{' per 1000 Users' if normalize_by_audience else ''} Over Time",
            labels={selected_metric: f"{selected_metric.upper()}{' per 1000 Users' if normalize_by_audience else ''}"}
        )
        
        # Add vertical lines for test start dates
        for cohort, start_date in test_start_dates.items():
            if selected_cohort == "All Cohorts" or selected_cohort == cohort:
                fig_time.add_vline(
                    x=start_date, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"{cohort} Test Start", 
                    annotation_position="top right"
                )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        # Download button for CSV export
        pre_post_csv = pre_post_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Pre vs Post Analysis as CSV",
            data=pre_post_csv,
            file_name="pre_post_test_analysis.csv",
            mime="text/csv",
            key="download-csv"
        )
        
        # Summary insights
        st.write("## 💡 Key Insights from Pre vs Post Analysis")
        
        # Extract key insights
        insights = []
        
        for cohort, test_groups_metrics in pre_post_summary.items():
            for test_group, metrics in test_groups_metrics.items():
                # Check for significant changes (>10% change)
                significant_metrics = []
                
                if not np.isnan(metrics['gmv_pct_change']) and abs(metrics['gmv_pct_change']) > 10:
                    direction = "increase" if metrics['gmv_pct_change'] > 0 else "decrease"
                    significant_metrics.append(f"GMV showed a {abs(metrics['gmv_pct_change']):.1f}% {direction}")
                
                if not np.isnan(metrics['app_opens_pct_change']) and abs(metrics['app_opens_pct_change']) > 10:
                    direction = "increase" if metrics['app_opens_pct_change'] > 0 else "decrease"
                    significant_metrics.append(f"App Opens showed a {abs(metrics['app_opens_pct_change']):.1f}% {direction}")
                
                if not np.isnan(metrics['orders_pct_change']) and abs(metrics['orders_pct_change']) > 10:
                    direction = "increase" if metrics['orders_pct_change'] > 0 else "decrease"
                    significant_metrics.append(f"Orders showed a {abs(metrics['orders_pct_change']):.1f}% {direction}")
                
                if not np.isnan(metrics['transactors_pct_change']) and abs(metrics['transactors_pct_change']) > 10:
                    direction = "increase" if metrics['transactors_pct_change'] > 0 else "decrease"
                    significant_metrics.append(f"Transactors showed a {abs(metrics['transactors_pct_change']):.1f}% {direction}")
                
                if significant_metrics:
                    insights.append(f"**{cohort} - {test_group}**: {'; '.join(significant_metrics)}.")
        
        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.write("No significant changes (>10%) detected between pre and post periods.")
        
        # Executive summary card
        st.write("## 📋 Executive Summary")
        
        # Create metrics for the summary
        with st.container():
            cols = st.columns(2)
            
            # Calculate overall metrics
            overall_gmv_change = pre_post_df['GMV Change (%)'].replace('N/A', np.nan).apply(lambda x: float(x.replace('%', '')) if isinstance(x, str) else x).mean()
            overall_app_opens_change = pre_post_df['App Opens Change (%)'].replace('N/A', np.nan).apply(lambda x: float(x.replace('%', '')) if isinstance(x, str) else x).mean()
            
            # Find best performing test group
            best_gmv_idx = None
            max_gmv_change = -float('inf')
            
            for idx, row in pre_post_df.iterrows():
                val = row['GMV Change (%)']
                try:
                    if isinstance(val, str) and "%" in val:
                        val = float(val.replace('%', ''))
                    if val > max_gmv_change:
                        max_gmv_change = val
                        best_gmv_idx = idx
                except:
                    continue
            
            with cols[0]:
                if not np.isnan(overall_gmv_change):
                    st.metric("Overall GMV Change", f"{overall_gmv_change:.2f}%", 
                              delta=f"{overall_gmv_change:.1f}%" if overall_gmv_change != 0 else "0%")
                
                if not np.isnan(overall_app_opens_change):
                    st.metric("Overall App Opens Change", f"{overall_app_opens_change:.2f}%", 
                              delta=f"{overall_app_opens_change:.1f}%" if overall_app_opens_change != 0 else "0%")
            
            with cols[1]:
                # Best performing cohort/test group
                if best_gmv_idx is not None:
                    best_cohort = pre_post_df.loc[best_gmv_idx, 'Cohort']
                    best_test_group = pre_post_df.loc[best_gmv_idx, 'Test Group']
                    st.metric("Best Performing Test Group", f"{best_cohort} - {best_test_group}", 
                              delta=f"{max_gmv_change:.1f}% GMV")
                
                # Total number of test groups analyzed
                st.metric("Test Groups Analyzed", f"{len(pre_post_df)}")
            
            # Summary box
            if best_gmv_idx is not None:
                st.info(f"""
                **Executive Summary**: Analysis of {len(pre_post_df)} test groups across {len(pre_post_df['Cohort'].unique())} cohorts shows 
                an overall GMV change of **{overall_gmv_change:.2f}%** and app opens change of **{overall_app_opens_change:.2f}%**. 
                
                The best performing implementation was **{best_cohort} - {best_test_group}** with a **{max_gmv_change:.2f}%** improvement in GMV.
                """)
    else:
        st.warning("No data available for pre vs post analysis with the selected filters.")
else:
    # Show a demo if no file is uploaded
    st.info("👆 Please upload your experiment data CSV file to get started.")
    
    # Example file format
    st.write("### Expected Data Format")
    st.write("Your CSV file should contain the following columns:")
    
    example_data = {
        'date': ['2025-03-01', '2025-03-01', '2025-03-02', '2025-03-02'],
        'cohort': ['resp', 'resp', 'resp', 'resp'],
        'data_set': ['Test Group 1', 'Control Set', 'Test Group 1', 'Control Set'],
        'audience_size': [1000, 500, 1000, 500],
        'app_opens': [150, 70, 160, 75],
        'transactors': [25, 12, 30, 14],
        'orders': [35, 16, 40, 19],
        'gmv': [5000, 2300, 5500, 2500]
    }
    
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)
    
    st.write("""
    ### Instructions
    1. Upload your CSV file with the required format
    2. Set the test start dates for each cohort
    3. Use the filters to focus on specific cohorts or recency segments
    4. Analyze the pre vs post metrics to understand test performance
    5. Download the analysis as a CSV for reporting
    """)
