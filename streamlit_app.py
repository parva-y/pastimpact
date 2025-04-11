import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide")

st.title("Pre vs Post Test Analysis")
st.write("### Test Performance Comparison")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("1MG_Test_and_control_report_transformed (2).csv", parse_dates=['date'])
    return df

df = load_data()

# Ensure necessary columns exist
required_columns = {'date', 'data_set', 'audience_size', 'app_opens', 'transactors', 'orders', 'gmv', 'cohort'}
has_recency_data = 'Recency' in df.columns

if not required_columns.issubset(df.columns):
    st.write("Missing required columns in the CSV file.")
    st.stop()

# Define control group and test groups
control_group = "Control Set"
test_groups = [g for g in df['data_set'].unique() if g != control_group]

# Test start dates
test_start_dates = {
    "resp": pd.Timestamp("2025-03-05"),
    "cardiac": pd.Timestamp("2025-03-18"),
    "diabetic": pd.Timestamp("2025-03-06"),
    "derma": pd.Timestamp("2025-03-18")
}

# Cohort selection with default "All Cohorts" option
cohort_options = ["All Cohorts"] + list(df['cohort'].unique())
selected_cohort = st.sidebar.selectbox("Select Cohort", cohort_options)

# Recency selector with default "All Recency" option
if has_recency_data:
    recency_values = sorted(df['Recency'].unique())
    recency_options = ["All Recency"] + list(recency_values)
    selected_recency = st.sidebar.selectbox("Select Recency", recency_options)
else:
    selected_recency = "All Recency"

# Add pre-post period selection
period_options = [7, 14, 30]
selected_period = st.sidebar.selectbox("Select Pre/Post Period (Days)", period_options)

# Function to filter data based on selections
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
    
    return filtered_df, start_date

filtered_df, start_date = filter_data(df, selected_cohort, selected_recency)

# Calculate pre and post period metrics
def calculate_pre_post_metrics(df, start_date, period_days):
    # Define pre and post periods
    pre_start = start_date - timedelta(days=period_days)
    pre_end = start_date - timedelta(days=1)
    post_start = start_date
    post_end = start_date + timedelta(days=period_days-1)
    
    # Filter by periods
    pre_df = df[(df['date'] >= pre_start) & (df['date'] <= pre_end)]
    post_df = df[(df['date'] >= post_start) & (df['date'] <= post_end)]
    
    # Group by data_set
    pre_metrics = pre_df.groupby('data_set').agg({
        'audience_size': 'sum',
        'app_opens': 'sum',
        'transactors': 'sum',
        'orders': 'sum',
        'gmv': 'sum'
    }).reset_index()
    
    post_metrics = post_df.groupby('data_set').agg({
        'audience_size': 'sum', 
        'app_opens': 'sum',
        'transactors': 'sum',
        'orders': 'sum',
        'gmv': 'sum'
    }).reset_index()
    
    # Calculate per user metrics
    for df in [pre_metrics, post_metrics]:
        df['app_opens_per_user'] = df['app_opens'] / df['audience_size']
        df['transactors_per_user'] = df['transactors'] / df['audience_size']
        df['orders_per_user'] = df['orders'] / df['audience_size']
        df['gmv_per_user'] = df['gmv'] / df['audience_size']
    
    return pre_metrics, post_metrics, pre_start, pre_end, post_start, post_end

# Function to calculate pre-post comparison
def calculate_comparison(pre_metrics, post_metrics):
    comparison = []
    
    for data_set in pre_metrics['data_set'].unique():
        pre_row = pre_metrics[pre_metrics['data_set'] == data_set].iloc[0]
        post_row = post_metrics[post_metrics['data_set'] == data_set].iloc[0]
        
        # Calculate percentage changes
        app_opens_change = ((post_row['app_opens_per_user'] - pre_row['app_opens_per_user']) / pre_row['app_opens_per_user']) * 100 if pre_row['app_opens_per_user'] > 0 else float('inf')
        transactors_change = ((post_row['transactors_per_user'] - pre_row['transactors_per_user']) / pre_row['transactors_per_user']) * 100 if pre_row['transactors_per_user'] > 0 else float('inf')
        orders_change = ((post_row['orders_per_user'] - pre_row['orders_per_user']) / pre_row['orders_per_user']) * 100 if pre_row['orders_per_user'] > 0 else float('inf')
        gmv_change = ((post_row['gmv_per_user'] - pre_row['gmv_per_user']) / pre_row['gmv_per_user']) * 100 if pre_row['gmv_per_user'] > 0 else float('inf')
        
        comparison.append({
            'Data Set': data_set,
            'Pre App Opens': pre_row['app_opens'],
            'Post App Opens': post_row['app_opens'],
            'App Opens % Change': app_opens_change,
            
            'Pre Transactors': pre_row['transactors'],
            'Post Transactors': post_row['transactors'], 
            'Transactors % Change': transactors_change,
            
            'Pre Orders': pre_row['orders'],
            'Post Orders': post_row['orders'],
            'Orders % Change': orders_change,
            
            'Pre GMV': pre_row['gmv'],
            'Post GMV': post_row['gmv'],
            'GMV % Change': gmv_change,
            
            'Pre App Opens per User': pre_row['app_opens_per_user'],
            'Post App Opens per User': post_row['app_opens_per_user'],
            
            'Pre Transactors per User': pre_row['transactors_per_user'],
            'Post Transactors per User': post_row['transactors_per_user'],
            
            'Pre Orders per User': pre_row['orders_per_user'],
            'Post Orders per User': post_row['orders_per_user'],
            
            'Pre GMV per User': pre_row['gmv_per_user'],
            'Post GMV per User': post_row['gmv_per_user'],
            
            'Audience Size': pre_row['audience_size']
        })
    
    return pd.DataFrame(comparison)

# Function to calculate lift comparison between test and control groups
def calculate_lift_vs_control(comparison_df):
    if 'Control Set' not in comparison_df['Data Set'].values:
        return pd.DataFrame()
    
    control_row = comparison_df[comparison_df['Data Set'] == 'Control Set'].iloc[0]
    lift_rows = []
    
    for index, test_row in comparison_df[comparison_df['Data Set'] != 'Control Set'].iterrows():
        # Calculate lifts against control
        app_opens_lift = test_row['App Opens % Change'] - control_row['App Opens % Change']
        transactors_lift = test_row['Transactors % Change'] - control_row['Transactors % Change']
        orders_lift = test_row['Orders % Change'] - control_row['Orders % Change']
        gmv_lift = test_row['GMV % Change'] - control_row['GMV % Change']
        
        # Post-period per-user lift compared to control
        post_app_opens_vs_control = ((test_row['Post App Opens per User'] - control_row['Post App Opens per User']) / 
                                     control_row['Post App Opens per User']) * 100 if control_row['Post App Opens per User'] > 0 else float('inf')
        
        post_transactors_vs_control = ((test_row['Post Transactors per User'] - control_row['Post Transactors per User']) / 
                                      control_row['Post Transactors per User']) * 100 if control_row['Post Transactors per User'] > 0 else float('inf')
        
        post_orders_vs_control = ((test_row['Post Orders per User'] - control_row['Post Orders per User']) / 
                                 control_row['Post Orders per User']) * 100 if control_row['Post Orders per User'] > 0 else float('inf')
        
        post_gmv_vs_control = ((test_row['Post GMV per User'] - control_row['Post GMV per User']) / 
                              control_row['Post GMV per User']) * 100 if control_row['Post GMV per User'] > 0 else float('inf')
        
        lift_rows.append({
            'Test Group': test_row['Data Set'],
            'App Opens Lift vs. Control': app_opens_lift,
            'Transactors Lift vs. Control': transactors_lift,
            'Orders Lift vs. Control': orders_lift,
            'GMV Lift vs. Control': gmv_lift,
            'Post App Opens vs. Control (%)': post_app_opens_vs_control,
            'Post Transactors vs. Control (%)': post_transactors_vs_control,
            'Post Orders vs. Control (%)': post_orders_vs_control,
            'Post GMV vs. Control (%)': post_gmv_vs_control
        })
    
    return pd.DataFrame(lift_rows)

# Calculate metrics and comparison
# Run calculations for specific cohort or all cohorts
if selected_cohort == "All Cohorts":
    st.write("## Pre vs Post Analysis for All Cohorts")
    
    all_cohorts_comparison = []
    all_cohorts_lift = []
    
    for cohort in df['cohort'].unique():
        st.write(f"### Cohort: {cohort}")
        
        cohort_df = df[df['cohort'] == cohort]
        start_date = test_start_dates.get(cohort, cohort_df['date'].min())
        
        # Apply recency filter if needed
        if selected_recency != "All Recency" and has_recency_data:
            cohort_df = cohort_df[cohort_df['Recency'] == selected_recency]
        
        pre_metrics, post_metrics, pre_start, pre_end, post_start, post_end = calculate_pre_post_metrics(
            cohort_df, start_date, selected_period
        )
        
        st.write(f"Pre-period: {pre_start.date()} to {pre_end.date()}")
        st.write(f"Post-period: {post_start.date()} to {post_end.date()}")
        
        comparison_df = calculate_comparison(pre_metrics, post_metrics)
        
        if not comparison_df.empty:
            # Add cohort column
            comparison_df['Cohort'] = cohort
            all_cohorts_comparison.append(comparison_df)
            
            # Show the comparison table
            st.write("#### Metrics Comparison")
            comparison_display = comparison_df[[
                'Data Set', 'Pre App Opens', 'Post App Opens', 'App Opens % Change',
                'Pre GMV', 'Post GMV', 'GMV % Change',
                'Pre Orders', 'Post Orders', 'Orders % Change',
                'Pre Transactors', 'Post Transactors', 'Transactors % Change'
            ]].copy()
            
            # Format the percentages
            for col in comparison_display.columns:
                if '% Change' in col:
                    comparison_display[col] = comparison_display[col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else 'N/A')
            
            st.dataframe(comparison_display)
            
            # Calculate lift vs control
            lift_df = calculate_lift_vs_control(comparison_df)
            if not lift_df.empty:
                lift_df['Cohort'] = cohort
                all_cohorts_lift.append(lift_df)
                
                # Show the lift table
                st.write("#### Lift vs Control")
                lift_display = lift_df[[
                    'Test Group', 'App Opens Lift vs. Control', 'GMV Lift vs. Control',
                    'Orders Lift vs. Control', 'Transactors Lift vs. Control'
                ]].copy()
                
                # Format the lifts
                for col in lift_display.columns:
                    if 'Lift' in col:
                        lift_display[col] = lift_display[col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else 'N/A')
                
                st.dataframe(lift_display)
            else:
                st.write("No lift data available - control group missing.")
        else:
            st.write("No comparison data available for this cohort.")
    
    # Combine all cohorts data
    if all_cohorts_comparison:
        combined_comparison = pd.concat(all_cohorts_comparison)
        st.write("## Summary Across All Cohorts")
        
        # Group by cohort and data_set
        summary_comparison = combined_comparison.groupby(['Cohort', 'Data Set']).agg({
            'App Opens % Change': 'mean',
            'GMV % Change': 'mean',
            'Orders % Change': 'mean',
            'Transactors % Change': 'mean'
        }).reset_index()
        
        # Format the summary table
        for col in summary_comparison.columns:
            if '% Change' in col:
                summary_comparison[col] = summary_comparison[col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else 'N/A')
        
        st.write("### Average % Change by Cohort and Data Set")
        st.dataframe(summary_comparison)
    
    # Combined lift table
    if all_cohorts_lift:
        combined_lift = pd.concat(all_cohorts_lift)
        
        # Group by cohort and test group
        summary_lift = combined_lift.groupby(['Cohort', 'Test Group']).agg({
            'App Opens Lift vs. Control': 'mean',
            'GMV Lift vs. Control': 'mean',
            'Orders Lift vs. Control': 'mean',
            'Transactors Lift vs. Control': 'mean'
        }).reset_index()
        
        # Format the summary lift table
        for col in summary_lift.columns:
            if 'Lift' in col:
                summary_lift[col] = summary_lift[col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else 'N/A')
        
        st.write("### Average Lift vs Control by Cohort and Test Group")
        st.dataframe(summary_lift)
        
else:
    # For a single selected cohort
    pre_metrics, post_metrics, pre_start, pre_end, post_start, post_end = calculate_pre_post_metrics(
        filtered_df, start_date, selected_period
    )
    
    st.write(f"## Pre vs Post Analysis for {selected_cohort}")
    if selected_recency != "All Recency":
        st.write(f"### Recency: {selected_recency}")
    
    st.write(f"Pre-period: {pre_start.date()} to {pre_end.date()}")
    st.write(f"Post-period: {post_start.date()} to {post_end.date()}")
    
    # Calculate comparison metrics
    comparison_df = calculate_comparison(pre_metrics, post_metrics)
    
    if not comparison_df.empty:
        # Show the comparison table - raw metrics
        st.write("### 📊 Raw Metrics Comparison")
        raw_metrics = comparison_df[[
            'Data Set', 'Pre App Opens', 'Post App Opens', 
            'Pre GMV', 'Post GMV',
            'Pre Orders', 'Post Orders',
            'Pre Transactors', 'Post Transactors',
            'Audience Size'
        ]].copy()
        
        st.dataframe(raw_metrics)
        
        # Per user metrics
        st.write("### 👤 Per User Metrics")
        per_user_metrics = comparison_df[[
            'Data Set', 
            'Pre App Opens per User', 'Post App Opens per User',
            'Pre GMV per User', 'Post GMV per User',
            'Pre Orders per User', 'Post Orders per User',
            'Pre Transactors per User', 'Post Transactors per User'
        ]].copy()
        
        # Format the per user metrics to 4 decimal places
        for col in per_user_metrics.columns:
            if 'per User' in col:
                per_user_metrics[col] = per_user_metrics[col].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(per_user_metrics)
        
        # Percentage change
        st.write("### 📈 Percentage Change")
        pct_change = comparison_df[[
            'Data Set', 'App Opens % Change', 'GMV % Change', 
            'Orders % Change', 'Transactors % Change'
        ]].copy()
        
        # Format the percentages
        for col in pct_change.columns:
            if '% Change' in col:
                pct_change[col] = pct_change[col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else 'N/A')
        
        st.dataframe(pct_change)
        
        # Calculate lift vs control
        lift_df = calculate_lift_vs_control(comparison_df)
        if not lift_df.empty:
            st.write("### 🚀 Lift vs Control")
            
            # Format lifts for display
            lift_display = lift_df.copy()
            for col in lift_display.columns:
                if 'Lift' in col or '%' in col:
                    lift_display[col] = lift_display[col].apply(lambda x: f"{x:.2f}%" if x != float('inf') else 'N/A')
            
            st.dataframe(lift_display)
            
            # Add minimal bar chart
            st.write("### Key Lifts Summary")
            fig = px.bar(
                lift_df, 
                x='Test Group', 
                y=['GMV Lift vs. Control', 'Orders Lift vs. Control', 'App Opens Lift vs. Control', 'Transactors Lift vs. Control'],
                barmode='group',
                title="Key Metrics Lift vs Control"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No lift data available - control group missing.")
    else:
        st.write("No comparison data available for selected filters.")

# Download button for CSV export
if 'comparison_df' in locals() and not comparison_df.empty:
    csv = comparison_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Pre-Post Analysis as CSV",
        data=csv,
        file_name=f"pre_post_analysis_{selected_cohort}_{selected_period}days.csv",
        mime="text/csv",
    )
    
    if 'lift_df' in locals() and not lift_df.empty:
        lift_csv = lift_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Lift Analysis as CSV",
            data=lift_csv,
            file_name=f"lift_analysis_{selected_cohort}_{selected_period}days.csv",
            mime="text/csv",
        )

# Show brief explanation
st.write("## 📝 How to Interpret Results")
st.markdown("""
This analysis compares metrics before and after the test start date for both control and test groups:

1. **Raw Metrics**: Total values for app opens, GMV, orders, and transactors
2. **Per User Metrics**: Normalized values accounting for audience size differences  
3. **Percentage Change**: How metrics changed from pre to post period within each group
4. **Lift vs Control**: The difference in percentage change between test groups and control group

A positive lift indicates that the test group outperformed the control group in terms of improvement from pre to post periods.
""")
