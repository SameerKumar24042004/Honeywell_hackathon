import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from utils import (
    load_process_data, 
    load_models, 
    simulate_real_time_data, 
    get_custom_css,
    filter_dataframe,
    get_process_status_html,
    get_status_indicator,
    get_risk_level
)

# Page configuration
st.set_page_config(
    page_title="Beer Manufacturing Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

@st.cache_data
def cached_load_process_data():
    """Cached wrapper for load_process_data"""
    return load_process_data()

@st.cache_resource
def cached_load_models():
    """Cached wrapper for load_models"""
    models = load_models()
    if models is None:
        st.warning("Models not found. Please run the training script first.")
    return models

def render_real_time_monitor():
    """Render the real-time monitoring section"""
    st.header("Real-time Process Monitor")
    
    # Update metrics display
    col1, col2, col3, col4 = st.columns(4)
    current_data = simulate_real_time_data()
    
    with col1:
        st.metric(
            label="Active Batch",
            value=current_data['batch_id'],
            delta=current_data['beer_style']
        )
    
    with col2:
        temp_status = get_status_indicator(current_data['mash_temp'], 15, 70)
        st.metric(
            label="Mash Temperature",
            value=f"{current_data['mash_temp']:.1f}°C",
            delta=temp_status
        )
    
    with col3:
        ph_status = get_status_indicator(current_data['fermentation_ph'], 3.8, 4.8)
        st.metric(
            label="Fermentation pH",
            value=f"{current_data['fermentation_ph']:.2f}",
            delta=ph_status
        )
    
    with col4:
        risk_level = get_risk_level(current_data['anomaly_probability'])
        st.metric(
            label="Process Risk",
            value=f"{current_data['anomaly_probability']:.2%}",
            delta=risk_level
        )
    
    # Display process status alert
    status_html = get_process_status_html(current_data['anomaly_probability'])
    st.markdown(status_html, unsafe_allow_html=True)

def render_process_trends_tab(filtered_df):
    """Render the process trends tab"""
    st.subheader("Process Parameter Trends")
    
    # Time series plots with improved spacing
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mash Temperature', 'Fermentation pH', 'Pump Pressure', 'ABV Quality'),
        vertical_spacing=0.35,
        horizontal_spacing=0.2,
        row_heights=[0.5, 0.5]
    )
    
    # Sample recent data for trends
    recent_data = filtered_df.tail(50).copy()
    
    fig.add_trace(
        go.Scatter(x=recent_data.index, y=recent_data['mash_temperature_c'], 
                  name='Mash Temp', line=dict(color='#1565C0')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=recent_data.index, y=recent_data['fermentation_ph'], 
                  name='pH', line=dict(color='#2E7D32')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=recent_data.index, y=recent_data['pump_pressure_bar'], 
                  name='Pressure', line=dict(color='#ED6C02')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=recent_data.index, y=recent_data['abv'], 
                  name='ABV', line=dict(color='#D32F2F')),
        row=2, col=2
    )
    
    # Update layout with better spacing and formatting
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=100, l=80, r=80, b=80),
        title=dict(
            text="Process Parameters Over Time",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        font=dict(size=12)
    )
    
    # Update axes with increased spacing
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(row=i, col=j, title_standoff=25)
            fig.update_yaxes(row=i, col=j, title_standoff=25)

    st.plotly_chart(fig, use_container_width=True)

def render_quality_analysis_tab(filtered_df, selected_style):
    """Render the quality analysis tab"""
    st.subheader("Quality Analysis Dashboard")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Quality grade distribution pie chart
        quality_counts = filtered_df['quality_grade'].value_counts()
        fig_quality = px.pie(
            values=quality_counts.values, 
            names=quality_counts.index,
            title="Quality Grade Distribution",
            color_discrete_map={'A': '#4CAF50', 'B': '#FF9800', 'C': '#F44336'}
        )
        st.plotly_chart(fig_quality, use_container_width=True)
    
    with col2:
        # Quality Grade Bar Chart
        fig_quality_bar = px.bar(
            x=quality_counts.index,
            y=quality_counts.values,
            title="Quality Grade Counts",
            color=quality_counts.index,
            color_discrete_map={'A': '#4CAF50', 'B': '#FF9800', 'C': '#F44336'},
            labels={'x': 'Grade', 'y': 'Count'}
        )
        fig_quality_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_quality_bar, use_container_width=True)
    
    with col3:
        # ABV distribution by style
        if selected_style == 'All':
            top_styles = filtered_df['style'].value_counts().head(10).index
            style_data = filtered_df[filtered_df['style'].isin(top_styles)]
            fig_abv = px.box(
                style_data, 
                x='style', 
                y='abv',
                title="ABV Distribution by Style"
            )
            fig_abv.update_xaxes(tickangle=45)
            st.plotly_chart(fig_abv, use_container_width=True)
        else:
            fig_abv = px.histogram(
                filtered_df, 
                x='abv', 
                title=f"ABV Distribution - {selected_style}",
                nbins=20
            )
            st.plotly_chart(fig_abv, use_container_width=True)
    
    # Quality metrics summary
    st.subheader("Quality Metrics Summary")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        avg_abv = filtered_df['abv'].mean()
        st.metric("Average ABV", f"{avg_abv:.2f}%")
    
    with metrics_col2:
        quality_a_pct = (filtered_df['quality_grade'] == 'A').mean() * 100
        st.metric("Grade A Quality", f"{quality_a_pct:.1f}%")
    
    with metrics_col3:
        anomaly_rate = filtered_df['process_anomaly'].mean() * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    with metrics_col4:
        total_batches = len(filtered_df)
        st.metric("Total Batches", f"{total_batches:,}")

def render_anomaly_detection_tab(filtered_df, models):
    """Render the anomaly detection tab"""
    st.subheader("Anomaly Detection Analysis")
    
    if models:
        st.success("✅ ML Models loaded successfully")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly distribution over time
            fig_anomaly = px.scatter(
                filtered_df.tail(200), 
                x='mash_temperature_c', 
                y='fermentation_ph',
                color='process_anomaly',
                color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                title="Anomalies in Process Parameters",
                labels={'process_anomaly': 'Anomaly Status'}
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
        
        with col2:
            # Feature importance (simulated)
            features = ['Mash Temperature', 'Fermentation pH', 'Pump Pressure', 'Fermentation Temp']
            importance = [0.35, 0.28, 0.22, 0.15]
            
            fig_importance = px.bar(
                x=importance, 
                y=features,
                orientation='h',
                title="Feature Importance for Anomaly Detection"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Anomaly alerts
        st.subheader("Recent Anomaly Alerts")
        
        anomaly_data = filtered_df[filtered_df['process_anomaly'] == 1].tail(10)
        if not anomaly_data.empty:
            for idx, row in anomaly_data.iterrows():
                with st.expander(f"⚠️ Anomaly Alert - Batch {row.get('batch_id', idx)}"):
                    alert_col1, alert_col2, alert_col3 = st.columns(3)
                    
                    with alert_col1:
                        st.write(f"**Style:** {row['style']}")
                        st.write(f"**ABV:** {row['abv']:.2f}%")
                    
                    with alert_col2:
                        st.write(f"**Mash Temp:** {row['mash_temperature_c']:.1f}°C")
                        st.write(f"**pH:** {row['fermentation_ph']:.2f}")
                    
                    with alert_col3:
                        st.write(f"**Pressure:** {row['pump_pressure_bar']:.1f} bar")
                        st.write(f"**Quality:** Grade {row['quality_grade']}")
        else:
            st.info("No recent anomalies detected.")
    else:
        st.error("❌ ML Models not available. Please train models first.")

def render_production_history_tab(filtered_df):
    """Render the production history tab"""
    st.subheader("Batch Production History")
    
    # Search and filter options
    search_col1, search_col2 = st.columns(2)
    
    with search_col1:
        batch_search = st.text_input("Search Batch ID", placeholder="Enter batch ID...")
    
    with search_col2:
        quality_filter = st.selectbox("Filter by Quality", ['All', 'A', 'B', 'C'])
    
    # Filter batch data
    batch_data = filtered_df.copy()
    
    if batch_search:
        if 'batch_id' in batch_data.columns:
            batch_data = batch_data[batch_data['batch_id'].str.contains(batch_search, na=False)]
    
    if quality_filter != 'All':
        batch_data = batch_data[batch_data['quality_grade'] == quality_filter]
    
    # Display batch history table
    if not batch_data.empty:
        display_columns = ['style', 'abv', 'mash_temperature_c', 'fermentation_ph', 
                         'pump_pressure_bar', 'quality_grade', 'process_anomaly']
        
        available_columns = [col for col in display_columns if col in batch_data.columns]
        
        st.dataframe(
            batch_data[available_columns].head(50),
            use_container_width=True,
            height=400
        )
        
        # Export functionality
        csv_data = batch_data[available_columns].to_csv(index=False)
        st.download_button(
            label="📥 Download Batch Data (CSV)",
            data=csv_data,
            file_name=f"batch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No batches found matching the current filters.")
    
    # Batch statistics
    st.subheader("Production Statistics")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    
    with stat_col1:
        daily_production = len(batch_data) / 30  # Assuming 30-day period
        st.metric("Avg Daily Production", f"{daily_production:.1f} batches")
    
    with stat_col2:
        if not batch_data.empty:
            yield_rate = (batch_data['quality_grade'] == 'A').mean() * 100
            st.metric("Grade A Yield Rate", f"{yield_rate:.1f}%")
    
    with stat_col3:
        if not batch_data.empty:
            efficiency = (1 - batch_data['process_anomaly'].mean()) * 100
            st.metric("Process Efficiency", f"{efficiency:.1f}%")

def main():
    # Title and header
    st.title("Beer Manufacturing Process Monitor")
    st.markdown("### Process Control & Quality Management System")
    
    # Sidebar
    st.sidebar.header("Process Controls")
    
    # Load data
    df = cached_load_process_data()
    models = cached_load_models()
    
    # Sidebar filters
    beer_styles = ['All'] + sorted(df['style'].unique().tolist())
    selected_style = st.sidebar.selectbox("Beer Style", beer_styles)
    
    time_range = st.sidebar.selectbox(
        "Time Range", 
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"]
    )
    
    # Filter data based on selections
    filtered_df = filter_dataframe(df, selected_style, time_range)
    
    # Real-time monitoring section
    render_real_time_monitor()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Process Trends", 
        "Quality Analysis", 
        "Anomaly Detection", 
        "Production History"
    ])
    
    with tab1:
        render_process_trends_tab(filtered_df)
    
    with tab2:
        render_quality_analysis_tab(filtered_df, selected_style)
    
    with tab3:
        render_anomaly_detection_tab(filtered_df, models)
    
    with tab4:
        render_production_history_tab(filtered_df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <div class='small-font' style='text-align: center; color: white;'>
        Beer Manufacturing Process Monitor | 
        Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
        System Status: Active
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()