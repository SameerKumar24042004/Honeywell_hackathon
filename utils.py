import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import random

def load_process_data():
    """Load and cache the beer manufacturing process data"""
    try:
        # Try to load enhanced data if available
        df = pd.read_csv('enhanced_beer_data.csv')
        return df
    except FileNotFoundError:
        # Fallback to original data with simulated process parameters
        df = pd.read_csv('beers.csv')
        df = df.dropna(subset=['abv', 'style'])
        df = df[df['abv'] > 0]
        
        # Add simulated process parameters
        n_samples = len(df)
        np.random.seed(42)
        
        df['mash_temperature_c'] = np.random.normal(67, 3, n_samples)
        df['fermentation_temp_c'] = np.random.normal(18, 4, n_samples)
        df['fermentation_ph'] = np.random.normal(4.2, 0.4, n_samples)
        df['pump_pressure_bar'] = np.random.normal(2.5, 0.5, n_samples)
        df['process_anomaly'] = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
        df['quality_grade'] = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.7, 0.25, 0.05])
        
        # Add timestamps
        start_date = datetime.now() - timedelta(days=30)
        df['batch_timestamp'] = [start_date + timedelta(hours=i*6) for i in range(n_samples)]
        df['batch_id'] = [f"BATCH_{i:06d}" for i in range(1, n_samples+1)]
        
        return df

def load_models():
    """Load trained models"""
    try:
        return joblib.load('beer_manufacturing_models.pkl')
    except FileNotFoundError:
        return None

def simulate_real_time_data():
    """Simulate real-time process data"""
    return {
        'timestamp': datetime.now(),
        'batch_id': f"BATCH_{random.randint(100000, 999999)}",
        'beer_style': random.choice(['IPA', 'Lager', 'Stout', 'APA']),
        'mash_temp': np.random.normal(67, 2),
        'fermentation_temp': np.random.normal(18, 1.5),
        'fermentation_ph': np.random.normal(4.2, 0.3),
        'pump_pressure': np.random.normal(2.5, 0.3),
        'predicted_abv': np.random.normal(5.5, 1.2),
        'anomaly_probability': np.random.beta(2, 8)  # Skewed towards low probability
    }

def get_custom_css():
    """Return custom CSS styles"""
    return """
    <style>
    [data-testid="stToolbar"] {
        position: relative;
    }
    [data-testid="stToolbar"]::before {
        content: "HONEYWELL";
        position: absolute;
        top: 50%;
        left: 80px;
        transform: translateY(-50%);
        font-size: 24px;
        font-weight: 700;
        font-family: 'Arial', sans-serif;
        color: #FF4814;
        z-index: 1000;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1.2rem;
        border-radius: 4px;
        border-left: 4px solid #1565c0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .alert-critical {
        background-color: #fef2f2;
        padding: 1.2rem;
        border-radius: 4px;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
        color: #000000;
    }
    .alert-warning {
        background-color: #fff7ed;
        padding: 1.2rem;
        border-radius: 4px;
        border-left: 4px solid #ea580c;
        margin: 1rem 0;
        color: #000000;
    }
    .status-normal {
        background-color: #f0fdf4;
        padding: 1.2rem;
        border-radius: 4px;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
        color: #000000;
    }
    .small-font {
        font-size: 1rem;
    }
    </style>
    """

def filter_dataframe(df, selected_style, time_range=None):
    """Filter dataframe based on selections"""
    filtered_df = df.copy()
    if selected_style != 'All':
        filtered_df = filtered_df[filtered_df['style'] == selected_style]
    
    # Add time range filtering logic here if needed
    # This can be expanded based on your requirements
    
    return filtered_df

def get_process_status_html(anomaly_probability):
    """Return HTML for process status based on anomaly probability"""
    if anomaly_probability > 0.7:
        return """
        <div class="alert-critical">
        🔴 <strong>CRITICAL ALERT:</strong> High risk of quality deviation detected. 
        Immediate process intervention required.
        </div>
        """
    elif anomaly_probability > 0.5:
        return """
        <div class="alert-warning">
        🟡 <strong>WARNING:</strong> Process parameters approaching critical thresholds. 
        Review recommended.
        </div>
        """
    else:
        return """
        <div class="status-normal">
        🟢 <strong>Status:</strong> All parameters within specification.
        </div>
        """

def get_status_indicator(value, min_val, max_val):
    """Get status indicator for process parameters"""
    return "🟢 Normal" if min_val <= value <= max_val else "🔴 Alert"

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "🟢 Low"
    elif probability < 0.7:
        return "🟡 Medium"
    else:
        return "🔴 High"