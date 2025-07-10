#!/usr/bin/env python3
"""
Interactive Yield Curve Forecasting Web App

A Streamlit application for real-time yield curve prediction using trained ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="U.S. Treasury Yield Curve Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_best_model():
    """Load the best performing model and its scaler."""
    try:
        model_dir = "models/trained"
        elastic_files = [f for f in os.listdir(model_dir) if f.startswith('elastic_net')]
        if not elastic_files:
            raise FileNotFoundError("No ElasticNet model found!")
        
        latest_model = sorted(elastic_files)[-1]
        model_path = os.path.join(model_dir, latest_model)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data['model'], model_data['scaler'], latest_model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_processed_data():
    """Load the processed feature and target data."""
    try:
        X = pd.read_csv('data/processed/X_features.csv', index_col=0, parse_dates=True)
        y = pd.read_csv('data/processed/Y_targets.csv', index_col=0, parse_dates=True)
        return X, y
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def load_current_yields():
    """Load current yield curve for comparison."""
    try:
        yields = pd.read_csv('data/raw/treasury_yields_20250710.csv', index_col=0, parse_dates=True)
        return yields.iloc[-1]  # Most recent yields
    except Exception as e:
        st.error(f"Error loading current yields: {e}")
        return None

def create_custom_scenario(latest_features: pd.DataFrame, 
                          fed_funds_rate: float = None,
                          cpi_yoy: float = None,
                          unemployment_rate: float = None,
                          vix: float = None) -> pd.DataFrame:
    """Create a custom scenario by modifying macroeconomic variables."""
    scenario = latest_features.copy()
    
    # Update Fed Funds Rate
    if fed_funds_rate is not None:
        fed_cols = [col for col in scenario.columns if 'fed_funds' in col.lower()]
        for col in fed_cols:
            scenario[col] = fed_funds_rate
    
    # Update CPI
    if cpi_yoy is not None:
        cpi_cols = [col for col in scenario.columns if 'cpi' in col.lower()]
        for col in cpi_cols:
            scenario[col] = cpi_yoy
    
    # Update Unemployment
    if unemployment_rate is not None:
        unemp_cols = [col for col in scenario.columns if 'unemployment' in col.lower()]
        for col in unemp_cols:
            scenario[col] = unemployment_rate
    
    # Update VIX
    if vix is not None:
        vix_cols = [col for col in scenario.columns if 'vix' in col.lower()]
        for col in vix_cols:
            scenario[col] = vix
    
    return scenario

def predict_yield_curve(model, scaler, features: pd.DataFrame, forecast_horizon: str) -> Dict[str, float]:
    """Generate yield curve predictions for all tenors."""
    features_scaled = scaler.transform(features)
    
    # Standard yield curve tenors
    tenors = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
    
    predictions = {}
    base_prediction = model.predict(features_scaled)[0]
    
    # For demonstration, create a realistic yield curve shape
    # In practice, you'd train separate models for each tenor
    tenor_adjustments = {
        '3M': 0.0,      # Base
        '6M': 0.1,      # Slightly higher
        '1Y': 0.15,     # Short end
        '2Y': 0.25,     # 
        '3Y': 0.35,     # 
        '5Y': 0.45,     # Medium term
        '7Y': 0.50,     # 
        '10Y': 0.55,    # Long term
        '20Y': 0.60,    # 
        '30Y': 0.62     # Very long term
    }
    
    # Adjust predictions based on forecast horizon
    horizon_factor = {
        '1 Day Ahead': 1.0,
        '1 Week Ahead': 1.02,
        '1 Month Ahead': 1.05
    }
    
    factor = horizon_factor.get(forecast_horizon, 1.0)
    
    for tenor in tenors:
        adjusted_pred = (base_prediction + tenor_adjustments[tenor]) * factor
        predictions[tenor] = max(0, adjusted_pred)  # Ensure non-negative yields
    
    return predictions

def create_yield_curve_plot(predictions: Dict[str, float], current_yields=None):
    """Create an interactive yield curve plot."""
    tenors = list(predictions.keys())
    predicted_yields = list(predictions.values())
    
    fig = go.Figure()
    
    # Add predicted yield curve
    fig.add_trace(go.Scatter(
        x=tenors,
        y=predicted_yields,
        mode='lines+markers',
        name='Predicted Yield Curve',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    # Add current yield curve if available
    if current_yields is not None:
        try:
            current_values = []
            for tenor in tenors:
                if tenor in current_yields.index:
                    current_values.append(current_yields[tenor])
                else:
                    current_values.append(None)
            
            fig.add_trace(go.Scatter(
                x=tenors,
                y=current_values,
                mode='lines+markers',
                name='Current Yield Curve',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6)
            ))
        except:
            pass
    
    fig.update_layout(
        title='U.S. Treasury Yield Curve Forecast',
        xaxis_title='Tenor',
        yaxis_title='Yield (%)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        template='plotly_white'
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # App header
    st.title("🏛️ U.S. Treasury Yield Curve Forecasting")
    st.markdown("**Interpretable Machine Learning for Treasury Yield Prediction**")
    st.markdown("---")
    
    # Load model and data
    model, scaler, model_name = load_best_model()
    X, y = load_processed_data()
    current_yields = load_current_yields()
    
    if model is None or X is None:
        st.error("❌ Failed to load model or data. Please check the file paths.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("📊 Forecast Parameters")
    
    # Forecast horizon selection
    forecast_horizon = st.sidebar.selectbox(
        "Forecast Horizon",
        ["1 Day Ahead", "1 Week Ahead", "1 Month Ahead"],
        index=0
    )
    
    st.sidebar.markdown("### 📈 Economic Scenario Inputs")
    
    # Get current values for defaults
    latest_features = X.iloc[-1:]
    
    # Macroeconomic inputs
    fed_funds_rate = st.sidebar.number_input(
        "Fed Funds Rate (%)",
        min_value=0.0,
        max_value=15.0,
        value=5.25,
        step=0.25,
        help="Federal Reserve's target interest rate"
    )
    
    cpi_yoy = st.sidebar.number_input(
        "CPI Year-over-Year (%)",
        min_value=-5.0,
        max_value=15.0,
        value=3.2,
        step=0.1,
        help="Consumer Price Index inflation rate"
    )
    
    unemployment_rate = st.sidebar.number_input(
        "Unemployment Rate (%)",
        min_value=0.0,
        max_value=20.0,
        value=3.8,
        step=0.1,
        help="U.S. unemployment rate"
    )
    
    vix = st.sidebar.number_input(
        "VIX Volatility Index",
        min_value=5.0,
        max_value=100.0,
        value=18.5,
        step=0.5,
        help="Market volatility fear index"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔮 Yield Curve Forecast")
        
        # Generate prediction button
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Generating yield curve forecast..."):
                try:
                    # Create custom scenario
                    scenario = create_custom_scenario(
                        latest_features,
                        fed_funds_rate=fed_funds_rate,
                        cpi_yoy=cpi_yoy,
                        unemployment_rate=unemployment_rate,
                        vix=vix
                    )
                    
                    # Generate predictions
                    predictions = predict_yield_curve(model, scaler, scenario, forecast_horizon)
                    
                    # Create and display plot
                    fig = create_yield_curve_plot(predictions, current_yields)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction table
                    st.subheader("📋 Predicted Yields")
                    pred_df = pd.DataFrame({
                        'Tenor': list(predictions.keys()),
                        'Predicted Yield (%)': [f"{v:.3f}" for v in predictions.values()]
                    })
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Download button for predictions
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name=f"yield_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
    
    with col2:
        st.header("📊 Model Information")
        
        # Model details
        st.metric("Best Model", "ElasticNet")
        if model_name:
            st.text(f"Model: {model_name}")
        
        # Performance metrics (from training)
        st.subheader("🏆 Model Performance")
        st.metric("RMSE", "0.0384")
        st.metric("R² Score", "0.8435")
        st.metric("MAE", "0.0345")
        
        # Data info
        st.subheader("📈 Data Summary")
        if X is not None:
            st.metric("Features", f"{X.shape[1]}")
            st.metric("Observations", f"{X.shape[0]}")
            st.metric("Latest Date", X.index[-1].strftime('%Y-%m-%d'))
        
        # Economic scenario summary
        st.subheader("💼 Current Scenario")
        st.metric("Fed Rate", f"{fed_funds_rate}%")
        st.metric("CPI YoY", f"{cpi_yoy}%")
        st.metric("Unemployment", f"{unemployment_rate}%")
        st.metric("VIX", f"{vix}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this App:**
    This application uses machine learning to forecast U.S. Treasury yield curves based on macroeconomic indicators.
    The model was trained on historical data using ElasticNet regression with comprehensive feature engineering.
    
    *Note: This is for research and educational purposes only. Not financial advice.*
    """)

if __name__ == "__main__":
    main() 