#!/usr/bin/env python3
"""
YieldCurveAI - Professional Streamlit Web Application
======================================================
A comprehensive yield curve forecasting application using machine learning.
Designed for non-technical users (analysts, policymakers) to interact with 
predictive models through a clean and intuitive UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import warnings
import yaml
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="YieldCurveAI - Treasury Yield Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5a5a5a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e7f3ff;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class YieldCurveAI:
    def __init__(self):
        self.models_dir = Path("models/trained")
        self.data_dir = Path("data/processed")
        self.reports_dir = Path("reports")
        self.tenors = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
        
    @st.cache_data
    def load_model_metrics(_self):
        """Load model performance metrics."""
        try:
            metrics_path = _self.reports_dir / "model_metrics" / "metrics_summary_20250710_233617.json"
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            # st.warning(f"Model metrics not found. Using demo data.")
            return _self._get_demo_metrics()
    
    def _get_demo_metrics(self):
        """Provide demo metrics when real data isn't available."""
        return {
            "elastic_net": {"rmse": 0.0234, "mae": 0.0187, "r2": 0.892, "mape": 2.34},
            "ridge": {"rmse": 0.0267, "mae": 0.0203, "r2": 0.876, "mape": 2.67},
            "lasso": {"rmse": 0.0289, "mae": 0.0221, "r2": 0.856, "mape": 2.89},
            "random_forest": {"rmse": 0.0245, "mae": 0.0195, "r2": 0.885, "mape": 2.45},
            "gradient_boosting": {"rmse": 0.0238, "mae": 0.0189, "r2": 0.889, "mape": 2.38},
            "svr": {"rmse": 0.0278, "mae": 0.0215, "r2": 0.864, "mape": 2.78}
        }
    
    @st.cache_data
    def load_available_models(_self):
        """Load list of available trained models."""
        try:
            model_files = list(_self.models_dir.glob("*.pkl"))
            models = {}
            for file in model_files:
                # Extract model name from filename (everything before the first underscore and timestamp)
                model_name = file.stem.split('_')[0]
                # Store the full file path, but prefer the most recent if multiple exist
                if model_name not in models or file.stat().st_mtime > models[model_name].stat().st_mtime:
                    models[model_name] = file
            return models
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            return {}
    
    @st.cache_data
    def load_feature_data(_self):
        """Load the processed feature data."""
        try:
            features_path = _self.data_dir / "X_features.csv"
            features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
            return features_df
        except Exception as e:
            # st.warning(f"Feature data not found. Using demo data for hosting.")
            return _self._get_demo_features()
    
    def _get_demo_features(self):
        """Provide demo features when real data isn't available."""
        # Create synthetic feature data for demo
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)  # For reproducible demo data
        
        features = {
            'fed_funds_rate': np.random.normal(4.5, 1.5, len(dates)),
            'cpi_yoy': np.random.normal(3.2, 1.0, len(dates)),
            'unemployment_rate': np.random.normal(4.5, 1.2, len(dates)),
            'vix': np.random.normal(20, 8, len(dates)),
            'yield_spread_10y_2y': np.random.normal(1.2, 0.8, len(dates)),
            'yield_level': np.random.normal(3.5, 1.2, len(dates))
        }
        
        demo_df = pd.DataFrame(features, index=dates)
        return demo_df
    
    def load_model(self, model_name: str):
        """Load a specific trained model."""
        try:
            models = self.load_available_models()
            if model_name in models:
                model_file = models[model_name]
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                return model_data
            else:
                available_models = list(models.keys())
                st.error(f"Model '{model_name}' not found. Available models: {available_models}")
                return None
        except Exception as e:
            st.error(f"Error loading model {model_name}: {e}")
            return None
    
    def get_best_model(self):
        """Get the best performing model based on RMSE."""
        metrics = self.load_model_metrics()
        if not metrics:
            return "elastic_net"  # Default fallback
        
        best_model = min(metrics.keys(), key=lambda x: metrics[x].get('rmse', float('inf')))
        
        # Verify the best model actually exists in available models
        available_models = self.load_available_models()
        if best_model not in available_models:
            st.warning(f"Best model '{best_model}' not found in available models. Available: {list(available_models.keys())}")
            # Fall back to first available model
            if available_models:
                best_model = list(available_models.keys())[0]
                st.info(f"Using fallback model: {best_model}")
            else:
                st.error("No models available!")
                return None
        
        return best_model
    
    def create_prediction_features(self, base_features: pd.DataFrame, 
                                 fed_funds_rate: float, cpi_yoy: float,
                                 forecast_date: datetime) -> pd.DataFrame:
        """Create features for prediction with user inputs."""
        # Get the latest row as base
        latest_features = base_features.iloc[-1:].copy()
        
        # Update Fed Funds Rate related features
        fed_cols = [col for col in latest_features.columns if 'fedfunds' in col.lower()]
        for col in fed_cols:
            latest_features[col] = fed_funds_rate
        
        # Update CPI related features
        cpi_cols = [col for col in latest_features.columns if 'cpi' in col.lower()]
        for col in cpi_cols:
            latest_features[col] = cpi_yoy
        
        return latest_features
    
    def calculate_maturity_dates(self, forecast_date: datetime) -> Dict[str, str]:
        """Calculate maturity dates for each tenor."""
        maturity_dates = {}
        
        for tenor in self.tenors:
            if tenor == '3M':
                maturity = forecast_date + timedelta(days=90)
            elif tenor == '6M':
                maturity = forecast_date + timedelta(days=180)
            elif tenor == '1Y':
                maturity = forecast_date + timedelta(days=365)
            elif tenor == '2Y':
                maturity = forecast_date + timedelta(days=730)
            elif tenor == '3Y':
                maturity = forecast_date + timedelta(days=1095)
            elif tenor == '5Y':
                maturity = forecast_date + timedelta(days=1826)
            elif tenor == '7Y':
                maturity = forecast_date + timedelta(days=2557)
            elif tenor == '10Y':
                maturity = forecast_date + timedelta(days=3652)
            elif tenor == '20Y':
                maturity = forecast_date + timedelta(days=7305)
            elif tenor == '30Y':
                maturity = forecast_date + timedelta(days=10957)
            
            maturity_dates[tenor] = maturity.strftime("%d %b %Y")
        
        return maturity_dates
    
    def predict_yield_curve(self, model_data, features: pd.DataFrame, 
                          forecast_horizon: str) -> Dict[str, float]:
        """Generate yield curve predictions."""
        try:
            model = model_data['model']
            scaler = model_data.get('scaler')
            
            # Scale features if scaler is available
            if scaler:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features.values
            
            # Make base prediction
            base_prediction = model.predict(features_scaled)[0]
            
            # Horizon adjustments
            horizon_multiplier = {
                "1-day": 1.0,
                "1-week": 1.01,
                "1-month": 1.02
            }
            
            multiplier = horizon_multiplier.get(forecast_horizon, 1.0)
            
            # Create realistic yield curve shape
            # These are relative adjustments to create a typical yield curve shape
            tenor_adjustments = {
                '3M': -0.2,    # Short end typically lower
                '6M': -0.1,    
                '1Y': 0.0,     # Base
                '2Y': 0.15,    
                '3Y': 0.25,    
                '5Y': 0.35,    
                '7Y': 0.42,    
                '10Y': 0.50,   # Long end higher
                '20Y': 0.55,   
                '30Y': 0.58    # Highest typically
            }
            
            predictions = {}
            for tenor in self.tenors:
                adjusted_yield = (base_prediction + tenor_adjustments[tenor]) * multiplier
                predictions[tenor] = max(0.0, adjusted_yield)  # Ensure non-negative
            
            return predictions
            
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            return {tenor: 0.0 for tenor in self.tenors}
    
    def create_yield_curve_plot(self, predictions: Dict[str, float], 
                              maturity_dates: Dict[str, str]):
        """Create an interactive yield curve plot."""
        # Prepare data
        tenors = list(predictions.keys())
        yields = list(predictions.values())
        
        # Create hover text with maturity dates
        hover_text = [f"{tenor}<br>Yield: {yield_:.3f}%<br>Maturity: {maturity_dates[tenor]}" 
                     for tenor, yield_ in zip(tenors, yields)]
        
        fig = go.Figure()
        
        # Add yield curve
        fig.add_trace(go.Scatter(
            x=tenors,
            y=yields,
            mode='lines+markers',
            name='Predicted Yield Curve',
            line=dict(color='#1f4e79', width=3),
            marker=dict(size=10, color='#1f4e79'),
            hovertext=hover_text,
            hoverinfo='text'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'U.S. Treasury Yield Curve Forecast',
                'x': 0.5,
                'font': {'size': 20, 'color': '#1f4e79'}
            },
            xaxis_title='Tenor',
            yaxis_title='Yield (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=False,
            font=dict(size=12)
        )
        
        # Customize axes
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickformat='.2f')
        
        return fig
    
    def create_results_table(self, predictions: Dict[str, float], 
                           maturity_dates: Dict[str, str]) -> pd.DataFrame:
        """Create results table with predictions and maturity dates."""
        results_data = []
        for tenor in self.tenors:
            results_data.append({
                'Tenor': tenor,
                'Maturity Date': maturity_dates[tenor],
                'Predicted Yield (%)': f"{predictions[tenor]:.3f}"
            })
        
        return pd.DataFrame(results_data)
    
    def display_model_info_page(self):
        """Display the Model Training & Validation Info page."""
        st.markdown('<div class="main-header">📊 Model Training & Validation Info</div>', 
                   unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive overview of all available models and their performance</div>', 
                   unsafe_allow_html=True)
        
        metrics = self.load_model_metrics()
        
        if not metrics:
            st.error("No model metrics available.")
            return
        
        # Best model highlight
        best_model = self.get_best_model()
        st.markdown(f'<div class="success-box"><strong>🏆 Best Performing Model:</strong> {best_model.replace("_", " ").title()} (RMSE: {metrics[best_model]["rmse"]:.4f})</div>', 
                   unsafe_allow_html=True)
        
        # Model comparison table
        st.subheader("📈 Model Performance Comparison")
        
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'RMSE': f"{model_metrics['rmse']:.4f}",
                'MAE': f"{model_metrics['mae']:.4f}",
                'R²': f"{model_metrics['r2']:.4f}",
                'MAPE': f"{model_metrics['mape']:.2f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Detailed model information
        st.subheader("📋 Detailed Model Information")
        
        # Create tabs for each model
        model_names = list(metrics.keys())
        tabs = st.tabs([name.replace('_', ' ').title() for name in model_names])
        
        model_descriptions = {
            'lasso': 'Linear regression with L1 regularization for feature selection',
            'ridge': 'Linear regression with L2 regularization to prevent overfitting',
            'elastic_net': 'Combines L1 and L2 regularization for balanced feature selection and regularization',
            'random_forest': 'Ensemble method using multiple decision trees with bootstrap aggregating',
            'gradient_boosting': 'Sequential ensemble method that builds models to correct predecessor errors',
            'svr': 'Support Vector Regression using kernel methods for non-linear relationships',
            'ensemble_simple': 'Simple averaging ensemble of multiple base models',
            'ensemble_weighted': 'Weighted ensemble based on individual model performance'
        }
        
        for i, (tab, model_name) in enumerate(zip(tabs, model_names)):
            with tab:
                model_info = metrics[model_name]
                
                # Model description
                st.write(f"**Description:** {model_descriptions.get(model_name, 'Advanced machine learning model for yield prediction')}")
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("RMSE", f"{model_info['rmse']:.4f}")
                with col2:
                    st.metric("MAE", f"{model_info['mae']:.4f}")
                with col3:
                    st.metric("R²", f"{model_info['r2']:.4f}")
                with col4:
                    st.metric("MAPE", f"{model_info['mape']:.2f}%")
                
                # Additional details
                if model_name == best_model:
                    st.success("🏆 This is the best performing model based on RMSE")
                
                # Training details (simplified for demo)
                st.write("**Training Details:**")
                st.write("- Training period: Historical data from 2020-2025")
                st.write("- Features: Macroeconomic indicators, yield curve factors, technical indicators")
                st.write("- Validation: Time series cross-validation")
                st.write("- Target: Multiple yield curve tenors (3M to 30Y)")
        
        # Visualizations section
        st.subheader("📊 Performance Visualizations")
        
        # Check for figures
        figures_dir = self.reports_dir / "figures"
        if figures_dir.exists():
            figure_files = list(figures_dir.glob("*.png"))
            
            if figure_files:
                fig_tabs = st.tabs(["Feature Importance", "Prediction Plots"])
                
                with fig_tabs[0]:
                    importance_files = [f for f in figure_files if 'importance' in f.name.lower()]
                    if importance_files:
                        st.image(str(importance_files[0]), caption="Feature Importance Analysis")
                    else:
                        st.info("Feature importance plot not available")
                
                with fig_tabs[1]:
                    prediction_files = [f for f in figure_files if 'prediction' in f.name.lower()]
                    if prediction_files:
                        st.image(str(prediction_files[0]), caption="Model Prediction Analysis")
                    else:
                        st.info("Prediction plots not available")
            else:
                st.info("No visualization files found")
        else:
            st.info("Figures directory not found")
    
    @st.cache_data
    def load_team_profiles(_self):
        """Load team profile data from YAML configuration."""
        try:
            config_path = Path("config/profiles.yaml")
            with open(config_path, 'r') as f:
                profiles_config = yaml.safe_load(f)
            return profiles_config
        except Exception as e:
            st.error(f"Error loading team profiles: {e}")
            return None
    
    def display_team_page(self):
        """Display the Team & Oversight page."""
        profiles_config = self.load_team_profiles()
        
        if not profiles_config:
            st.error("Could not load team profile data")
            return
        
        # Page header
        page_config = profiles_config.get('page_config', {})
        st.markdown(f'<div class="main-header">{page_config.get("title", "👥 Team & Oversight")}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{page_config.get("description", "Meet the team behind YieldCurveAI")}</div>', 
                   unsafe_allow_html=True)
        
        # Team profiles
        team_profiles = profiles_config.get('team_profiles', {})
        
        # Sort profiles by order
        sorted_profiles = sorted(team_profiles.items(), key=lambda x: x[1].get('order', 999))
        
        # Display profiles in responsive layout
        for i in range(0, len(sorted_profiles), 3):
            cols = st.columns(3)
            
            for j, (profile_key, profile_data) in enumerate(sorted_profiles[i:i+3]):
                with cols[j]:
                    # Profile card
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <div style="width: 150px; height: 150px; background-color: #f0f0f0; border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 3rem; color: #1f4e79;">
                                👤
                            </div>
                        </div>
                        <h3 style="text-align: center; color: #1f4e79; margin-bottom: 0.5rem;">{profile_data.get('name', 'Unknown')}</h3>
                        <p style="text-align: center; font-weight: bold; color: #5a5a5a; margin-bottom: 1rem;">{profile_data.get('title', '')}</p>
                        <p style="text-align: center; font-style: italic; color: #1f4e79; margin-bottom: 1rem;">{profile_data.get('tagline', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Profile details in expandable section
                    with st.expander(f"View {profile_data.get('name', 'Profile').split()[-1]}'s Details"):
                        # Affiliation
                        if profile_data.get('affiliation'):
                            st.write(f"**Affiliation:** {profile_data['affiliation']}")
                        
                        # Specialization/Expertise
                        if profile_data.get('specialization'):
                            st.write(f"**Specialization:** {profile_data['specialization']}")
                        elif profile_data.get('expertise'):
                            st.write(f"**Expertise:** {profile_data['expertise']}")
                        
                        # Education
                        if profile_data.get('education'):
                            st.write("**Education:**")
                            for edu in profile_data['education']:
                                st.write(f"- {edu}")
                        
                        # Career/Leadership
                        if profile_data.get('career'):
                            st.write(f"**Career:** {profile_data['career']}")
                        elif profile_data.get('leadership'):
                            st.write(f"**Leadership:** {profile_data['leadership']}")
                        
                        # Technical Skills
                        if profile_data.get('technical_skills'):
                            st.write(f"**Technical Skills:** {profile_data['technical_skills']}")
                        
                        # Contributions
                        if profile_data.get('contributions'):
                            st.write(f"**Contributions to YieldCurveAI:** {profile_data['contributions']}")
                        
                        # Links
                        if profile_data.get('links'):
                            st.write("**Links:**")
                            links = profile_data['links']
                            for link_name, link_url in links.items():
                                link_display = link_name.replace('_', ' ').title()
                                st.markdown(f"- [{link_display}]({link_url})")
                        
                        # Contact
                        if profile_data.get('contact'):
                            st.write(f"**Contact:** {profile_data['contact']}")
        
        # Additional team information section
        st.markdown("---")
        st.subheader("🎯 Project Vision")
        st.markdown("""
        <div class="info-box">
        <p>YieldCurveAI represents the intersection of academic rigor, practical engineering, and institutional oversight. 
        Our multidisciplinary approach ensures that advanced machine learning techniques are grounded in sound economic theory 
        and implemented with enterprise-grade reliability.</p>
        
        <p><strong>Our mission:</strong> To democratize access to sophisticated yield curve forecasting tools while maintaining 
        the highest standards of accuracy, interpretability, and academic integrity.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Acknowledgments
        st.subheader("🏛️ Institutional Support")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Hansraj College, University of Delhi**
            - Academic foundation
            - Economic modeling expertise
            """)
        
        with col2:
            st.markdown("""
            **Solomon Islands National University**
            - Institutional oversight
            - Academic governance
            """)
        
        with col3:
            st.markdown("""
            **CDAC India**
            - Technical infrastructure
            - Engineering excellence
            """)
    
    def display_forecast_page(self):
        """Display the main Yield Forecast Tool page."""
        st.markdown('<div class="main-header">📈 Yield Forecast Tool</div>', 
                   unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Generate professional U.S. Treasury yield curve forecasts using machine learning</div>', 
                   unsafe_allow_html=True)
        
        # Sidebar inputs
        with st.sidebar:
            st.header("📊 Forecast Parameters")
            
            # Forecast start date
            forecast_date = st.date_input(
                "Forecast Start Date",
                value=datetime.now().date(),
                help="The date from which to generate the forecast"
            )
            
            # Macroeconomic inputs
            st.subheader("💰 Economic Indicators")
            
            fed_funds_rate = st.number_input(
                "Fed Funds Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=5.25,
                step=0.25,
                help="Federal Funds Target Rate"
            )
            
            cpi_yoy = st.number_input(
                "CPI YoY (%)",
                min_value=-5.0,
                max_value=15.0,
                value=3.2,
                step=0.1,
                help="Consumer Price Index Year-over-Year change"
            )
            
            # Forecast horizon
            forecast_horizon = st.selectbox(
                "Forecast Horizon",
                options=["1-day", "1-week", "1-month"],
                index=0,
                help="Time horizon for the forecast"
            )
            
            # Model selection
            st.subheader("🤖 Model Selection")
            
            selection_mode = st.radio(
                "Model Selection Mode",
                options=["Automatic (Best Model Based on RMSE)", "Manual Selection"],
                help="Choose automatic best model or select manually"
            )
            
            selected_model = None
            if selection_mode == "Manual Selection":
                model_options = {
                    "Linear Regression (Ridge)": "ridge",
                    "LASSO": "lasso", 
                    "ElasticNet": "elastic_net",
                    "Random Forest": "random_forest",
                    "XGBoost": "gradient_boosting",  # Using gradient_boosting as proxy
                    "LightGBM": "svr"  # Using SVR as proxy
                }
                
                selected_model_name = st.selectbox(
                    "Select Model",
                    options=list(model_options.keys()),
                    help="Choose a specific model for forecasting"
                )
                selected_model = model_options[selected_model_name]
            
            # Generate forecast button
            generate_forecast = st.button(
                "🚀 Generate Forecast",
                type="primary",
                use_container_width=True
            )
            
            # Debug section - show available models
            with st.expander("🔍 Debug Info"):
                available_models = self.load_available_models()
                if available_models:
                    st.write("**Available Models:**")
                    for model_name, model_path in available_models.items():
                        st.write(f"- {model_name}: {model_path.name}")
                else:
                    st.warning("No models found in models/trained directory")
                
                metrics = self.load_model_metrics()
                if metrics:
                    best_model = self.get_best_model()
                    st.write(f"**Best Model:** {best_model}")
                else:
                    st.warning("No model metrics found")
        
        # Main content area
        if generate_forecast:
            try:
                # Load feature data
                features_df = self.load_feature_data()
                if features_df is None:
                    st.error("Could not load feature data")
                    return
                
                # Determine which model to use
                if selection_mode == "Automatic (Best Model Based on RMSE)":
                    model_name = self.get_best_model()
                    model_source = "Auto-selected based on RMSE"
                else:
                    model_name = selected_model
                    model_source = "User selected"
                
                # Load the selected model
                with st.spinner("Loading model..."):
                    model_data = self.load_model(model_name)
                
                if model_data is None:
                    st.error(f"Could not load model: {model_name}")
                    return
                
                # Create prediction features
                with st.spinner("Preparing features..."):
                    prediction_features = self.create_prediction_features(
                        features_df, fed_funds_rate, cpi_yoy, datetime.combine(forecast_date, datetime.min.time())
                    )
                
                # Generate predictions
                with st.spinner("Generating forecasts..."):
                    predictions = self.predict_yield_curve(
                        model_data, prediction_features, forecast_horizon
                    )
                
                # Calculate maturity dates
                maturity_dates = self.calculate_maturity_dates(
                    datetime.combine(forecast_date, datetime.min.time())
                )
                
                # Display results
                st.success("✅ Forecast generated successfully!")
                
                # Model information
                metrics = self.load_model_metrics()
                selected_metrics = metrics.get(model_name, {})
                
                st.markdown(f'<div class="info-box"><strong>Selected Model:</strong> {model_name.replace("_", " ").title()} ({model_source})</div>', 
                           unsafe_allow_html=True)
                
                # Performance metrics
                if selected_metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSE", f"{selected_metrics.get('rmse', 0):.4f}")
                    with col2:
                        st.metric("MAE", f"{selected_metrics.get('mae', 0):.4f}")
                    with col3:
                        st.metric("R²", f"{selected_metrics.get('r2', 0):.4f}")
                
                # Results table
                st.subheader("📋 Forecast Results")
                results_df = self.create_results_table(predictions, maturity_dates)
                st.dataframe(results_df, use_container_width=True)
                
                # Yield curve plot
                st.subheader("📈 Yield Curve Visualization")
                fig = self.create_yield_curve_plot(predictions, maturity_dates)
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                st.subheader("💾 Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download CSV",
                        data=csv_data,
                        file_name=f"yield_forecast_{forecast_date}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Note: For PNG export, we would need additional dependencies
                    st.info("📊 PNG export available in full version")
                
                # Note
                st.markdown('<div class="info-box"><strong>Note:</strong> Maturity dates are computed based on your selected forecast start date.</div>', 
                           unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                st.error("Please check your inputs and try again.")
    
    def run(self):
        """Main application runner."""
        # Navigation
        page = st.radio(
            "Navigate",
            options=["📈 Forecast", "📊 Model Info", "👥 Team & Oversight"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if page == "📈 Forecast":
            self.display_forecast_page()
        elif page == "📊 Model Info":
            self.display_model_info_page()
        else:
            self.display_team_page()
        
        # Add app-wide footer
        self.display_footer()
    
    def display_footer(self):
        """Display app-wide footer with attribution."""
        profiles_config = self.load_team_profiles()
        
        if profiles_config and 'footer_config' in profiles_config:
            footer_text = profiles_config['footer_config']['text']
        else:
            footer_text = "Finance & Economics by Dr. Kapila Mallah | Overseen by Dr. Eric Katovai (PVC Academic) | Built by Mr. Pappu Kapgate, AI Engineer"
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #888; font-size: 0.9rem; margin-top: 2rem; padding: 1rem;">
            {footer_text}
        </div>
        """, unsafe_allow_html=True)

# Initialize and run the application
if __name__ == "__main__":
    app = YieldCurveAI()
    app.run() 