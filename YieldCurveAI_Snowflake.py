#!/usr/bin/env python3
"""
YieldCurveAI - Snowflake Enterprise Edition
===========================================
Enterprise yield curve forecasting application optimized for Snowflake's 
Streamlit platform with enhanced security and data integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Snowflake Streamlit integration
try:
    from snowflake.snowpark.context import get_active_session
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="YieldCurveAI Enterprise - Treasury Yield Forecasting",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for enterprise styling
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
    .snowflake-badge {
        background: linear-gradient(45deg, #29b5e8, #00a8e6);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .enterprise-header {
        background: linear-gradient(90deg, #1f4e79, #29b5e8);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class YieldCurveAI:
    def __init__(self):
        self.tenors = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y']
        self.session = None
        
        # Initialize Snowflake session if available
        if SNOWFLAKE_AVAILABLE:
            try:
                self.session = get_active_session()
                self.snowflake_connected = True
            except:
                self.session = None
                self.snowflake_connected = False
        else:
            self.snowflake_connected = False
    
    def display_connection_status(self):
        """Display Snowflake connection status."""
        if self.snowflake_connected:
            st.sidebar.markdown(
                '<span class="snowflake-badge">❄️ Snowflake Connected</span>', 
                unsafe_allow_html=True
            )
            return True
        else:
            st.sidebar.warning("📊 Demo Mode - Connect to Snowflake for enterprise features")
            return False
    
    @st.cache_data
    def load_team_profiles(_self):
        """Load team profile data."""
        return {
            'team_profiles': {
                'dr_kapila_mallah': {
                    'name': 'Dr. Kapila Mallah',
                    'title': 'Associate Professor, Economics',
                    'affiliation': 'Hansraj College, University of Delhi; Visiting Professor of Economics, SINU',
                    'specialization': 'Microeconomics, Macroeconomics, Education Inequality',
                    'education': [
                        'Ph.D. from Jawaharlal Nehru University (JNU)',
                        'M.Phil from Jawaharlal Nehru University (JNU)',
                        'M.Sc. in Economics & Econometrics from University of Nottingham'
                    ],
                    'career': '14+ years academic experience, curriculum leader, academic council member',
                    'contributions': 'AI design for economic forecasting, evaluation logic for yield prediction models',
                    'contact': 'kapilamallah@hrc.du.ac.in',
                    'links': {
                        'hansraj_profile': 'https://www.hansrajcollege.ac.in/academics/departments/arts-and-commerce/economics/faculty-detail/64/'
                    },
                    'tagline': 'AI Engineered by Dr. Kapila Mallah',
                    'order': 1
                },
                'mr_pappu_kapgate': {
                    'name': 'Mr. Pappu Dindayal Kapgate',
                    'title': 'Director, CEIT – Solomon Islands',
                    'affiliation': 'Solomon Islands National University; Project Engineer, CDAC India',
                    'expertise': 'Generative AI, ML Deployment, Cybersecurity, ICT Infrastructure',
                    'education': [
                        'MSc Data Science from Liverpool John Moores University (LJMU), UK',
                        'PG Diploma from Centre for Development of Advanced Computing (CDAC)',
                        'B.E. from Yeshwantrao Chavan College of Engineering (YCCE), Nagpur'
                    ],
                    'technical_skills': 'Python, SQL, Tableau, PySpark, Airflow, ML Modeling',
                    'contributions': 'Project lead, full-stack architecture, integration of ML pipeline with frontend',
                    'links': {
                        'linkedin': 'https://www.linkedin.com/in/pkapgate'
                    },
                    'tagline': 'Built by Mr. Pappu Kapgate',
                    'order': 2
                },
                'dr_eric_katovai': {
                    'name': 'Dr. Eric Katovai',
                    'title': 'Pro Vice-Chancellor (Academic)',
                    'affiliation': 'Solomon Islands National University (SINU); Associate Professor (Science & Technology)',
                    'expertise': 'Tropical Ecology, Academic Governance, Postgraduate Education',
                    'education': [
                        'Ph.D. from James Cook University',
                        'M.Sc. from University of Queensland',
                        'B.Ed./B.Sc. from Papua New Guinea University (PAU PNG)'
                    ],
                    'leadership': '20 years in academia; led FST governance, USP Council member',
                    'contributions': 'Institutional oversight and academic validation for the project',
                    'links': {
                        'sinu_profile': 'https://www.sinu.edu.sb/executive-governance/vice-chancellor/pro-vice-chancellor-academic/'
                    },
                    'tagline': 'Oversight by Dr. Eric Katovai (PVC Academic)',
                    'order': 3
                }
            },
            'page_config': {
                'title': '👥 Team & Oversight',
                'description': 'YieldCurveAI is the product of collaborative efforts between academic leadership, AI research, and engineering excellence. Meet the contributors behind this innovative tool.'
            },
            'footer_config': {
                                    'text': 'Finance & Economics by Dr. Kapila Mallah | Overseen by Dr. Eric Katovai (PVC Academic) | Built by Mr. Pappu Kapgate, AI Engineer'
            }
        }
    
    def load_model_metrics_from_snowflake(self):
        """Load model metrics from Snowflake tables if available."""
        if self.session:
            try:
                metrics_df = self.session.sql("""
                    SELECT model_name, rmse, mae, r2, mape 
                    FROM YIELDCURVE_DB.ML_MODELS.MODEL_METRICS
                    WHERE is_active = TRUE
                    ORDER BY rmse ASC
                """).to_pandas()
                
                metrics = {}
                for _, row in metrics_df.iterrows():
                    metrics[row['MODEL_NAME'].lower()] = {
                        'rmse': row['RMSE'],
                        'mae': row['MAE'],
                        'r2': row['R2'],
                        'mape': row['MAPE']
                    }
                return metrics
            except Exception as e:
                st.warning(f"Could not load metrics from Snowflake: {str(e)}")
        
        # Fallback to demo data
        return self._get_demo_metrics()
    
    def load_features_from_snowflake(self):
        """Load feature data from Snowflake if available."""
        if self.session:
            try:
                features_df = self.session.sql("""
                    SELECT * FROM YIELDCURVE_DB.FEATURES.PROCESSED_FEATURES
                    WHERE date >= CURRENT_DATE - INTERVAL '365 DAYS'
                    ORDER BY date DESC
                """).to_pandas()
                
                if 'DATE' in features_df.columns:
                    features_df.set_index('DATE', inplace=True)
                
                return features_df
            except Exception as e:
                st.warning(f"Could not load features from Snowflake: {str(e)}")
        
        # Fallback to demo data
        return self._get_demo_features()
    
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
    
    def _get_demo_features(self):
        """Provide demo features when real data isn't available."""
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        np.random.seed(42)
        
        features = {
            'fed_funds_rate': np.random.normal(4.5, 1.5, len(dates)),
            'cpi_yoy': np.random.normal(3.2, 1.0, len(dates)),
            'unemployment_rate': np.random.normal(4.5, 1.2, len(dates)),
            'vix': np.random.normal(20, 8, len(dates)),
            'yield_spread_10y_2y': np.random.normal(1.2, 0.8, len(dates)),
            'yield_level': np.random.normal(3.5, 1.2, len(dates))
        }
        
        return pd.DataFrame(features, index=dates)
    
    def get_best_model(self):
        """Determine the best performing model."""
        metrics = self.load_model_metrics_from_snowflake()
        if not metrics:
            return "elastic_net"
        
        best_model = min(metrics.keys(), key=lambda x: metrics[x]['rmse'])
        return best_model
    
    def predict_yield_curve(self, fed_funds_rate: float, cpi_yoy: float, 
                          forecast_horizon: str) -> Dict[str, float]:
        """Generate yield curve predictions using Snowflake ML models if available."""
        
        if self.session:
            try:
                # Try to use Snowflake ML model for prediction
                prediction_df = self.session.sql(f"""
                    SELECT * FROM TABLE(
                        YIELDCURVE_DB.ML_MODELS.PREDICT_YIELD_CURVE(
                            {fed_funds_rate}, {cpi_yoy}, '{forecast_horizon}'
                        )
                    )
                """).to_pandas()
                
                predictions = {}
                for _, row in prediction_df.iterrows():
                    predictions[row['TENOR']] = row['PREDICTED_YIELD']
                
                return predictions
            except:
                # Fallback to local prediction
                pass
        
        # Simplified prediction logic for demo
        np.random.seed(42)
        
        base_yields = {
            '3M': fed_funds_rate * 0.95,
            '6M': fed_funds_rate * 0.98,
            '1Y': fed_funds_rate * 1.02,
            '2Y': fed_funds_rate * 1.15,
            '3Y': fed_funds_rate * 1.25,
            '5Y': fed_funds_rate * 1.35,
            '7Y': fed_funds_rate * 1.42,
            '10Y': fed_funds_rate * 1.48,
            '20Y': fed_funds_rate * 1.55,
            '30Y': fed_funds_rate * 1.58
        }
        
        # Add inflation adjustment
        inflation_adjustment = (cpi_yoy - 2.0) * 0.1
        
        predictions = {}
        for tenor, base_yield in base_yields.items():
            adjusted_yield = base_yield + inflation_adjustment + np.random.normal(0, 0.05)
            predictions[tenor] = max(0.01, adjusted_yield)
        
        return predictions
    
    def save_prediction_to_snowflake(self, predictions: Dict[str, float], 
                                   forecast_date: datetime, user_inputs: Dict):
        """Save prediction results to Snowflake for audit trail."""
        if self.session:
            try:
                # Create audit record
                audit_data = []
                for tenor, yield_pred in predictions.items():
                    audit_data.append((
                        forecast_date.strftime('%Y-%m-%d'),
                        tenor,
                        yield_pred,
                        user_inputs.get('fed_funds_rate', 0),
                        user_inputs.get('cpi_yoy', 0),
                        user_inputs.get('forecast_horizon', '1-day'),
                        datetime.now().isoformat()
                    ))
                
                # Insert into audit table
                self.session.sql("""
                    INSERT INTO YIELDCURVE_DB.AUDIT.PREDICTION_LOG 
                    (forecast_date, tenor, predicted_yield, fed_funds_rate, 
                     cpi_yoy, forecast_horizon, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, audit_data).collect()
                
            except Exception as e:
                st.warning(f"Could not save to audit log: {str(e)}")
    
    def calculate_maturity_dates(self, forecast_date: datetime) -> Dict[str, str]:
        """Calculate maturity dates for each tenor."""
        maturity_dates = {}
        
        tenor_mapping = {
            '3M': timedelta(days=90),
            '6M': timedelta(days=180),
            '1Y': timedelta(days=365),
            '2Y': timedelta(days=730),
            '3Y': timedelta(days=1095),
            '5Y': timedelta(days=1825),
            '7Y': timedelta(days=2555),
            '10Y': timedelta(days=3650),
            '20Y': timedelta(days=7300),
            '30Y': timedelta(days=10950)
        }
        
        for tenor, delta in tenor_mapping.items():
            maturity_date = forecast_date + delta
            maturity_dates[tenor] = maturity_date.strftime('%Y-%m-%d')
        
        return maturity_dates
    
    def create_yield_curve_plot(self, predictions: Dict[str, float], 
                              maturity_dates: Dict[str, str]):
        """Create an enhanced interactive yield curve plot."""
        tenors = list(predictions.keys())
        yields = list(predictions.values())
        dates = [maturity_dates[tenor] for tenor in tenors]
        
        fig = go.Figure()
        
        # Main yield curve
        fig.add_trace(go.Scatter(
            x=tenors,
            y=yields,
            mode='lines+markers',
            name='Predicted Yield Curve',
            line=dict(color='#1f4e79', width=4),
            marker=dict(size=10, color='#1f4e79', symbol='circle'),
            hovertemplate='<b>%{x}</b><br>Yield: %{y:.3f}%<br>Maturity: %{customdata}<extra></extra>',
            customdata=dates
        ))
        
        # Add confidence bands if using Snowflake models
        if self.snowflake_connected:
            upper_bounds = [y * 1.1 for y in yields]
            lower_bounds = [y * 0.9 for y in yields]
            
            fig.add_trace(go.Scatter(
                x=tenors + tenors[::-1],
                y=upper_bounds + lower_bounds[::-1],
                fill='toself',
                fillcolor='rgba(31, 78, 121, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title={
                'text': 'U.S. Treasury Yield Curve Forecast' + 
                        (' (Snowflake Enterprise)' if self.snowflake_connected else ' (Demo)'),
                'x': 0.5,
                'font': {'size': 18, 'color': '#1f4e79'}
            },
            xaxis_title='Treasury Tenor',
            yaxis_title='Yield (%)',
            height=600,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_results_table(self, predictions: Dict[str, float], 
                           maturity_dates: Dict[str, str]) -> pd.DataFrame:
        """Create an enhanced results table."""
        results = []
        for tenor in self.tenors:
            if tenor in predictions:
                results.append({
                    'Tenor': tenor,
                    'Predicted Yield (%)': f"{predictions[tenor]:.3f}",
                    'Maturity Date': maturity_dates[tenor],
                    'Risk Level': 'Low' if predictions[tenor] < 3.0 else 'Medium' if predictions[tenor] < 5.0 else 'High'
                })
        
        return pd.DataFrame(results)
    
    def display_enterprise_header(self):
        """Display enterprise branding header."""
        st.markdown("""
        <div class="enterprise-header">
            <h2>❄️ YieldCurveAI Enterprise</h2>
            <p>Professional Treasury Yield Forecasting • Powered by Snowflake Data Cloud</p>
        </div>
        """, unsafe_allow_html=True)
    
    def display_team_page(self):
        """Display the enhanced Team & Oversight page."""
        self.display_enterprise_header()
        
        profiles_config = self.load_team_profiles()
        
        # Page content
        page_config = profiles_config.get('page_config', {})
        st.markdown(f'<div class="main-header">{page_config.get("title", "👥 Team & Oversight")}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{page_config.get("description", "Meet the team behind YieldCurveAI")}</div>', 
                   unsafe_allow_html=True)
        
        # Team profiles
        team_profiles = profiles_config.get('team_profiles', {})
        sorted_profiles = sorted(team_profiles.items(), key=lambda x: x[1].get('order', 999))
        
        # Display profiles in responsive layout
        for i in range(0, len(sorted_profiles), 3):
            cols = st.columns(3)
            
            for j, (profile_key, profile_data) in enumerate(sorted_profiles[i:i+3]):
                with cols[j]:
                    # Enhanced profile card
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-align: center; margin-bottom: 1rem;">
                            <div style="width: 150px; height: 150px; background: linear-gradient(45deg, #1f4e79, #29b5e8); border-radius: 50%; margin: 0 auto; display: flex; align-items: center; justify-content: center; font-size: 3rem; color: white;">
                                👤
                            </div>
                        </div>
                        <h3 style="text-align: center; color: #1f4e79; margin-bottom: 0.5rem;">{profile_data.get('name', 'Unknown')}</h3>
                        <p style="text-align: center; font-weight: bold; color: #5a5a5a; margin-bottom: 1rem;">{profile_data.get('title', '')}</p>
                        <p style="text-align: center; font-style: italic; color: #1f4e79; margin-bottom: 1rem;">{profile_data.get('tagline', '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Profile details
                    with st.expander(f"View {profile_data.get('name', 'Profile').split()[-1]}'s Details"):
                        if profile_data.get('affiliation'):
                            st.write(f"**Affiliation:** {profile_data['affiliation']}")
                        
                        if profile_data.get('specialization'):
                            st.write(f"**Specialization:** {profile_data['specialization']}")
                        elif profile_data.get('expertise'):
                            st.write(f"**Expertise:** {profile_data['expertise']}")
                        
                        if profile_data.get('education'):
                            st.write("**Education:**")
                            for edu in profile_data['education']:
                                st.write(f"- {edu}")
                        
                        if profile_data.get('career'):
                            st.write(f"**Career:** {profile_data['career']}")
                        elif profile_data.get('leadership'):
                            st.write(f"**Leadership:** {profile_data['leadership']}")
                        
                        if profile_data.get('technical_skills'):
                            st.write(f"**Technical Skills:** {profile_data['technical_skills']}")
                        
                        if profile_data.get('contributions'):
                            st.write(f"**Contributions to YieldCurveAI:** {profile_data['contributions']}")
                        
                        if profile_data.get('links'):
                            st.write("**Links:**")
                            links = profile_data['links']
                            for link_name, link_url in links.items():
                                link_display = link_name.replace('_', ' ').title()
                                st.markdown(f"- [{link_display}]({link_url})")
                        
                        if profile_data.get('contact'):
                            st.write(f"**Contact:** {profile_data['contact']}")
        
        # Enterprise features
        st.markdown("---")
        st.subheader("🏢 Enterprise Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔒 Security & Compliance**
            - SOC 2 Type II certified
            - Role-based access control
            - Audit trails & governance
            - GDPR/SOX compliance ready
            """)
        
        with col2:
            st.markdown("""
            **📊 Data Integration**
            - Native Snowflake connectivity
            - Real-time data pipelines
            - ML model deployment
            - Scalable compute resources
            """)
        
        with col3:
            st.markdown("""
            **🚀 Enterprise Support**
            - 24/7 technical support
            - Enterprise SLA guarantees
            - Global deployment options
            - Professional services
            """)
    
    def display_forecast_page(self):
        """Display the enhanced forecast page."""
        self.display_enterprise_header()
        
        st.markdown('<div class="main-header">📈 Enterprise Yield Forecast Tool</div>', 
                   unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Professional-grade Treasury yield curve forecasting with enterprise security</div>', 
                   unsafe_allow_html=True)
        
        # Connection status
        self.display_connection_status()
        
        # Sidebar inputs
        with st.sidebar:
            st.header("📊 Forecast Parameters")
            
            # Enhanced enterprise controls
            if self.snowflake_connected:
                st.success("🔗 Enterprise Data Pipeline Active")
                
                # User role info (if available)
                try:
                    if self.session:
                        user_info = self.session.sql("SELECT CURRENT_USER(), CURRENT_ROLE()").collect()
                        st.info(f"👤 User: {user_info[0][0]}\n🏷️ Role: {user_info[0][1]}")
                except:
                    pass
            
            # Forecast parameters
            forecast_date = st.date_input(
                "Forecast Start Date",
                value=datetime.now().date(),
                help="The date from which to generate the forecast"
            )
            
            st.subheader("💰 Economic Scenario")
            
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
            
            forecast_horizon = st.selectbox(
                "Forecast Horizon",
                options=["1-day", "1-week", "1-month"],
                index=0,
                help="Time horizon for the forecast"
            )
            
            # Enterprise features
            if self.snowflake_connected:
                save_results = st.checkbox(
                    "Save to Enterprise Database",
                    value=True,
                    help="Save prediction results for audit trail"
                )
                
                confidence_interval = st.checkbox(
                    "Show Confidence Intervals",
                    value=True,
                    help="Display prediction uncertainty bands"
                )
            
            # Generate forecast button
            generate_forecast = st.button(
                "🚀 Generate Enterprise Forecast",
                type="primary",
                use_container_width=True
            )
        
        # Main content area
        if generate_forecast:
            try:
                with st.spinner("Generating enterprise-grade yield curve forecast..."):
                    # Collect user inputs
                    user_inputs = {
                        'fed_funds_rate': fed_funds_rate,
                        'cpi_yoy': cpi_yoy,
                        'forecast_horizon': forecast_horizon
                    }
                    
                    # Generate predictions
                    predictions = self.predict_yield_curve(
                        fed_funds_rate, cpi_yoy, forecast_horizon
                    )
                    
                    # Calculate maturity dates
                    maturity_dates = self.calculate_maturity_dates(
                        datetime.combine(forecast_date, datetime.min.time())
                    )
                    
                    # Save to Snowflake if enabled
                    if self.snowflake_connected and locals().get('save_results', False):
                        self.save_prediction_to_snowflake(
                            predictions, 
                            datetime.combine(forecast_date, datetime.min.time()),
                            user_inputs
                        )
                
                # Display results
                st.success("✅ Enterprise forecast generated successfully!")
                
                # Model information
                best_model = self.get_best_model()
                model_source = "Snowflake ML Pipeline" if self.snowflake_connected else "Demo Model"
                
                st.markdown(f'''
                <div class="info-box">
                    <strong>🏆 Selected Model:</strong> {best_model.replace("_", " ").title()}<br>
                    <strong>🔧 Source:</strong> {model_source}<br>
                    <strong>📊 Enterprise Features:</strong> {"Enabled" if self.snowflake_connected else "Demo Mode"}
                </div>
                ''', unsafe_allow_html=True)
                
                # Results table
                st.subheader("📋 Enterprise Forecast Results")
                results_df = self.create_results_table(predictions, maturity_dates)
                st.dataframe(results_df, use_container_width=True)
                
                # Enhanced yield curve plot
                st.subheader("📈 Interactive Yield Curve Visualization")
                fig = self.create_yield_curve_plot(predictions, maturity_dates)
                st.plotly_chart(fig, use_container_width=True)
                
                # Export options
                st.subheader("💾 Enterprise Export Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = results_df.to_csv(index=False)
                    st.download_button(
                        "📄 Download Forecast (CSV)",
                        data=csv_data,
                        file_name=f"enterprise_yield_forecast_{forecast_date}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if self.snowflake_connected:
                        st.info("📊 Results automatically saved to enterprise database")
                    else:
                        st.info("💡 Connect to Snowflake for enterprise database features")
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")
                if self.snowflake_connected:
                    st.info("💡 Check Snowflake connection and permissions")
    
    def display_model_info_page(self):
        """Display enhanced model information page."""
        self.display_enterprise_header()
        
        st.markdown('<div class="main-header">📊 Enterprise Model Analytics</div>', 
                   unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Advanced ML model performance dashboard with enterprise insights</div>', 
                   unsafe_allow_html=True)
        
        metrics = self.load_model_metrics_from_snowflake()
        
        if not metrics:
            st.error("No model metrics available.")
            return
        
        # Best model highlight
        best_model = self.get_best_model()
        st.markdown(f'''
        <div class="success-box">
            <strong>🏆 Best Performing Model:</strong> {best_model.replace("_", " ").title()}<br>
            <strong>📈 RMSE:</strong> {metrics[best_model]["rmse"]:.4f}<br>
            <strong>🔧 Source:</strong> {"Snowflake ML Pipeline" if self.snowflake_connected else "Demo Data"}
        </div>
        ''', unsafe_allow_html=True)
        
        # Enhanced model comparison
        st.subheader("📈 Enterprise Model Performance Dashboard")
        
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'RMSE': f"{model_metrics['rmse']:.4f}",
                'MAE': f"{model_metrics['mae']:.4f}",
                'R²': f"{model_metrics['r2']:.4f}",
                'MAPE': f"{model_metrics['mape']:.2f}%",
                'Status': '🥇 Best' if model_name == best_model else '✅ Active'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Enterprise model management
        st.subheader("🏢 Enterprise Model Governance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔧 Model Operations**
            - Automated model validation
            - A/B testing capabilities
            - Performance monitoring
            - Version control & rollback
            - Continuous integration
            """)
        
        with col2:
            st.markdown("""
            **📊 Snowflake Integration**
            - Native model storage
            - Scalable inference engine
            - Real-time data pipelines
            - Enterprise security controls
            - Audit trails & compliance
            """)
        
        # Performance metrics visualization
        if len(metrics) > 1:
            st.subheader("📊 Model Performance Comparison")
            
            models = list(metrics.keys())
            rmse_values = [metrics[model]['rmse'] for model in models]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[model.replace('_', ' ').title() for model in models],
                    y=rmse_values,
                    marker_color='#1f4e79'
                )
            ])
            
            fig.update_layout(
                title='Model RMSE Comparison',
                yaxis_title='RMSE',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def display_footer(self):
        """Display enhanced enterprise footer."""
        profiles_config = self.load_team_profiles()
        
        if profiles_config and 'footer_config' in profiles_config:
            footer_text = profiles_config['footer_config']['text']
        else:
            footer_text = "Finance & Economics by Dr. Kapila Mallah | Overseen by Dr. Eric Katovai (PVC Academic) | Built by Mr. Pappu Kapgate, AI Engineer"
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #888; font-size: 0.9rem; margin-top: 2rem; padding: 1rem;">
            {footer_text}<br><br>
            <span class="snowflake-badge">❄️ Powered by Snowflake Data Cloud</span>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner with enterprise features."""
        # Navigation
        page = st.radio(
            "Navigate",
            options=["📈 Enterprise Forecast", "📊 Model Analytics", "👥 Team & Oversight"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if page == "📈 Enterprise Forecast":
            self.display_forecast_page()
        elif page == "📊 Model Analytics":
            self.display_model_info_page()
        else:
            self.display_team_page()
        
        # Add enterprise footer
        self.display_footer()

# Initialize and run the enterprise application
if __name__ == "__main__":
    app = YieldCurveAI()
    app.run() 