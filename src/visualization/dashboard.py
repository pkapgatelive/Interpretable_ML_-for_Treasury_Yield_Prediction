"""
Dashboard utilities for yield curve forecasting.

This module provides functions for creating interactive dashboards
using Streamlit and Plotly.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class YieldCurveDashboard:
    """
    Interactive yield curve dashboard class.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the dashboard with yield curve data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Yield curve data with dates as index and tenors as columns
        """
        self.data = data
        
    def create_plotly_chart(self) -> go.Figure:
        """
        Create a Plotly chart of the yield curve.
        
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add trace for latest date
        latest_date = self.data.index[-1]
        latest_yields = self.data.loc[latest_date]
        
        fig.add_trace(go.Scatter(
            x=latest_yields.index,
            y=latest_yields.values,
            mode='lines+markers',
            name=f'Yield Curve - {latest_date}',
            line=dict(width=3)
        ))
        
        fig.update_layout(
            title='Yield Curve',
            xaxis_title='Tenor (Years)',
            yaxis_title='Yield (%)',
            hovermode='x unified'
        )
        
        return fig


def create_streamlit_dashboard(data: pd.DataFrame) -> None:
    """
    Create a Streamlit dashboard for yield curve analysis.
    
    Parameters
    ----------
    data : pd.DataFrame
        Yield curve data with dates as index and tenors as columns
    """
    st.title("Yield Curve Forecasting Dashboard")
    st.sidebar.title("Controls")
    
    # Date selection
    selected_date = st.sidebar.selectbox(
        "Select Date",
        options=data.index[-10:]  # Show last 10 dates
    )
    
    # Display yield curve for selected date
    st.subheader(f"Yield Curve - {selected_date}")
    
    curve_data = data.loc[selected_date]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_data.index,
        y=curve_data.values,
        mode='lines+markers',
        name='Yield Curve'
    ))
    
    fig.update_layout(
        xaxis_title='Tenor (Years)',
        yaxis_title='Yield (%)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(data.tail(10))


def create_plotly_dashboard(data: pd.DataFrame) -> go.Figure:
    """
    Create a Plotly dashboard figure.
    
    Parameters
    ----------
    data : pd.DataFrame
        Yield curve data with dates as index and tenors as columns
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add multiple yield curves
    for i, date in enumerate(data.index[-5:]):  # Show last 5 dates
        yields = data.loc[date]
        
        fig.add_trace(go.Scatter(
            x=yields.index,
            y=yields.values,
            mode='lines+markers',
            name=str(date),
            opacity=0.7
        ))
    
    fig.update_layout(
        title='Yield Curve Evolution',
        xaxis_title='Tenor (Years)',
        yaxis_title='Yield (%)',
        height=600,
        hovermode='x unified'
    )
    
    return fig 