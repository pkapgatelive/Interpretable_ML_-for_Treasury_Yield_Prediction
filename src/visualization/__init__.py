"""
Visualization package for yield curve forecasting.

This package contains modules for:
- Plotting yield curves and forecasts
- Creating interactive dashboards
- Generating publication-ready figures
"""

from src.visualization.plotting import (
    YieldCurvePlotter,
    plot_yield_curve,
    plot_forecast,
    plot_model_performance,
    plot_feature_importance,
    plot_correlation_matrix,
)

from src.visualization.dashboard import (
    create_streamlit_dashboard,
    create_plotly_dashboard,
    YieldCurveDashboard,
)

__all__ = [
    # Plotting functions
    "YieldCurvePlotter",
    "plot_yield_curve",
    "plot_forecast",
    "plot_model_performance",
    "plot_feature_importance",
    "plot_correlation_matrix",
    
    # Dashboard functions
    "create_streamlit_dashboard",
    "create_plotly_dashboard",
    "YieldCurveDashboard",
] 