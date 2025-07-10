"""
Yield Curve Forecasting Project

Interpretable Machine Learning Models for Yield Curve Forecasting
and Monetary Policy Scenario Analysis.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@institution.edu"

# Import main modules for easier access
from src.data import data_loader, data_cleaner, feature_engineering
from src.models import baseline, ml_models, ensemble, evaluation
from src.visualization import plotting, dashboard
# from src.explainability import shap_analysis, interpretability  # TODO: Create these modules
from src.utils import helpers, constants

__all__ = [
    "data_loader",
    "data_cleaner", 
    "feature_engineering",
    "baseline",
    "ml_models",
    "ensemble",
    "evaluation",
    "plotting",
    "dashboard",
    # "shap_analysis",
    # "interpretability",
    "helpers",
    "constants",
] 