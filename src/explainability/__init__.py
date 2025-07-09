"""
Explainability package for yield curve forecasting models.

This package contains modules for:
- SHAP (SHapley Additive exPlanations) analysis
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Model interpretability tools
"""

from src.explainability.shap_analysis import (
    SHAPExplainer,
    calculate_shap_values,
    plot_shap_summary,
    plot_shap_waterfall,
    plot_shap_dependence,
)

from src.explainability.interpretability import (
    ModelInterpreter,
    calculate_feature_importance,
    plot_partial_dependence,
    analyze_model_behavior,
    generate_interpretation_report,
)

__all__ = [
    # SHAP analysis
    "SHAPExplainer",
    "calculate_shap_values",
    "plot_shap_summary",
    "plot_shap_waterfall",
    "plot_shap_dependence",
    
    # General interpretability
    "ModelInterpreter",
    "calculate_feature_importance",
    "plot_partial_dependence",
    "analyze_model_behavior",
    "generate_interpretation_report",
] 