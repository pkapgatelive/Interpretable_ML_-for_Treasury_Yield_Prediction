"""
Models package for yield curve forecasting.

This package contains modules for:
- Baseline models (Linear regression, ARIMA, Nelson-Siegel)
- Machine learning models (Random Forest, XGBoost, Neural Networks)
- Ensemble methods (Voting, Stacking, Bagging)
- Model evaluation and comparison
"""

from src.models.baseline import (
    LinearRegressionModel,
    ARIMAModel,
    VARModel,
    NelsonSiegelModel,
)

from src.models.ml_models import (
    RandomForestModel,
    XGBoostModel,
    LightGBMModel,
    MLPModel,
    LSTMModel,
    TransformerModel,
)

from src.models.ensemble import (
    VotingEnsemble,
    StackingEnsemble,
    BaggingEnsemble,
)

from src.models.evaluation import (
    ModelEvaluator,
    calculate_metrics,
    cross_validate_model,
    backtesting_evaluation,
)

__all__ = [
    # Baseline models
    "LinearRegressionModel",
    "ARIMAModel",
    "VARModel", 
    "NelsonSiegelModel",
    
    # ML models
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "MLPModel",
    "LSTMModel",
    "TransformerModel",
    
    # Ensemble methods
    "VotingEnsemble",
    "StackingEnsemble",
    "BaggingEnsemble",
    
    # Evaluation
    "ModelEvaluator",
    "calculate_metrics",
    "cross_validate_model",
    "backtesting_evaluation",
] 