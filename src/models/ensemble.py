"""
Ensemble methods for yield curve forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Any
import logging

logger = logging.getLogger(__name__)


class VotingEnsemble:
    """Voting ensemble of multiple models."""
    
    def __init__(self, estimators: List[Any], voting='soft'):
        self.estimators = estimators
        self.voting = voting
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit all base models."""
        for estimator in self.estimators:
            estimator.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for estimator in self.estimators:
            pred = estimator.predict(X)
            predictions.append(pred)
        
        # Average predictions
        return np.mean(predictions, axis=0)


class StackingEnsemble:
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, base_models: List[Any], meta_model=None):
        self.base_models = base_models
        if meta_model is None:
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression()
        else:
            self.meta_model = meta_model
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit base models and meta-learner."""
        # Fit base models
        base_predictions = []
        for model in self.base_models:
            model.fit(X, y)
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack predictions and fit meta-model
        stacked_features = np.column_stack(base_predictions)
        self.meta_model.fit(stacked_features, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make stacked predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            base_predictions.append(pred)
        
        # Stack and predict with meta-model
        stacked_features = np.column_stack(base_predictions)
        return self.meta_model.predict(stacked_features)


class BaggingEnsemble:
    """Bagging ensemble with bootstrap sampling."""
    
    def __init__(self, base_model, n_estimators=10, random_state=None):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit models with bootstrap samples."""
        np.random.seed(self.random_state)
        self.models = []
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            n_samples = len(X)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Train model
            model = type(self.base_model)(**self.base_model.__dict__)
            model.fit(X_boot, y_boot)
            self.models.append(model)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make bagged predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Average predictions
        return np.mean(predictions, axis=0) 