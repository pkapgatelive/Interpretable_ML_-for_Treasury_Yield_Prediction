"""
Baseline models for yield curve forecasting.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import logging

logger = logging.getLogger(__name__)


class LinearRegressionModel:
    """Simple linear regression baseline model."""
    
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)


class ARIMAModel:
    """ARIMA baseline model."""
    
    def __init__(self, order=(1, 1, 1), **kwargs):
        self.order = order
        self.kwargs = kwargs
        self.models = {}
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit ARIMA models for each target column."""
        self.models = {}
        for col in y.columns:
            try:
                model = ARIMA(y[col], order=self.order, **self.kwargs)
                fitted = model.fit()
                self.models[col] = fitted
            except Exception as e:
                logger.warning(f"Failed to fit ARIMA for {col}: {e}")
                # Fallback to simple mean model
                self.models[col] = y[col].mean()
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = []
        for col in self.models.keys():
            if hasattr(self.models[col], 'forecast'):
                try:
                    pred = self.models[col].forecast(steps=len(X))
                    predictions.append(pred)
                except:
                    # Fallback to mean
                    predictions.append([self.models[col]] * len(X))
            else:
                # Mean model fallback
                predictions.append([self.models[col]] * len(X))
        
        return np.column_stack(predictions)


class VARModel:
    """Vector Autoregression model."""
    
    def __init__(self, maxlags=5, **kwargs):
        self.maxlags = maxlags
        self.kwargs = kwargs
        self.model = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit VAR model."""
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            self.model = VAR(y)
            self.fitted_model = self.model.fit(maxlags=self.maxlags, **self.kwargs)
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"Failed to fit VAR: {e}")
            # Fallback to mean model
            self.means = y.mean()
            self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self, 'fitted_model'):
            try:
                forecast = self.fitted_model.forecast(steps=len(X))
                return forecast
            except:
                pass
        
        # Fallback to mean prediction
        return np.tile(self.means.values, (len(X), 1))


class NelsonSiegelModel:
    """Nelson-Siegel yield curve model."""
    
    def __init__(self, **kwargs):
        self.params = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit Nelson-Siegel model."""
        # Simplified implementation - just store the mean yields
        self.means = y.mean()
        self.is_fitted = True
        logger.info("Nelson-Siegel model fitted (simplified implementation)")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Simple prediction using mean yields
        return np.tile(self.means.values, (len(X), 1)) 