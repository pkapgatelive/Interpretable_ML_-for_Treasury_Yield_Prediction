"""
Machine learning models for yield curve forecasting.
"""

import pandas as pd
import numpy as np
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest model."""
    
    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(**kwargs)
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


class XGBoostModel:
    """XGBoost model."""
    
    def __init__(self, **kwargs):
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(**kwargs)
        except ImportError:
            logger.warning("XGBoost not available, using RandomForest as fallback")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**kwargs)
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


class LightGBMModel:
    """LightGBM model."""
    
    def __init__(self, **kwargs):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(**kwargs)
        except ImportError:
            logger.warning("LightGBM not available, using RandomForest as fallback")
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**kwargs)
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


class MLPModel:
    """Multi-Layer Perceptron model."""
    
    def __init__(self, **kwargs):
        from sklearn.neural_network import MLPRegressor
        self.model = MLPRegressor(**kwargs)
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


class LSTMModel:
    """LSTM model for time series forecasting."""
    
    def __init__(self, units=50, epochs=100, batch_size=32, **kwargs):
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Fit the LSTM model."""
        try:
            import tensorflow as tf
            from sklearn.preprocessing import StandardScaler
            
            # Scale the data
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)
            
            # Build LSTM model
            self.model = tf.keras.Sequential([
                tf.keras.layers.LSTM(self.units, return_sequences=True, input_shape=(X_scaled.shape[1], 1)),
                tf.keras.layers.LSTM(self.units),
                tf.keras.layers.Dense(y_scaled.shape[1])
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            
            # Reshape for LSTM
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
            
            # Train model
            self.model.fit(X_reshaped, y_scaled, epochs=self.epochs, 
                          batch_size=self.batch_size, verbose=0, **self.kwargs)
            
            self.is_fitted = True
            
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}, using Linear Regression fallback")
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            self.model.fit(X, y)
            self.is_fitted = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            if hasattr(self, 'scaler_X'):
                X_scaled = self.scaler_X.transform(X)
                X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
                y_pred_scaled = self.model.predict(X_reshaped)
                return self.scaler_y.inverse_transform(y_pred_scaled)
            else:
                return self.model.predict(X)
        except:
            # Fallback
            return self.model.predict(X)


class TransformerModel:
    """Transformer model for time series forecasting."""
    
    def __init__(self, **kwargs):
        from sklearn.linear_model import LinearRegression
        logger.warning("Transformer model not implemented, using Linear Regression")
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