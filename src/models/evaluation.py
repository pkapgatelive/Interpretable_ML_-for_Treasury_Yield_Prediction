"""
Model evaluation utilities for yield curve forecasting models.

This module provides functions for evaluating model performance,
cross-validation, and backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation class for yield curve forecasting models.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the model evaluator.
        
        Parameters
        ----------
        metrics : Optional[List[str]]
            List of metrics to compute. Default includes RMSE, MAE, R2.
        """
        self.metrics = metrics or ['rmse', 'mae', 'r2']
        
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model predictions.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns
        -------
        Dict[str, float]
            Dictionary of metric scores
        """
        results = {}
        
        if 'rmse' in self.metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            
        if 'mae' in self.metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
            
        if 'r2' in self.metrics:
            results['r2'] = r2_score(y_true, y_pred)
            
        return results


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    Dict[str, float]
        Dictionary of metric scores
    """
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv: int = 5,
    test_size: int = 252
) -> Dict[str, Any]:
    """
    Perform time series cross-validation.
    
    Parameters
    ----------
    model : Any
        Model to evaluate
    X : pd.DataFrame
        Features
    y : pd.DataFrame
        Target variables
    cv : int
        Number of CV folds
    test_size : int
        Size of test set for each fold
        
    Returns
    -------
    Dict[str, Any]
        Cross-validation results
    """
    tscv = TimeSeriesSplit(n_splits=cv, test_size=test_size)
    cv_scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        scores = calculate_metrics(y_test.values, y_pred)
        cv_scores.append(scores)
    
    # Aggregate results
    cv_results = {}
    for metric in cv_scores[0].keys():
        scores_list = [score[metric] for score in cv_scores]
        cv_results[f'{metric}_mean'] = np.mean(scores_list)
        cv_results[f'{metric}_std'] = np.std(scores_list)
    
    return cv_results


def backtesting_evaluation(
    model: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    start_date: str,
    end_date: str,
    refit_frequency: str = '1M'
) -> Dict[str, Any]:
    """
    Perform backtesting evaluation.
    
    Parameters
    ----------
    model : Any
        Model to evaluate
    X : pd.DataFrame
        Features with datetime index
    y : pd.DataFrame
        Target variables with datetime index
    start_date : str
        Start date for backtesting
    end_date : str
        End date for backtesting
    refit_frequency : str
        How often to refit the model
        
    Returns
    -------
    Dict[str, Any]
        Backtesting results
    """
    # Filter data for backtesting period
    mask = (X.index >= start_date) & (X.index <= end_date)
    X_backtest = X[mask]
    y_backtest = y[mask]
    
    predictions = []
    actual_values = []
    
    # Simple walk-forward validation
    for i in range(len(X_backtest) - 1):
        X_train = X_backtest.iloc[:i+1]
        y_train = y_backtest.iloc[:i+1]
        X_test = X_backtest.iloc[i+1:i+2]
        y_test = y_backtest.iloc[i+1:i+2]
        
        if len(X_train) > 50:  # Minimum training samples
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            predictions.append(y_pred[0])
            actual_values.append(y_test.values[0])
    
    # Calculate metrics
    if predictions:
        metrics = calculate_metrics(
            np.array(actual_values), 
            np.array(predictions)
        )
        return {
            'metrics': metrics,
            'predictions': predictions,
            'actual_values': actual_values
        }
    else:
        return {'metrics': {}, 'predictions': [], 'actual_values': []} 