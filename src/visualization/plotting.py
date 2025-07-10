"""
Plotting utilities for yield curve forecasting visualizations.

This module provides functions for creating publication-ready plots
of yield curves, forecasts, and model performance metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class YieldCurvePlotter:
    """
    Yield curve plotting class with consistent styling.
    """
    
    def __init__(self, figsize: tuple = (12, 8), dpi: int = 300):
        """
        Initialize the plotter with default settings.
        
        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        dpi : int
            Dots per inch for output
        """
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_curve(self, yields: pd.Series, title: str = "Yield Curve") -> plt.Figure:
        """
        Plot a single yield curve.
        
        Parameters
        ----------
        yields : pd.Series
            Yield values with tenor as index
        title : str
            Plot title
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        ax.plot(yields.index, yields.values, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Tenor (Years)')
        ax.set_ylabel('Yield (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig


def plot_yield_curve(
    yields: pd.DataFrame,
    date: Optional[str] = None,
    title: str = "Yield Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot yield curve for a specific date.
    
    Parameters
    ----------
    yields : pd.DataFrame
        Yield data with dates as index and tenors as columns
    date : Optional[str]
        Specific date to plot, if None uses latest
    title : str
        Plot title
    save_path : Optional[str]
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    if date is None:
        date = yields.index[-1]
    
    curve_data = yields.loc[date]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(curve_data.index, curve_data.values, 'o-', linewidth=2)
    ax.set_xlabel('Tenor (Years)')
    ax.set_ylabel('Yield (%)')
    ax.set_title(f"{title} - {date}")
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_forecast(
    actual: pd.Series,
    predicted: pd.Series,
    title: str = "Yield Forecast",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot actual vs predicted yield values.
    
    Parameters
    ----------
    actual : pd.Series
        Actual yield values
    predicted : pd.Series
        Predicted yield values
    title : str
        Plot title
    save_path : Optional[str]
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(actual.index, actual.values, label='Actual', linewidth=2)
    ax.plot(predicted.index, predicted.values, label='Predicted', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Yield (%)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_model_performance(
    metrics: Dict[str, float],
    title: str = "Model Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model performance metrics.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary of metric names and values
    title : str
        Plot title
    save_path : Optional[str]
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values)
    ax.set_title(title)
    ax.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_feature_importance(
    importance: pd.Series,
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Parameters
    ----------
    importance : pd.Series
        Feature importance values
    title : str
        Plot title
    top_n : int
        Number of top features to show
    save_path : Optional[str]
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    # Get top N features
    top_features = importance.nlargest(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(top_features)), top_features.values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.
    
    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    title : str
        Plot title
    save_path : Optional[str]
        Path to save the plot
        
    Returns
    -------
    plt.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        ax=ax
    )
    
    ax.set_title(title)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig 