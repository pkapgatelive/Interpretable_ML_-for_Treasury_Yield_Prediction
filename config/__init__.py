"""
Configuration package for yield curve forecasting project.

This package contains all configuration files and utilities for managing
project parameters, model settings, and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def load_model_config(config_path: str = "config/model_config.yaml") -> Dict[str, Any]:
    """
    Load model configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the model configuration file
        
    Returns
    -------
    Dict[str, Any]
        Model configuration dictionary
    """
    return load_config(config_path)

__all__ = ["load_config", "load_model_config"] 