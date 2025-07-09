#!/usr/bin/env python3
"""
Model training script for yield curve forecasting project.

This script trains various ML models for yield curve forecasting with
proper experiment tracking, cross-validation, and model persistence.

Usage:
    python scripts/train_model.py --model random_forest --config config/config.yaml
    python scripts/train_model.py --model lstm --epochs 100 --batch_size 32
    python scripts/train_model.py --all-models --cv 5
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml
import mlflow
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import YieldCurveDataLoader, MacroDataLoader
from src.data.feature_engineering import YieldCurveFeatureEngineer
from src.models.baseline import LinearRegressionModel, ARIMAModel
from src.models.ml_models import RandomForestModel, XGBoostModel, LSTMModel
from src.models.ensemble import VotingEnsemble, StackingEnsemble
from src.models.evaluation import ModelEvaluator, cross_validate_model
from src.utils.helpers import setup_logging, save_model, create_directory
from src.utils.constants import MODEL_TYPES, RANDOM_SEEDS
from config import load_config, load_model_config

# Setup logging
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train yield curve forecasting models"
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["linear", "arima", "random_forest", "xgboost", "lstm", "ensemble"],
        help="Model type to train"
    )
    
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train all available models"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model-config", 
        type=str,
        default="config/model_config.yaml",
        help="Path to model configuration file"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (for neural networks)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    
    # Experiment tracking
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name"
    )
    
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for experiment tracking"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/trained",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save model predictions to file"
    )
    
    return parser.parse_args()


def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare training data.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Features and target data
    """
    logger.info("Loading training data...")
    
    # TODO: Implement actual data loading
    # This is a placeholder implementation
    
    # Load yield curve data
    yield_loader = YieldCurveDataLoader(
        source=config["data"]["sources"]["fred"]["api_key"] and "fred" or "mock",
        start_date=config["data"]["date_range"]["start_date"],
        end_date=config["data"]["date_range"]["end_date"]
    )
    
    yield_data = yield_loader.load_data(config["data"]["tenors"])
    
    # Load macro data
    macro_loader = MacroDataLoader()
    macro_data = macro_loader.load_indicators(
        config["features"]["macro_features"]
    )
    
    # Feature engineering
    feature_engineer = YieldCurveFeatureEngineer()
    features = feature_engineer.create_features(
        yield_data=yield_data,
        macro_data=macro_data,
        config=config["features"]
    )
    
    # Create target variable (next day yields)
    target = yield_data.shift(-1).dropna()
    
    # Align features and target
    common_dates = features.index.intersection(target.index)
    features = features.loc[common_dates]
    target = target.loc[common_dates]
    
    logger.info(f"Loaded data: {features.shape[0]} samples, {features.shape[1]} features")
    
    return features, target


def create_model(
    model_type: str,
    model_config: Dict[str, Any],
    **kwargs
) -> Any:
    """
    Create model instance based on type and configuration.
    
    Parameters
    ----------
    model_type : str
        Type of model to create
    model_config : Dict[str, Any]
        Model configuration parameters
    **kwargs
        Additional model parameters
        
    Returns
    -------
    Any
        Model instance
    """
    if model_type == "linear":
        return LinearRegressionModel(**model_config.get("linear_regression", {}))
    
    elif model_type == "arima":
        return ARIMAModel(**model_config.get("arima", {}))
    
    elif model_type == "random_forest":
        params = model_config.get("random_forest", {})
        params.update(kwargs)
        return RandomForestModel(**params)
    
    elif model_type == "xgboost":
        params = model_config.get("xgboost", {})
        params.update(kwargs)
        return XGBoostModel(**params)
    
    elif model_type == "lstm":
        params = model_config.get("lstm", {})
        params.update(kwargs)
        return LSTMModel(**params)
    
    elif model_type == "ensemble":
        # Create ensemble of best performing models
        base_models = [
            RandomForestModel(**model_config.get("random_forest", {})),
            XGBoostModel(**model_config.get("xgboost", {})),
            LSTMModel(**model_config.get("lstm", {}))
        ]
        return VotingEnsemble(estimators=base_models)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_single_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    config: Dict[str, Any],
    model_config: Dict[str, Any],
    args: argparse.Namespace
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a single model and evaluate performance.
    
    Parameters
    ----------
    model_type : str
        Type of model to train
    X_train, y_train : pd.DataFrame
        Training data
    X_val, y_val : pd.DataFrame
        Validation data
    config : Dict[str, Any]
        General configuration
    model_config : Dict[str, Any]
        Model-specific configuration
    args : argparse.Namespace
        Command line arguments
        
    Returns
    -------
    Tuple[Any, Dict[str, float]]
        Trained model and validation metrics
    """
    logger.info(f"Training {model_type} model...")
    
    # Create model
    model_params = {}
    if args.epochs and model_type in ["lstm"]:
        model_params["epochs"] = args.epochs
    if args.batch_size and model_type in ["lstm"]:
        model_params["batch_size"] = args.batch_size
        
    model = create_model(model_type, model_config, **model_params)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "model_type": model_type,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
            **model_params
        })
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Calculate metrics
        evaluator = ModelEvaluator()
        train_metrics = evaluator.calculate_metrics(y_train, y_pred_train)
        val_metrics = evaluator.calculate_metrics(y_val, y_pred_val)
        
        # Log metrics
        for metric, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric}", value)
        for metric, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric}", value)
            
        mlflow.log_metric("training_time", training_time)
        
        # Save model
        model_path = Path(args.output_dir) / f"{model_type}_model.pkl"
        create_directory(model_path.parent)
        
        metadata = {
            "model_type": model_type,
            "training_time": training_time,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "model_config": model_config
        }
        
        save_model(model, model_path, metadata=metadata)
        mlflow.log_artifact(str(model_path))
        
        logger.info(f"Model {model_type} trained. Val RMSE: {val_metrics.get('rmse', 'N/A'):.4f}")
        
    return model, val_metrics


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting model training...")
    
    # Load configurations
    config = load_config(args.config)
    model_config = load_model_config(args.model_config)
    
    # Setup experiment tracking
    if args.use_wandb:
        wandb.init(
            project=config["experiment"]["wandb"]["project"],
            entity=config["experiment"]["wandb"]["entity"],
            config={**config, **model_config}
        )
    
    # Setup MLflow
    experiment_name = args.experiment_name or config["experiment"]["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    # Load and prepare data
    X, y = load_data(config)
    
    # Split data
    from src.utils.helpers import split_time_series
    
    train_ratio = config["modeling"]["split"]["train_ratio"]
    val_ratio = config["modeling"]["split"]["validation_ratio"]
    test_ratio = config["modeling"]["split"]["test_ratio"]
    
    X_train, X_val, X_test = split_time_series(X, train_ratio, val_ratio, test_ratio)
    y_train, y_val, y_test = split_time_series(y, train_ratio, val_ratio, test_ratio)
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train models
    results = {}
    
    if args.all_models:
        models_to_train = ["linear", "random_forest", "xgboost", "lstm", "ensemble"]
    else:
        models_to_train = [args.model] if args.model else ["random_forest"]
    
    for model_type in models_to_train:
        try:
            model, metrics = train_single_model(
                model_type, X_train, y_train, X_val, y_val,
                config, model_config, args
            )
            results[model_type] = {"model": model, "metrics": metrics}
            
        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            continue
    
    # Cross-validation for best model
    if results:
        best_model_type = min(results.keys(), key=lambda k: results[k]["metrics"].get("rmse", float("inf")))
        logger.info(f"Best model: {best_model_type}")
        
        # Perform cross-validation
        best_model = create_model(best_model_type, model_config)
        cv_scores = cross_validate_model(
            best_model, X_train, y_train, cv=args.cv
        )
        
        logger.info(f"Cross-validation RMSE: {cv_scores['rmse'].mean():.4f} ± {cv_scores['rmse'].std():.4f}")
    
    # Save results summary
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "models_trained": list(results.keys()),
        "best_model": best_model_type if results else None,
        "performance": {k: v["metrics"] for k, v in results.items()}
    }
    
    results_path = Path(args.output_dir) / "training_results.yaml"
    with open(results_path, "w") as f:
        yaml.dump(results_summary, f, default_flow_style=False)
    
    logger.info("Training completed successfully!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 