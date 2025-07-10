#!/usr/bin/env python3
"""
Machine Learning Model Training for Yield Curve Forecasting

This script trains multiple ML models with hyperparameter optimization
and proper time series cross-validation.

Models trained:
- Linear Regression (LASSO, Ridge, ElasticNet)
- Random Forest
- XGBoost
- LightGBM
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Optimization
import optuna

# Tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

def load_processed_data():
    """Load processed feature and target data."""
    print("📂 Loading processed data...")
    
    X = pd.read_csv('data/processed/X_features.csv', index_col=0, parse_dates=True)
    y = pd.read_csv('data/processed/Y_targets.csv', index_col=0, parse_dates=True)
    
    print(f"✅ Loaded features: {X.shape}")
    print(f"✅ Loaded targets: {y.shape}")
    
    return X, y

def create_time_series_splits(X: pd.DataFrame, n_splits: int = 5, test_size: int = 180):
    """Create time series cross-validation splits."""
    print(f"📊 Creating {n_splits} time series CV splits...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    splits = list(tscv.split(X))
    
    print(f"✅ Created {len(splits)} splits")
    return splits

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100
    }

def train_linear_models(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                       X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """Train linear regression models with different regularization."""
    print("🔵 Training Linear Models...")
    
    results = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train on first target column (simplify for demo)
    target_col = y_train.columns[0]
    y_train_target = y_train[target_col].values
    y_test_target = y_test[target_col].values
    
    models = {
        'lasso': Lasso(alpha=0.01, random_state=42),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
    }
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train_target)
        y_pred = model.predict(X_test_scaled)
        
        metrics = calculate_metrics(y_test_target, y_pred)
        results[name] = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'predictions': y_pred
        }
        
        print(f"    {name} RMSE: {metrics['rmse']:.4f}")
    
    return results

def optimize_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                          splits: List, n_trials: int = 20) -> Dict[str, Any]:
    """Optimize Random Forest with Optuna."""
    print("🟢 Optimizing Random Forest...")
    
    target_col = y_train.columns[0]
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42
        }
        
        cv_scores = []
        for train_idx, val_idx in splits:
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train[target_col].iloc[train_idx]
            y_fold_val = y_train[target_col].iloc[val_idx]
            
            model = RandomForestRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"  Best RF RMSE: {study.best_value:.4f}")
    return study.best_params

def optimize_xgboost(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                    splits: List, n_trials: int = 20) -> Dict[str, Any]:
    """Optimize XGBoost with Optuna."""
    print("🟠 Optimizing XGBoost...")
    
    target_col = y_train.columns[0]
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        
        cv_scores = []
        for train_idx, val_idx in splits:
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train[target_col].iloc[train_idx]
            y_fold_val = y_train[target_col].iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_fold_train, y_fold_train, verbose=False)
            y_pred = model.predict(X_fold_val)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"  Best XGB RMSE: {study.best_value:.4f}")
    return study.best_params

def optimize_lightgbm(X_train: pd.DataFrame, y_train: pd.DataFrame, 
                     splits: List, n_trials: int = 20) -> Dict[str, Any]:
    """Optimize LightGBM with Optuna."""
    print("🟡 Optimizing LightGBM...")
    
    target_col = y_train.columns[0]
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        cv_scores = []
        for train_idx, val_idx in splits:
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train[target_col].iloc[train_idx]
            y_fold_val = y_train[target_col].iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"  Best LGB RMSE: {study.best_value:.4f}")
    return study.best_params

def train_final_models(X_train: pd.DataFrame, y_train: pd.DataFrame,
                      X_test: pd.DataFrame, y_test: pd.DataFrame,
                      best_params: Dict[str, Dict]) -> Dict[str, Any]:
    """Train final models with best parameters."""
    print("🎯 Training final models with optimized parameters...")
    
    results = {}
    target_col = y_train.columns[0]
    y_train_target = y_train[target_col]
    y_test_target = y_test[target_col]
    
    # Random Forest
    print("  Training optimized Random Forest...")
    rf_model = RandomForestRegressor(**best_params['random_forest'])
    rf_model.fit(X_train, y_train_target)
    rf_pred = rf_model.predict(X_test)
    
    results['random_forest'] = {
        'model': rf_model,
        'metrics': calculate_metrics(y_test_target, rf_pred),
        'predictions': rf_pred,
        'params': best_params['random_forest']
    }
    
    # XGBoost
    print("  Training optimized XGBoost...")
    xgb_model = xgb.XGBRegressor(**best_params['xgboost'])
    xgb_model.fit(X_train, y_train_target, verbose=False)
    xgb_pred = xgb_model.predict(X_test)
    
    results['xgboost'] = {
        'model': xgb_model,
        'metrics': calculate_metrics(y_test_target, xgb_pred),
        'predictions': xgb_pred,
        'params': best_params['xgboost']
    }
    
    # LightGBM
    print("  Training optimized LightGBM...")
    lgb_model = lgb.LGBMRegressor(**best_params['lightgbm'])
    lgb_model.fit(X_train, y_train_target)
    lgb_pred = lgb_model.predict(X_test)
    
    results['lightgbm'] = {
        'model': lgb_model,
        'metrics': calculate_metrics(y_test_target, lgb_pred),
        'predictions': lgb_pred,
        'params': best_params['lightgbm']
    }
    
    return results

def save_models_and_results(linear_results: Dict, tree_results: Dict, 
                          metrics_summary: Dict):
    """Save trained models and results."""
    print("💾 Saving models and results...")
    
    # Create directories
    os.makedirs('models/trained', exist_ok=True)
    os.makedirs('reports/model_metrics', exist_ok=True)
    
    # Save models
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Linear models
    for name, result in linear_results.items():
        model_path = f'models/trained/{name}_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Saved {name} model: {model_path}")
    
    # Tree models
    for name, result in tree_results.items():
        model_path = f'models/trained/{name}_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"  Saved {name} model: {model_path}")
    
    # Save metrics
    metrics_file = f'reports/model_metrics/metrics_summary_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    print(f"  Saved metrics: {metrics_file}")
    
    # Save as CSV for easy viewing
    metrics_df = pd.DataFrame.from_dict(
        {model: metrics for model, metrics in metrics_summary.items()}, 
        orient='index'
    )
    metrics_csv = f'reports/model_metrics/metrics_summary_{timestamp}.csv'
    metrics_df.to_csv(metrics_csv)
    print(f"  Saved metrics CSV: {metrics_csv}")

def main():
    """Main training pipeline."""
    print("🚀 Starting ML Model Training Pipeline\n")
    
    try:
        # Set MLflow experiment
        mlflow.set_experiment("yield_curve_forecasting")
        
        # 1. Load data
        X, y = load_processed_data()
        
        # 2. Split data (80/20 train/test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Data splits:")
        print(f"  Training: {X_train.shape}")
        print(f"  Testing: {X_test.shape}")
        
        # 3. Create CV splits for hyperparameter optimization
        cv_splits = create_time_series_splits(X_train, n_splits=3, test_size=60)
        
        # 4. Train linear models (baseline)
        linear_results = train_linear_models(X_train, y_train, X_test, y_test)
        
        # 5. Optimize tree-based models
        print("\n🔧 Hyperparameter Optimization...")
        best_params = {}
        
        best_params['random_forest'] = optimize_random_forest(X_train, y_train, cv_splits, n_trials=10)
        best_params['xgboost'] = optimize_xgboost(X_train, y_train, cv_splits, n_trials=10)
        best_params['lightgbm'] = optimize_lightgbm(X_train, y_train, cv_splits, n_trials=10)
        
        # 6. Train final models
        tree_results = train_final_models(X_train, y_train, X_test, y_test, best_params)
        
        # 7. Compile results
        all_results = {**linear_results, **tree_results}
        
        metrics_summary = {}
        for model_name, result in all_results.items():
            metrics_summary[model_name] = result['metrics']
        
        # 8. Display results
        print(f"\n📊 Model Performance Summary:")
        print("=" * 60)
        results_df = pd.DataFrame.from_dict(metrics_summary, orient='index')
        print(results_df.round(4))
        
        # Find best model
        best_model = results_df['rmse'].idxmin()
        best_rmse = results_df.loc[best_model, 'rmse']
        print(f"\n🏆 Best Model: {best_model} (RMSE: {best_rmse:.4f})")
        
        # 9. Save everything
        save_models_and_results(linear_results, tree_results, metrics_summary)
        
        print(f"\n🎉 Model training completed successfully!")
        print(f"📁 Models saved to: models/trained/")
        print(f"📁 Metrics saved to: reports/model_metrics/")
        
        return all_results, metrics_summary
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        raise

if __name__ == "__main__":
    results, metrics = main() 