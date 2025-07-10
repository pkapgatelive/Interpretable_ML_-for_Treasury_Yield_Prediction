#!/usr/bin/env python3
"""
Simplified ML Model Training for Yield Curve Forecasting

This script trains multiple ML models without XGBoost to avoid macOS OpenMP issues.

Models trained:
- Linear Regression (LASSO, Ridge, ElasticNet)
- Random Forest
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
import lightgbm as lgb

# Optimization
import optuna

def load_processed_data():
    """Load processed feature and target data."""
    print("📂 Loading processed data...")
    
    X = pd.read_csv('data/processed/X_features.csv', index_col=0, parse_dates=True)
    y = pd.read_csv('data/processed/Y_targets.csv', index_col=0, parse_dates=True)
    
    print(f"✅ Loaded features: {X.shape}")
    print(f"✅ Loaded targets: {y.shape}")
    
    return X, y

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
    
    # Train on first target column (3M yield 1-day future)
    target_col = y_train.columns[0]
    y_train_target = y_train[target_col].values
    y_test_target = y_test[target_col].values
    
    print(f"  Target variable: {target_col}")
    
    models = {
        'lasso': Lasso(alpha=0.01, random_state=42, max_iter=2000),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=2000)
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

def train_random_forest(X_train: pd.DataFrame, y_train: pd.DataFrame,
                       X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """Train Random Forest with basic optimization."""
    print("🟢 Training Random Forest...")
    
    target_col = y_train.columns[0]
    y_train_target = y_train[target_col]
    y_test_target = y_test[target_col]
    
    # Simple parameter grid
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train_target)
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test_target, y_pred)
    
    print(f"  Random Forest RMSE: {metrics['rmse']:.4f}")
    
    return {
        'random_forest': {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'params': params
        }
    }

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.DataFrame,
                  X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """Train LightGBM model."""
    print("🟡 Training LightGBM...")
    
    target_col = y_train.columns[0]
    y_train_target = y_train[target_col]
    y_test_target = y_test[target_col]
    
    # Simple parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train_target)
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test_target, y_pred)
    
    print(f"  LightGBM RMSE: {metrics['rmse']:.4f}")
    
    return {
        'lightgbm': {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'params': params
        }
    }

def create_ensemble_model(models_dict: Dict[str, Any], 
                         X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """Create simple ensemble (average) of all models."""
    print("🎯 Creating Ensemble Model...")
    
    target_col = y_test.columns[0]
    y_test_target = y_test[target_col]
    
    # Get predictions from all models
    predictions = []
    for model_name, result in models_dict.items():
        if 'predictions' in result:
            predictions.append(result['predictions'])
    
    if len(predictions) > 0:
        # Simple average ensemble
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_metrics = calculate_metrics(y_test_target, ensemble_pred)
        
        print(f"  Ensemble RMSE: {ensemble_metrics['rmse']:.4f}")
        
        return {
            'ensemble': {
                'predictions': ensemble_pred,
                'metrics': ensemble_metrics,
                'components': list(models_dict.keys())
            }
        }
    else:
        return {}

def save_models_and_results(all_results: Dict, metrics_summary: Dict):
    """Save trained models and results."""
    print("💾 Saving models and results...")
    
    # Create directories
    os.makedirs('models/trained', exist_ok=True)
    os.makedirs('reports/model_metrics', exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save individual models
    for model_name, result in all_results.items():
        if model_name != 'ensemble' and 'model' in result:
            model_path = f'models/trained/{model_name}_{timestamp}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"  Saved {model_name} model: {model_path}")
    
    # Save metrics
    metrics_file = f'reports/model_metrics/metrics_summary_{timestamp}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2, default=str)
    print(f"  Saved metrics: {metrics_file}")
    
    # Save as CSV
    metrics_df = pd.DataFrame.from_dict(metrics_summary, orient='index')
    metrics_csv = f'reports/model_metrics/metrics_summary_{timestamp}.csv'
    metrics_df.to_csv(metrics_csv)
    print(f"  Saved metrics CSV: {metrics_csv}")

def create_prediction_plots(all_results: Dict, y_test: pd.DataFrame):
    """Create prediction vs actual plots."""
    print("📈 Creating prediction plots...")
    
    import matplotlib.pyplot as plt
    
    target_col = y_test.columns[0]
    y_actual = y_test[target_col]
    
    # Create plots directory
    os.makedirs('reports/figures', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    plot_idx = 0
    for model_name, result in all_results.items():
        if plot_idx >= 4:
            break
            
        if 'predictions' in result:
            ax = axes[plot_idx]
            y_pred = result['predictions']
            
            ax.scatter(y_actual, y_pred, alpha=0.6)
            ax.plot([y_actual.min(), y_actual.max()], 
                   [y_actual.min(), y_actual.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name.title()} - RMSE: {result["metrics"]["rmse"]:.4f}')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Remove empty subplots
    for i in range(plot_idx, 4):
        axes[i].remove()
    
    plt.tight_layout()
    plot_file = f'reports/figures/prediction_plots_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots: {plot_file}")

def main():
    """Main training pipeline."""
    print("🚀 Starting Simplified ML Model Training Pipeline\n")
    
    try:
        # 1. Load data
        X, y = load_processed_data()
        
        # 2. Split data (80/20 train/test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"📊 Data splits:")
        print(f"  Training: {X_train.shape}")
        print(f"  Testing: {X_test.shape}")
        print(f"  Target: {y_train.columns[0]}")
        
        # 3. Train models
        print("\n🔧 Training Models...")
        
        linear_results = train_linear_models(X_train, y_train, X_test, y_test)
        rf_results = train_random_forest(X_train, y_train, X_test, y_test)
        lgb_results = train_lightgbm(X_train, y_train, X_test, y_test)
        
        # 4. Combine results
        all_results = {**linear_results, **rf_results, **lgb_results}
        
        # 5. Create ensemble
        ensemble_results = create_ensemble_model(all_results, X_test, y_test)
        all_results.update(ensemble_results)
        
        # 6. Compile metrics
        metrics_summary = {}
        for model_name, result in all_results.items():
            metrics_summary[model_name] = result['metrics']
        
        # 7. Display results
        print(f"\n📊 Model Performance Summary:")
        print("=" * 80)
        results_df = pd.DataFrame.from_dict(metrics_summary, orient='index')
        print(results_df.round(4))
        
        # Find best model
        best_model = results_df['rmse'].idxmin()
        best_rmse = results_df.loc[best_model, 'rmse']
        print(f"\n🏆 Best Model: {best_model} (RMSE: {best_rmse:.4f})")
        
        # 8. Save everything
        save_models_and_results(all_results, metrics_summary)
        
        # 9. Create plots
        create_prediction_plots(all_results, y_test)
        
        print(f"\n🎉 Model training completed successfully!")
        print(f"📁 Models saved to: models/trained/")
        print(f"📁 Metrics saved to: reports/model_metrics/")
        print(f"📁 Plots saved to: reports/figures/")
        
        return all_results, metrics_summary
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        raise

if __name__ == "__main__":
    results, metrics = main() 