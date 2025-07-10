#!/usr/bin/env python3
"""
Scikit-learn ML Model Training for Yield Curve Forecasting

This script trains multiple ML models using only scikit-learn to avoid
dependency issues on macOS.

Models trained:
- Linear Regression (LASSO, Ridge, ElasticNet)
- Random Forest
- Gradient Boosting (scikit-learn)
- Support Vector Regression
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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def train_tree_models(X_train: pd.DataFrame, y_train: pd.DataFrame,
                     X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """Train tree-based models."""
    print("🌳 Training Tree-based Models...")
    
    results = {}
    target_col = y_train.columns[0]
    y_train_target = y_train[target_col]
    y_test_target = y_test[target_col]
    
    # Random Forest
    print("  Training Random Forest...")
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X_train, y_train_target)
    rf_pred = rf_model.predict(X_test)
    
    results['random_forest'] = {
        'model': rf_model,
        'metrics': calculate_metrics(y_test_target, rf_pred),
        'predictions': rf_pred,
        'params': rf_params
    }
    
    print(f"    Random Forest RMSE: {results['random_forest']['metrics']['rmse']:.4f}")
    
    # Gradient Boosting
    print("  Training Gradient Boosting...")
    gb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42
    }
    
    gb_model = GradientBoostingRegressor(**gb_params)
    gb_model.fit(X_train, y_train_target)
    gb_pred = gb_model.predict(X_test)
    
    results['gradient_boosting'] = {
        'model': gb_model,
        'metrics': calculate_metrics(y_test_target, gb_pred),
        'predictions': gb_pred,
        'params': gb_params
    }
    
    print(f"    Gradient Boosting RMSE: {results['gradient_boosting']['metrics']['rmse']:.4f}")
    
    return results

def train_svm_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, Any]:
    """Train Support Vector Regression model."""
    print("🎯 Training Support Vector Regression...")
    
    target_col = y_train.columns[0]
    y_train_target = y_train[target_col]
    y_test_target = y_test[target_col]
    
    # Standardize features for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use a subset of data for SVM (it's slow on large datasets)
    sample_size = min(1000, len(X_train_scaled))
    indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
    
    X_train_sample = X_train_scaled[indices]
    y_train_sample = y_train_target.iloc[indices]
    
    svm_params = {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'epsilon': 0.01
    }
    
    svm_model = SVR(**svm_params)
    svm_model.fit(X_train_sample, y_train_sample)
    svm_pred = svm_model.predict(X_test_scaled)
    
    results = {
        'svr': {
            'model': svm_model,
            'scaler': scaler,
            'metrics': calculate_metrics(y_test_target, svm_pred),
            'predictions': svm_pred,
            'params': svm_params
        }
    }
    
    print(f"    SVR RMSE: {results['svr']['metrics']['rmse']:.4f}")
    
    return results

def create_ensemble_model(models_dict: Dict[str, Any], 
                         y_test: pd.DataFrame) -> Dict[str, Any]:
    """Create simple ensemble (average) of all models."""
    print("🎯 Creating Ensemble Model...")
    
    target_col = y_test.columns[0]
    y_test_target = y_test[target_col]
    
    # Get predictions from all models
    predictions = []
    model_names = []
    for model_name, result in models_dict.items():
        if 'predictions' in result:
            predictions.append(result['predictions'])
            model_names.append(model_name)
    
    if len(predictions) > 0:
        # Simple average ensemble
        ensemble_pred = np.mean(predictions, axis=0)
        ensemble_metrics = calculate_metrics(y_test_target, ensemble_pred)
        
        # Weighted ensemble (weight by inverse RMSE)
        weights = []
        for model_name, result in models_dict.items():
            if 'predictions' in result:
                rmse = result['metrics']['rmse']
                weights.append(1.0 / rmse)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_pred = np.average(predictions, axis=0, weights=weights)
        weighted_metrics = calculate_metrics(y_test_target, weighted_pred)
        
        print(f"  Simple Ensemble RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"  Weighted Ensemble RMSE: {weighted_metrics['rmse']:.4f}")
        
        return {
            'ensemble_simple': {
                'predictions': ensemble_pred,
                'metrics': ensemble_metrics,
                'components': model_names
            },
            'ensemble_weighted': {
                'predictions': weighted_pred,
                'metrics': weighted_metrics,
                'components': model_names,
                'weights': weights.tolist()
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
        if not model_name.startswith('ensemble') and 'model' in result:
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
    
    # Determine plot layout
    n_models = len([k for k in all_results.keys() if 'predictions' in all_results[k]])
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    plot_idx = 0
    for model_name, result in all_results.items():
        if 'predictions' in result and plot_idx < len(axes):
            ax = axes[plot_idx]
            y_pred = result['predictions']
            
            ax.scatter(y_actual, y_pred, alpha=0.6, s=20)
            ax.plot([y_actual.min(), y_actual.max()], 
                   [y_actual.min(), y_actual.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{model_name.replace("_", " ").title()}\nRMSE: {result["metrics"]["rmse"]:.4f}')
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Remove empty subplots
    for i in range(plot_idx, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plot_file = f'reports/figures/prediction_plots_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots: {plot_file}")

def create_feature_importance_plot(rf_model, feature_names):
    """Create feature importance plot from Random Forest."""
    print("📊 Creating feature importance plot...")
    
    import matplotlib.pyplot as plt
    
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # Top 20 features
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(top_n), importance[top_indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in top_indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Most Important Features (Random Forest)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = f'reports/figures/feature_importance_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved feature importance plot: {plot_file}")

def main():
    """Main training pipeline."""
    print("🚀 Starting Scikit-learn ML Model Training Pipeline\n")
    
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
        tree_results = train_tree_models(X_train, y_train, X_test, y_test)
        svm_results = train_svm_model(X_train, y_train, X_test, y_test)
        
        # 4. Combine results
        all_results = {**linear_results, **tree_results, **svm_results}
        
        # 5. Create ensembles
        ensemble_results = create_ensemble_model(all_results, y_test)
        all_results.update(ensemble_results)
        
        # 6. Compile metrics
        metrics_summary = {}
        for model_name, result in all_results.items():
            if 'metrics' in result:
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
        
        # 10. Feature importance
        if 'random_forest' in all_results:
            create_feature_importance_plot(
                all_results['random_forest']['model'], 
                X.columns.tolist()
            )
        
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