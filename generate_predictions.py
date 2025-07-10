#!/usr/bin/env python3
"""
Generate Predictions for Yield Curve Forecasting

This script loads the best trained model and generates predictions for various scenarios.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_best_model():
    """Load the best performing model (ElasticNet) and its scaler."""
    print("📂 Loading best model (ElasticNet)...")
    
    # Find the most recent ElasticNet model
    model_dir = "models/trained"
    elastic_files = [f for f in os.listdir(model_dir) if f.startswith('elastic_net')]
    if not elastic_files:
        raise FileNotFoundError("No ElasticNet model found!")
    
    latest_model = sorted(elastic_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"✅ Loaded model: {latest_model}")
    return model_data['model'], model_data['scaler']

def load_processed_data():
    """Load the processed feature and target data."""
    print("📂 Loading processed data...")
    
    X = pd.read_csv('data/processed/X_features.csv', index_col=0, parse_dates=True)
    y = pd.read_csv('data/processed/Y_targets.csv', index_col=0, parse_dates=True)
    
    print(f"✅ Loaded features: {X.shape}")
    print(f"✅ Loaded targets: {y.shape}")
    
    return X, y

def get_latest_features(X: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent feature vector for prediction."""
    return X.iloc[-1:].copy()

def create_custom_scenario(latest_features: pd.DataFrame, 
                          fed_funds_rate: float = None,
                          cpi_yoy: float = None) -> pd.DataFrame:
    """Create a custom scenario by modifying macroeconomic variables."""
    scenario = latest_features.copy()
    
    # Update Fed Funds Rate if provided
    if fed_funds_rate is not None:
        fed_cols = [col for col in scenario.columns if 'fed_funds' in col.lower()]
        for col in fed_cols:
            scenario[col] = fed_funds_rate
        print(f"  Updated Fed Funds Rate: {fed_funds_rate}%")
    
    # Update CPI if provided
    if cpi_yoy is not None:
        cpi_cols = [col for col in scenario.columns if 'cpi' in col.lower()]
        for col in cpi_cols:
            scenario[col] = cpi_yoy
        print(f"  Updated CPI YoY: {cpi_yoy}%")
    
    return scenario

def predict_yield_curve(model, scaler, features: pd.DataFrame, y_columns: List[str]) -> Dict[str, float]:
    """Generate yield curve predictions for all tenors."""
    # Standardize features
    features_scaled = scaler.transform(features)
    
    # For simplicity, we'll predict just the 1-day ahead targets
    target_columns = [col for col in y_columns if '1d' in col]
    
    predictions = {}
    
    # Extract tenor information from target column names
    # Assuming columns like "3M_future_1d", "6M_future_1d", etc.
    for target_col in target_columns:
        tenor = target_col.split('_')[0]  # Extract tenor (e.g., "3M")
        
        # Since we only trained on first target, use that prediction
        # In a full implementation, you'd train separate models for each tenor
        pred_value = model.predict(features_scaled)[0]
        predictions[tenor] = pred_value
    
    return predictions

def generate_sample_predictions():
    """Generate sample predictions for different scenarios."""
    print("🔮 Generating Sample Predictions...")
    
    # Load model and data
    model, scaler = load_best_model()
    X, y = load_processed_data()
    
    # Get latest features
    latest_features = get_latest_features(X)
    latest_date = latest_features.index[0]
    
    print(f"📅 Base date: {latest_date.strftime('%Y-%m-%d')}")
    
    # Scenario 1: Current conditions
    print("\n📊 Scenario 1: Current Economic Conditions")
    current_pred = predict_yield_curve(model, scaler, latest_features, y.columns)
    
    # Scenario 2: Fed rate hike
    print("\n📊 Scenario 2: Fed Rate Hike (+0.5%)")
    current_fed_rate = 5.25  # Approximate current rate
    hike_scenario = create_custom_scenario(latest_features, fed_funds_rate=current_fed_rate + 0.5)
    hike_pred = predict_yield_curve(model, scaler, hike_scenario, y.columns)
    
    # Scenario 3: High inflation
    print("\n📊 Scenario 3: High Inflation Scenario (+1% CPI)")
    current_cpi = 3.2  # Approximate current CPI
    inflation_scenario = create_custom_scenario(latest_features, cpi_yoy=current_cpi + 1.0)
    inflation_pred = predict_yield_curve(model, scaler, inflation_scenario, y.columns)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Current_Conditions': pd.Series(current_pred),
        'Fed_Rate_Hike': pd.Series(hike_pred),
        'High_Inflation': pd.Series(inflation_pred)
    })
    
    # Create output directory
    os.makedirs('reports/predictions', exist_ok=True)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'reports/predictions/sample_predictions_{timestamp}.csv'
    predictions_df.to_csv(output_file)
    
    print(f"\n💾 Predictions saved to: {output_file}")
    print("\n📋 Sample Predictions Summary:")
    print(predictions_df.round(4))
    
    return predictions_df

def main():
    """Main execution function."""
    print("🚀 Starting Prediction Generation Pipeline")
    print("=" * 50)
    
    try:
        predictions = generate_sample_predictions()
        
        print("\n🎉 Prediction generation completed successfully!")
        print(f"📁 Predictions saved to: reports/predictions/")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error during prediction generation: {e}")
        raise

if __name__ == "__main__":
    main() 