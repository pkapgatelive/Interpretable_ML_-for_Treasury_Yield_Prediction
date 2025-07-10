#!/usr/bin/env python3
"""
Feature Engineering Script for Yield Curve Forecasting

This script processes raw Treasury yield and macro data to create
features for machine learning models.

Features created:
- Yield curve slope, curvature, level
- PCA components
- Lagged macro variables
- Technical indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """Load raw data files."""
    print("📂 Loading raw data files...")
    
    # Find the latest files
    raw_files = os.listdir('data/raw/')
    
    # Load yield data
    yield_files = [f for f in raw_files if f.startswith('treasury_yields_')]
    if not yield_files:
        raise FileNotFoundError("No Treasury yield files found in data/raw/")
    
    latest_yield_file = sorted(yield_files)[-1]
    yield_data = pd.read_csv(f'data/raw/{latest_yield_file}', index_col=0, parse_dates=True)
    print(f"✅ Loaded yield data: {yield_data.shape}")
    
    # Load macro data
    macro_files = [f for f in raw_files if f.startswith('macro_indicators_')]
    if not macro_files:
        raise FileNotFoundError("No macro indicator files found in data/raw/")
    
    latest_macro_file = sorted(macro_files)[-1]
    macro_data = pd.read_csv(f'data/raw/{latest_macro_file}', index_col=0, parse_dates=True)
    print(f"✅ Loaded macro data: {macro_data.shape}")
    
    return yield_data, macro_data

def compute_yield_curve_features(yield_data):
    """Compute yield curve features: level, slope, curvature."""
    print("📈 Computing yield curve features...")
    
    features = pd.DataFrame(index=yield_data.index)
    
    # Convert column names to numeric for calculations
    tenor_mapping = {
        '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, '3Y': 3,
        '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
    }
    
    # Level (average yield across curve)
    features['level'] = yield_data.mean(axis=1)
    
    # Slope (10Y - 2Y spread)
    if '10Y' in yield_data.columns and '2Y' in yield_data.columns:
        features['slope'] = yield_data['10Y'] - yield_data['2Y']
    
    # Curvature (2 * 5Y - 2Y - 10Y)
    if all(col in yield_data.columns for col in ['2Y', '5Y', '10Y']):
        features['curvature'] = 2 * yield_data['5Y'] - yield_data['2Y'] - yield_data['10Y']
    
    # Short-end slope (2Y - 3M)
    if '2Y' in yield_data.columns and '3M' in yield_data.columns:
        features['short_slope'] = yield_data['2Y'] - yield_data['3M']
    
    # Long-end slope (30Y - 10Y)
    if '30Y' in yield_data.columns and '10Y' in yield_data.columns:
        features['long_slope'] = yield_data['30Y'] - yield_data['10Y']
    
    # Butterfly (5Y - 0.5 * (2Y + 10Y))
    if all(col in yield_data.columns for col in ['2Y', '5Y', '10Y']):
        features['butterfly'] = yield_data['5Y'] - 0.5 * (yield_data['2Y'] + yield_data['10Y'])
    
    print(f"✅ Created {features.shape[1]} yield curve features")
    return features

def compute_pca_features(yield_data, n_components=3):
    """Compute PCA components of yield curves."""
    print(f"🔍 Computing PCA features ({n_components} components)...")
    
    # Standardize yields
    scaler = StandardScaler()
    yield_scaled = scaler.fit_transform(yield_data.fillna(method='ffill'))
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(yield_scaled)
    
    # Create DataFrame
    pca_features = pd.DataFrame(
        pca_components,
        index=yield_data.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Print explained variance
    explained_var = pca.explained_variance_ratio_
    print(f"✅ PCA explained variance: {explained_var}")
    
    return pca_features

def compute_technical_indicators(yield_data):
    """Compute technical indicators for yield curves."""
    print("📊 Computing technical indicators...")
    
    features = pd.DataFrame(index=yield_data.index)
    
    # Use 10Y yield as reference
    if '10Y' not in yield_data.columns:
        reference_yield = yield_data.iloc[:, -1]  # Use last column
        ref_name = yield_data.columns[-1]
    else:
        reference_yield = yield_data['10Y']
        ref_name = '10Y'
    
    # Moving averages
    for window in [5, 20, 60]:
        features[f'ma_{window}'] = reference_yield.rolling(window=window).mean()
    
    # Volatility (rolling standard deviation)
    features['volatility_20'] = reference_yield.rolling(window=20).std()
    
    # RSI (simplified)
    delta = reference_yield.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Momentum
    features['momentum_5'] = reference_yield - reference_yield.shift(5)
    features['momentum_20'] = reference_yield - reference_yield.shift(20)
    
    print(f"✅ Created {features.shape[1]} technical indicators")
    return features

def prepare_macro_features(macro_data, yield_data):
    """Prepare macro features with appropriate lags."""
    print("🏛️ Preparing macro features...")
    
    # Interpolate macro data to daily frequency to match yield data
    macro_daily = macro_data.reindex(yield_data.index, method='ffill')
    
    features = pd.DataFrame(index=yield_data.index)
    
    # Add current and lagged macro variables
    for col in macro_data.columns:
        # Current values
        features[f'{col}_current'] = macro_daily[col]
        
        # Lagged values (1, 3, 6 months)
        for lag in [20, 60, 120]:  # Approximately 1, 3, 6 months in business days
            features[f'{col}_lag_{lag}'] = macro_daily[col].shift(lag)
    
    # Compute macro changes
    for col in macro_data.columns:
        features[f'{col}_change_1m'] = macro_daily[col] - macro_daily[col].shift(20)
        features[f'{col}_change_3m'] = macro_daily[col] - macro_daily[col].shift(60)
    
    print(f"✅ Created {features.shape[1]} macro features")
    return features

def create_target_variables(yield_data, horizons=[1, 5, 10, 20]):
    """Create target variables for different forecast horizons."""
    print(f"🎯 Creating target variables for horizons: {horizons}")
    
    targets = pd.DataFrame(index=yield_data.index)
    
    # For each yield tenor and horizon, create forward-looking targets
    for tenor in yield_data.columns:
        for horizon in horizons:
            # Future yield levels
            targets[f'{tenor}_future_{horizon}d'] = yield_data[tenor].shift(-horizon)
            
            # Future yield changes
            targets[f'{tenor}_change_{horizon}d'] = (
                yield_data[tenor].shift(-horizon) - yield_data[tenor]
            )
    
    print(f"✅ Created {targets.shape[1]} target variables")
    return targets

def create_lagged_features(yield_data, max_lags=5):
    """Create lagged features of yield data."""
    print(f"⏰ Creating lagged yield features (max_lags={max_lags})...")
    
    features = pd.DataFrame(index=yield_data.index)
    
    for tenor in yield_data.columns:
        for lag in range(1, max_lags + 1):
            features[f'{tenor}_lag_{lag}'] = yield_data[tenor].shift(lag)
    
    print(f"✅ Created {features.shape[1]} lagged features")
    return features

def clean_and_align_data(features_list, targets):
    """Clean and align all feature datasets."""
    print("🧹 Cleaning and aligning data...")
    
    # Combine all features
    all_features = pd.concat(features_list, axis=1)
    
    # Find common dates (intersection of all datasets)
    common_dates = all_features.index.intersection(targets.index)
    
    # Filter to common dates
    X = all_features.loc[common_dates]
    y = targets.loc[common_dates]
    
    # Remove rows with any NaN values
    before_shape = X.shape
    combined = pd.concat([X, y], axis=1)
    combined_clean = combined.dropna()
    
    # Split back
    X_clean = combined_clean[X.columns]
    y_clean = combined_clean[y.columns]
    
    print(f"✅ Data shape before cleaning: {before_shape}")
    print(f"✅ Data shape after cleaning: {X_clean.shape}")
    print(f"✅ Removed {before_shape[0] - X_clean.shape[0]} rows with missing values")
    
    return X_clean, y_clean

def main():
    """Main feature engineering pipeline."""
    print("🚀 Starting Feature Engineering Pipeline\n")
    
    # Create output directory
    os.makedirs('data/processed', exist_ok=True)
    
    try:
        # 1. Load raw data
        yield_data, macro_data = load_raw_data()
        
        # 2. Create various feature sets
        features_list = []
        
        # Yield curve features
        yield_features = compute_yield_curve_features(yield_data)
        features_list.append(yield_features)
        
        # PCA features
        pca_features = compute_pca_features(yield_data, n_components=3)
        features_list.append(pca_features)
        
        # Technical indicators
        tech_features = compute_technical_indicators(yield_data)
        features_list.append(tech_features)
        
        # Macro features
        macro_features = prepare_macro_features(macro_data, yield_data)
        features_list.append(macro_features)
        
        # Lagged yield features
        lagged_features = create_lagged_features(yield_data, max_lags=5)
        features_list.append(lagged_features)
        
        # 3. Create target variables
        targets = create_target_variables(yield_data, horizons=[1, 5, 10, 20])
        
        # 4. Clean and align data
        X_features, y_targets = clean_and_align_data(features_list, targets)
        
        # 5. Save processed data
        features_file = 'data/processed/X_features.csv'
        targets_file = 'data/processed/Y_targets.csv'
        
        X_features.to_csv(features_file)
        y_targets.to_csv(targets_file)
        
        print(f"\n💾 Saved processed data:")
        print(f"   📁 Features: {features_file} ({X_features.shape})")
        print(f"   📁 Targets: {targets_file} ({y_targets.shape})")
        
        # Display sample of processed data
        print(f"\n📋 Sample Features (first 5 columns):")
        print(X_features.iloc[:5, :5])
        
        print(f"\n📋 Sample Targets (first 5 columns):")
        print(y_targets.iloc[:5, :5])
        
        # Feature importance info
        print(f"\n📊 Feature Summary:")
        print(f"   • Total features: {X_features.shape[1]}")
        print(f"   • Total targets: {y_targets.shape[1]}")
        print(f"   • Time period: {X_features.index[0]} to {X_features.index[-1]}")
        print(f"   • Number of observations: {X_features.shape[0]}")
        
        print(f"\n🎉 Feature engineering completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        raise

if __name__ == "__main__":
    main() 