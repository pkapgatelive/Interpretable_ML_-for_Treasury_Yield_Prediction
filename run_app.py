#!/usr/bin/env python3
"""
YieldCurveAI Application Runner
===============================
Simple script to launch the YieldCurveAI Streamlit application with proper setup.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def main():
    """Main function to run the YieldCurveAI application."""
    print("🚀 Starting YieldCurveAI Application...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("YieldCurveAI.py").exists():
        print("❌ YieldCurveAI.py not found in current directory")
        print("📁 Please run this script from the yield-curve-forecasting directory")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if required data files exist
    required_files = [
        "data/processed/X_features.csv",
        "models/trained",
        "reports/model_metrics/metrics_summary_20250710_233617.json"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Warning: Some data files are missing:")
        for file in missing_files:
            print(f"   - {file}")
        print("📊 The application may not work properly without these files.")
        
        response = input("Continue anyway? (y/n): ").lower().strip()
        if response != 'y':
            print("❌ Application startup cancelled.")
            return
    
    print("✅ All checks passed!")
    print("🌐 Launching YieldCurveAI web application...")
    print("📈 Open your browser and navigate to the URL shown below")
    print("=" * 50)
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 
            'YieldCurveAI.py',
            '--theme.base', 'light',
            '--theme.primaryColor', '#1f4e79',
            '--theme.backgroundColor', '#ffffff',
            '--theme.secondaryBackgroundColor', '#f8f9fa'
        ])
    except KeyboardInterrupt:
        print("\n👋 YieldCurveAI application stopped.")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        print("💡 Try running manually: streamlit run YieldCurveAI.py")

if __name__ == "__main__":
    main() 