#!/usr/bin/env python3
"""
Validation script for YieldCurveAI application
"""
import ast
import sys
from pathlib import Path

def validate_python_syntax(file_path):
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, "Valid Python syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_required_methods(file_path):
    """Check if required methods exist in the YieldCurveAI class."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find the YieldCurveAI class
        yield_curve_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'YieldCurveAI':
                yield_curve_class = node
                break
        
        if not yield_curve_class:
            return False, "YieldCurveAI class not found"
        
        # Check for required methods
        required_methods = [
            'load_model_metrics',
            'load_available_models', 
            'load_feature_data',
            'predict_yield_curve',
            'create_yield_curve_plot',
            'display_forecast_page',
            'display_model_info_page',
            'run'
        ]
        
        found_methods = []
        for node in ast.walk(yield_curve_class):
            if isinstance(node, ast.FunctionDef):
                found_methods.append(node.name)
        
        missing_methods = [m for m in required_methods if m not in found_methods]
        
        if missing_methods:
            return False, f"Missing methods: {missing_methods}"
        
        return True, f"All required methods found: {found_methods}"
        
    except Exception as e:
        return False, f"Error checking methods: {e}"

def main():
    """Main validation function."""
    print("🔍 Validating YieldCurveAI Application...")
    print("=" * 50)
    
    # Check if main file exists
    main_file = Path("YieldCurveAI.py")
    if not main_file.exists():
        print("❌ YieldCurveAI.py not found")
        return False
    
    print("✅ YieldCurveAI.py found")
    
    # Validate syntax
    is_valid, message = validate_python_syntax(main_file)
    if not is_valid:
        print(f"❌ {message}")
        return False
    
    print("✅ Python syntax is valid")
    
    # Check required methods
    methods_valid, methods_message = check_required_methods(main_file)
    if not methods_valid:
        print(f"❌ {methods_message}")
        return False
    
    print("✅ All required methods present")
    
    # Check runner script
    runner_file = Path("run_app.py")
    if runner_file.exists():
        runner_valid, runner_message = validate_python_syntax(runner_file)
        if runner_valid:
            print("✅ run_app.py is valid")
        else:
            print(f"⚠️  run_app.py has issues: {runner_message}")
    
    # Check README
    readme_file = Path("README_YieldCurveAI.md")
    if readme_file.exists():
        print("✅ README_YieldCurveAI.md found")
    
    print("\n🎉 Validation Summary:")
    print("✅ Core application structure is valid")
    print("✅ All required classes and methods are present") 
    print("✅ Python syntax is correct")
    print("\n📋 To run the application:")
    print("1. Install dependencies: pip install streamlit pandas numpy plotly scikit-learn")
    print("2. Ensure required data files are present")
    print("3. Run: streamlit run YieldCurveAI.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 