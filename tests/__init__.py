"""
Test package for yield curve forecasting project.

This package contains unit tests, integration tests, and test utilities
for validating the functionality of the yield curve forecasting models
and supporting infrastructure.

Test Structure:
- test_data/: Tests for data loading and preprocessing
- test_models/: Tests for model training and prediction
- test_utils/: Tests for utility functions
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
TEST_CONFIG_PATH = project_root / "config" / "config.yaml"
TEST_MODEL_CONFIG_PATH = project_root / "config" / "model_config.yaml"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
(TEST_DATA_DIR / "sample_data").mkdir(exist_ok=True)
(TEST_DATA_DIR / "sample_models").mkdir(exist_ok=True)

# Test utilities
def get_test_data_path(filename: str) -> Path:
    """Get path to test data file."""
    return TEST_DATA_DIR / "sample_data" / filename

def get_test_model_path(filename: str) -> Path:
    """Get path to test model file."""
    return TEST_DATA_DIR / "sample_models" / filename

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "data: marks tests related to data processing"
    )
    config.addinivalue_line(
        "markers", "models: marks tests related to model training/prediction"
    )

# Test fixtures and utilities will be added here
__all__ = [
    "TEST_DATA_DIR",
    "TEST_CONFIG_PATH", 
    "TEST_MODEL_CONFIG_PATH",
    "get_test_data_path",
    "get_test_model_path"
] 