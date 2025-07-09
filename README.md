# Interpretable Machine-Learning Models for Yield-Curve Forecasting and Monetary-Policy Scenario Analysis

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📖 Project Overview

This project develops interpretable machine learning models for forecasting government bond yield curves and conducting monetary policy scenario analysis. The research focuses on creating transparent, explainable models that can assist central banks and financial institutions in understanding interest rate dynamics and policy impacts.

### 🎯 Objectives

- **Yield Curve Forecasting**: Develop accurate ML models to predict future yield curve shapes
- **Policy Impact Analysis**: Simulate how monetary policy changes affect interest rate structures
- **Model Interpretability**: Ensure all models provide clear, actionable insights using SHAP and other explainability techniques
- **Academic Rigor**: Maintain reproducible research standards suitable for publication

## 🗂️ Project Organization

```
├── README.md                     <- The top-level README for developers using this project
├── requirements.txt              <- Python dependencies for reproducing the environment
├── environment.yml               <- Conda environment specification
├── pyproject.toml               <- Modern Python project configuration
├── .gitignore                   <- Git ignore patterns
├── .pre-commit-config.yaml      <- Pre-commit hooks configuration
├── dvc.yaml                     <- DVC pipeline configuration
│
├── config/                      <- Configuration files
│   ├── config.yaml             <- Main project configuration
│   └── model_config.yaml       <- Model-specific parameters
│
├── data/                        <- Data directory (tracked with DVC)
│   ├── raw/                    <- Original, immutable data
│   ├── processed/              <- Cleaned and transformed data
│   ├── features/               <- Engineered features for modeling
│   └── external/               <- External reference data
│
├── src/                         <- Source code for use in this project
│   ├── data/                   <- Data loading and processing modules
│   ├── models/                 <- Model training and evaluation code
│   ├── visualization/          <- Plotting and dashboard utilities
│   ├── explainability/         <- Model interpretation tools
│   └── utils/                  <- Utility functions and constants
│
├── notebooks/                   <- Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_ml_models.ipynb
│   ├── 05_explainability_analysis.ipynb
│   └── 06_policy_scenario_simulation.ipynb
│
├── models/                      <- Trained models and artifacts
│   ├── trained/                <- Final trained models
│   ├── checkpoints/            <- Model checkpoints during training
│   └── artifacts/              <- Model metadata and performance metrics
│
├── reports/                     <- Generated reports and visualizations
│   ├── figures/                <- Publication-ready figures
│   ├── tables/                 <- Results tables and summaries
│   └── presentations/          <- Presentation materials
│
├── scripts/                     <- Standalone scripts for automation
│   ├── train_model.py          <- Model training pipeline
│   ├── evaluate_model.py       <- Model evaluation script
│   ├── generate_forecasts.py   <- Forecast generation
│   └── run_pipeline.py         <- Full pipeline execution
│
└── tests/                       <- Unit tests
    ├── test_data/              <- Data processing tests
    ├── test_models/            <- Model testing
    └── test_utils/             <- Utility function tests
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- DVC (Data Version Control)
- Conda or pip for package management

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd yield-curve-forecasting
   ```

2. **Set up the environment**:
   
   Using conda:
   ```bash
   conda env create -f environment.yml
   conda activate yield-forecasting
   ```
   
   Or using pip:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Initialize DVC** (if not already done):
   ```bash
   dvc init
   ```

### First Steps

1. **Explore the data**: Start with `notebooks/01_data_exploration.ipynb`
2. **Configure parameters**: Edit `config/config.yaml` for your specific use case
3. **Run the pipeline**: Execute `python scripts/run_pipeline.py`

## 📊 Data Sources

*[To be completed with specific yield curve data sources]*

- Federal Reserve Economic Data (FRED)
- Central bank databases
- Bloomberg/Refinitiv data feeds
- Academic datasets

## 🔬 Methodology

### Models Implemented
- **Baseline Models**: Linear regression, ARIMA, Nelson-Siegel
- **Machine Learning**: Random Forest, Gradient Boosting, Neural Networks
- **Ensemble Methods**: Stacking, voting classifiers
- **Deep Learning**: LSTM, Transformer architectures

### Interpretability Techniques
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Partial dependence plots

## 📈 Results

*[To be completed with model performance metrics]*

## 📝 Publication Status

*[To be updated with publication information]*

- Conference submissions: [TBD]
- Journal submissions: [TBD]
- Working papers: [TBD]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

[Your Name] - [your.email@institution.edu]

Project Link: [https://github.com/username/yield-curve-forecasting](https://github.com/username/yield-curve-forecasting)

## 🙏 Acknowledgments

- [Research institution/supervisor acknowledgments]
- [Data provider acknowledgments]
- [Funding source acknowledgments] 