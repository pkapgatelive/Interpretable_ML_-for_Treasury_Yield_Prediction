# Interpretable Machine-Learning Models for Yield-Curve Forecasting and Monetary-Policy Scenario Analysis

A comprehensive research project implementing state-of-the-art machine learning models for yield curve forecasting with a focus on interpretability and monetary policy analysis.

## 🎯 Project Objectives

- **Forecasting Excellence**: Develop accurate yield curve forecasting models using advanced ML techniques
- **Interpretability**: Ensure model decisions are explainable and actionable for policy analysis
- **Policy Analysis**: Enable scenario-based analysis for monetary policy decision-making
- **Research Rigor**: Maintain reproducibility and academic standards throughout

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Phase 1: Data Acquisition](#phase-1-data-acquisition)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- FRED API Key (free registration at [FRED](https://fred.stlouisfed.org/docs/api/fred/))

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd yield-curve-forecasting
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using conda
   conda env create -f environment.yml
   conda activate yield-curve-env
   
   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Create .env file
   echo "FRED_API_KEY=your_fred_api_key_here" > .env
   ```

4. **Initialize data versioning**
   ```bash
   dvc init
   ```

5. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## 📁 Project Structure

```
yield-curve-forecasting/
├── config/                 # Configuration files
│   ├── config.yaml        # Main configuration
│   └── model_config.yaml  # Model-specific settings
├── data/                  # Data directory (managed by DVC)
│   ├── raw/              # Raw downloaded data
│   ├── processed/        # Cleaned and processed data
│   └── external/         # External data sources
├── docs/                 # Documentation
├── models/               # Trained models (managed by DVC)
├── notebooks/            # Jupyter notebooks for analysis
├── reports/              # Generated reports and figures
├── scripts/              # Automation scripts
├── src/                  # Source code
│   ├── data/            # Data acquisition and processing
│   ├── models/          # Model implementations
│   ├── visualization/   # Plotting and visualization
│   ├── explainability/ # Model interpretation
│   └── utils/           # Utility functions
└── tests/               # Unit tests
```

## 📊 Phase 1: Data Acquisition

Phase 1 implements automated data acquisition from trusted financial data sources with comprehensive validation and documentation.

### 🏛️ Data Sources

| Source | Description | Coverage | API |
|--------|-------------|----------|-----|
| **FRED** | Federal Reserve Economic Data | US Treasury yields, macro indicators | fredapi |
| **ECB SDW** | European Central Bank Statistical Data Warehouse | European government bond yields | pandasdmx |

### 📈 Available Data

#### US Treasury Yield Curves
- **Tenors**: 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
- **Frequency**: Daily
- **Coverage**: 1990-present
- **Source**: FRED

#### Macroeconomic Covariates
- **Monetary Policy**: Fed Funds Rate, Daily Fed Funds Rate
- **Inflation**: CPI, Core CPI, PCE, Breakeven Inflation (5Y, 10Y)
- **Economic Activity**: Industrial Production, Payrolls, Unemployment, PMI
- **Financial Markets**: VIX, S&P 500, USD/EUR, USD/JPY
- **Credit**: Corporate Spreads, TED Spread
- **Housing**: Housing Starts, Case-Shiller Index

#### European Government Bond Yields
- **Countries**: Germany (DE), France (FR), Italy (IT)
- **Tenors**: 1Y, 2Y, 5Y, 10Y, 30Y
- **Frequency**: Daily
- **Source**: ECB Statistical Data Warehouse

### 🚀 Quick Start

#### 1. Download US Treasury Data
```bash
# Download all tenors for default date range
python src/data/get_treasury_yield.py download

# Download specific tenors
python src/data/get_treasury_yield.py download --tenors 2Y,5Y,10Y,30Y

# Download historical data
python src/data/get_treasury_yield.py download \
    --start-date 2000-01-01 \
    --end-date 2023-12-31 \
    --format parquet

# View available tenors and series information
python src/data/get_treasury_yield.py info
```

#### 2. Download Macroeconomic Data
```bash
# Download default macro variables
python src/data/get_macro_covariates.py download

# Download specific variables
python src/data/get_macro_covariates.py download \
    --variables FEDFUNDS,CPIAUCSL,UNRATE,VIXCLS

# Download with custom date range
python src/data/get_macro_covariates.py download \
    --start-date 1990-01-01 \
    --end-date 2024-12-31 \
    --format parquet

# View available variables by category
python src/data/get_macro_covariates.py info
```

#### 3. Download ECB Yield Data
```bash
# Download German Bunds
python src/data/get_ecb_yields.py download --countries DE

# Download multiple European countries
python src/data/get_ecb_yields.py download \
    --countries DE,FR,IT \
    --tenors 2Y,5Y,10Y

# Download with validation
python src/data/get_ecb_yields.py download \
    --start-date 2010-01-01 \
    --validate

# View available countries and tenors
python src/data/get_ecb_yields.py info
```

#### 4. Generate Data Dictionary
```bash
# Generate comprehensive data documentation
python src/data/generate_data_dictionary.py generate

# Generate in JSON format
python src/data/generate_data_dictionary.py generate \
    --format json \
    --output-dir docs/

# Include statistical metadata
python src/data/generate_data_dictionary.py generate \
    --include-stats
```

### 📋 CLI Reference

#### Common Parameters

All data acquisition scripts support these common parameters:

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--start-date` | Start date (YYYY-MM-DD) | 1990-01-01 | 2020-01-01 |
| `--end-date` | End date (YYYY-MM-DD) | Today | 2023-12-31 |
| `--output-dir` | Output directory | data/raw | data/custom |
| `--format` | Output format | csv | parquet |
| `--validate` | Validate downloaded data | True | --no-validate |
| `--api-key` | FRED API key | From env | your_key_here |

#### Script-Specific Parameters

**Treasury Yields (`get_treasury_yield.py`)**
```bash
--tenors        # Comma-separated tenor list (default: all)
--save-validation  # Save validation report to file
```

**Macro Covariates (`get_macro_covariates.py`)**
```bash
--variables     # Comma-separated FRED series IDs
```

**ECB Yields (`get_ecb_yields.py`)**
```bash
--countries     # Comma-separated country codes (DE,FR,IT)
--tenors        # Comma-separated tenor list
```

### 🔍 Data Validation

All scripts include comprehensive data validation:

#### Schema Validation
- **Data Types**: Ensures proper date and numeric types
- **Range Checks**: Validates reasonable ranges for financial data
- **Required Fields**: Checks for mandatory columns

#### Quality Checks
- **Missing Data**: Reports missing value patterns
- **Date Continuity**: Validates chronological ordering
- **Financial Logic**: Checks yield curve consistency
- **Outlier Detection**: Identifies unusual values

#### Validation Reports
```bash
# Save detailed validation reports
python src/data/get_treasury_yield.py download --save-validation

# Validation reports include:
# - Schema compliance
# - Missing data analysis  
# - Range validation
# - Statistical summaries
# - Data quality metrics
```

### 📂 Output File Structure

Data files are automatically timestamped and organized:

```
data/raw/
├── yieldcurve_us_20241201.csv          # US Treasury yields
├── macro_fred_20241201.csv             # Macro covariates  
├── yieldcurve_ecb_20241201.csv         # ECB yields
├── treasury_validation_20241201.json   # Validation report
└── data_dictionary_20241201.md         # Data documentation
```

### 🔧 Advanced Usage

#### Custom Data Pipeline
```bash
# Download all data sources in sequence
python src/data/get_treasury_yield.py download --format parquet
python src/data/get_macro_covariates.py download --format parquet  
python src/data/get_ecb_yields.py download --format parquet
python src/data/generate_data_dictionary.py generate --include-stats
```

#### Automated Updates
```bash
# Set up daily data updates (example cron job)
# 0 6 * * * cd /path/to/project && python src/data/get_treasury_yield.py download
```

#### Environment Variables
```bash
# Set FRED API key
export FRED_API_KEY="your_api_key_here"

# Configure output directory
export DATA_OUTPUT_DIR="data/raw"

# Set validation options
export ENABLE_VALIDATION="true"
```

### 📊 Data Quality Metrics

Each download includes comprehensive quality metrics:

- **Completeness**: Percentage of non-missing values
- **Timeliness**: Data coverage and update frequency  
- **Consistency**: Cross-validation between related series
- **Accuracy**: Range and outlier validation
- **Uniqueness**: Duplicate detection

### 🚨 Troubleshooting

#### Common Issues

**API Key Errors**
```bash
# Error: FRED API key not found
export FRED_API_KEY="your_key_here"
# Or pass directly
python get_treasury_yield.py download --api-key your_key_here
```

**Network Issues**
```bash
# Retry with smaller date ranges
python get_treasury_yield.py download --start-date 2023-01-01
```

**Validation Failures**
```bash
# Skip validation if needed
python get_treasury_yield.py download --no-validate
```

**Missing Dependencies**
```bash
# Install additional packages
pip install pandasdmx rich typer pandera
```

## ⚙️ Configuration

The project uses YAML configuration files for all settings:

- **`config/config.yaml`**: Main project configuration
- **`config/model_config.yaml`**: Model hyperparameters and settings

Configuration supports environment variable substitution and hierarchical settings.

## 🧪 Development

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test category
pytest tests/test_data/
```

### Code Quality
```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type checking  
mypy src/

# Sort imports
isort src/ tests/
```

### Pre-commit Hooks
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure all tests pass and code follows the project style guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research Context

This project is part of academic research in financial modeling and monetary policy analysis. If you use this code in your research, please cite:

```bibtex
@misc{yield-curve-forecasting,
  title={Interpretable Machine-Learning Models for Yield-Curve Forecasting and Monetary-Policy Scenario Analysis},
  author={[Your Name]},
  year={2024},
  howpublished={\url{https://github.com/your-username/yield-curve-forecasting}}
}
```

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and community interaction
- **Documentation**: Check the `docs/` directory for detailed guides

---

**Next Phase**: Data Processing and Feature Engineering 