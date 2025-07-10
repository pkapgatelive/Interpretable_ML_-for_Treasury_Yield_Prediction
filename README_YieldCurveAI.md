# YieldCurveAI 📈
**Professional U.S. Treasury Yield Curve Forecasting Application**

A sophisticated Streamlit web application designed for non-technical users (analysts, policymakers) to generate U.S. Treasury yield curve forecasts using state-of-the-art machine learning models.

![YieldCurveAI](https://img.shields.io/badge/YieldCurveAI-Professional-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge)

## 🌟 Features

### 📈 Yield Forecast Tool
- **Interactive Sidebar Controls**: Easy-to-use inputs for forecast parameters
- **Economic Indicators**: Fed Funds Rate and CPI Year-over-Year inputs
- **Flexible Horizons**: 1-day, 1-week, and 1-month forecasting options
- **Smart Model Selection**: Automatic best model selection or manual choice
- **Real-time Predictions**: Instant yield curve generation across all tenors (3M to 30Y)
- **Maturity Date Calculation**: Precise maturity dates based on forecast start date
- **Professional Visualizations**: Interactive Plotly charts with hover tooltips
- **Export Options**: CSV download functionality for results

### 📊 Model Training & Validation Info
- **Comprehensive Model Overview**: Performance metrics for all available models
- **Best Model Highlighting**: Automatic identification of top-performing model
- **Detailed Comparisons**: RMSE, MAE, R², and MAPE metrics side-by-side
- **Model Descriptions**: Clear explanations of each algorithm
- **Training Details**: Information about data, features, and validation methods
- **Performance Visualizations**: Feature importance and prediction plots

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Required data files (see Project Structure below)

### Installation & Setup

1. **Navigate to the project directory:**
   ```bash
   cd yield-curve-forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn
   ```

3. **Launch the application:**
   ```bash
   python run_app.py
   ```
   
   Or manually:
   ```bash
   streamlit run YieldCurveAI.py
   ```

4. **Open your browser** and navigate to the URL displayed in the terminal (typically `http://localhost:8501`)

## 📁 Project Structure

Ensure the following files and directories exist:

```
yield-curve-forecasting/
├── YieldCurveAI.py                 # Main application file
├── run_app.py                      # Application runner
├── data/processed/
│   └── X_features.csv              # Feature data for predictions
├── models/trained/
│   ├── elastic_net_*.pkl           # Trained ElasticNet model
│   ├── ridge_*.pkl                 # Trained Ridge model
│   ├── lasso_*.pkl                 # Trained LASSO model
│   ├── random_forest_*.pkl         # Trained Random Forest model
│   ├── gradient_boosting_*.pkl     # Trained Gradient Boosting model
│   └── svr_*.pkl                   # Trained SVR model
└── reports/
    ├── model_metrics/
    │   └── metrics_summary_*.json  # Model performance metrics
    └── figures/
        ├── feature_importance_*.png
        └── prediction_plots_*.png
```

## 🎯 How to Use

### 1. Yield Forecasting

1. **Select Forecast Parameters:**
   - Choose your forecast start date
   - Set Fed Funds Rate (current market rate)
   - Set CPI Year-over-Year (current inflation rate)
   - Select forecast horizon (1-day, 1-week, or 1-month)

2. **Choose Model Selection Mode:**
   - **Automatic**: Uses the best-performing model based on RMSE
   - **Manual**: Select from Ridge, LASSO, ElasticNet, Random Forest, etc.

3. **Generate Forecast:**
   - Click "🚀 Generate Forecast"
   - View results table with tenors, maturity dates, and predicted yields
   - Analyze the interactive yield curve visualization
   - Download results as CSV if needed

### 2. Model Analysis

1. **Navigate to Model Info page**
2. **Review Best Model:** See which model performs best
3. **Compare Models:** Analyze performance metrics across all models
4. **Explore Details:** Click on individual model tabs for detailed information
5. **View Visualizations:** Examine feature importance and prediction plots

## 🔧 Technical Details

### Available Models
- **Linear Regression (Ridge)**: L2 regularization for stability
- **LASSO**: L1 regularization for feature selection
- **ElasticNet**: Combined L1/L2 regularization (often best performer)
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble learning
- **Support Vector Regression**: Kernel-based non-linear modeling

### Input Features
The application uses macroeconomic indicators including:
- Federal Funds Rate (current and lagged values)
- Consumer Price Index (CPI) data
- Unemployment Rate
- GDP indicators
- Dollar strength measures
- VIX volatility index
- Yield curve factors and technical indicators

### Yield Curve Tenors
- 3-Month Treasury
- 6-Month Treasury
- 1-Year Treasury
- 2-Year Treasury
- 3-Year Treasury
- 5-Year Treasury
- 7-Year Treasury
- 10-Year Treasury
- 20-Year Treasury
- 30-Year Treasury

## 📊 Understanding the Output

### Results Table
- **Tenor**: Treasury security maturity period
- **Maturity Date**: Exact date when the security matures
- **Predicted Yield (%)**: Forecasted annual yield

### Yield Curve Chart
- **X-axis**: Treasury tenors from short (3M) to long (30Y)
- **Y-axis**: Predicted yields as percentages
- **Tooltips**: Hover for detailed information including maturity dates

### Performance Metrics
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **R²**: Coefficient of determination (higher is better, max 1.0)
- **MAPE**: Mean Absolute Percentage Error (lower is better)

## 🎨 User Interface

### Professional Design
- Clean, intuitive layout optimized for non-technical users
- Professional color scheme with blue/gray palette
- Responsive design that works on different screen sizes
- Clear navigation between Forecast and Model Info pages

### Interactive Elements
- Real-time input validation
- Loading spinners for long operations
- Success/error messages with clear feedback
- Hover tooltips for additional information

## ⚠️ Important Notes

1. **Data Requirements**: The application requires trained models and processed data files
2. **Forecast Accuracy**: Results are model predictions and should be used alongside other analysis
3. **Market Conditions**: Consider current market volatility and economic conditions
4. **Professional Use**: Designed for institutional analysis and policy decision support

## 🔍 Troubleshooting

### Common Issues

**Application won't start:**
- Ensure all required files exist in the correct directories
- Check that Python dependencies are installed
- Verify you're in the correct directory

**No predictions generated:**
- Check model files are properly formatted
- Ensure feature data is available and readable
- Verify input parameters are within reasonable ranges

**Visualization not displaying:**
- Update Plotly to the latest version
- Check browser compatibility
- Clear browser cache if needed

## 📈 Future Enhancements

- Real-time data integration from Federal Reserve APIs
- Additional economic indicators and features
- Model ensemble voting mechanisms
- Historical backtesting visualizations
- Confidence intervals for predictions
- Advanced scenario analysis tools

## 🤝 Support

For technical support or questions about the YieldCurveAI application:

1. Check this README for common solutions
2. Verify all required files and dependencies are present
3. Review the terminal output for error messages
4. Ensure input parameters are within reasonable ranges

## 📄 License

This application is designed for educational and research purposes. Please ensure appropriate usage in professional environments.

---

**YieldCurveAI** - Bringing professional-grade yield curve forecasting to your browser. 📈✨ 