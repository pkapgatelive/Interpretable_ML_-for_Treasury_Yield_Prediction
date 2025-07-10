# Data Dictionary
**Project:** Interpretable Machine-Learning Models for Yield-Curve Forecasting
**Generated:** 2025-07-10T22:59:56.934555
**Version:** 1.0.0

Comprehensive data dictionary for yield curve forecasting project

## Data Sources

### FRED
**Name:** Federal Reserve Economic Data
**Provider:** Federal Reserve Bank of St. Louis
**URL:** https://fred.stlouisfed.org/
**License:** Public Domain
**Update Frequency:** Daily/Monthly (varies by series)
**Access Method:** REST API

### ECB
**Name:** European Central Bank Statistical Data Warehouse
**Provider:** European Central Bank
**URL:** https://sdw.ecb.europa.eu/
**License:** Creative Commons Attribution 4.0
**Update Frequency:** Daily
**Access Method:** SDMX API

## Variable Categories

### Yield Curves
**Description:** Government bond yield curves representing the term structure of interest rates
**Frequency:** Daily
**Unit:** Percent per annum

### Monetary Policy
**Description:** Central bank policy rates and related monetary policy indicators
**Frequency:** Daily/Monthly
**Unit:** Percent per annum

### Inflation
**Description:** Price level indicators and inflation expectations
**Frequency:** Monthly/Daily
**Unit:** Index/Percent

### Economic Activity
**Description:** Real economy indicators measuring economic growth and business conditions
**Frequency:** Monthly
**Unit:** Index/Thousands/Percent

### Financial Markets
**Description:** Financial market indicators including equity prices, exchange rates, and volatility
**Frequency:** Daily
**Unit:** Index/Rate/Percent

### Credit Conditions
**Description:** Credit market conditions and risk spreads
**Frequency:** Daily
**Unit:** Basis points/Percent

## Variables

| Variable | Description | Source | Frequency | Unit | Country |
|----------|-------------|---------|-----------|------|---------|
| US_TREASURY_1M | 1-Month Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_3M | 3-Month Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_6M | 6-Month Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_1Y | 1-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_2Y | 2-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_3Y | 3-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_5Y | 5-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_7Y | 7-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_10Y | 10-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_20Y | 20-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| US_TREASURY_30Y | 30-Year Treasury Constant Maturity Rate | FRED | Daily | Percent per annum | United States |
| FEDFUNDS | Federal Funds Rate | FRED | Monthly | Percent per annum | United States |
| DFF | Daily Federal Funds Rate | FRED | Daily | Percent per annum | United States |
| CPIAUCSL | Consumer Price Index for All Urban Consumers: All Items | FRED | Monthly | Index (1982-84=100) | United States |
| UNRATE | Unemployment Rate | FRED | Monthly | Percent | United States |
| VIXCLS | CBOE Volatility Index: VIX | FRED | Daily | Percent | United States |
| SP500 | S&P 500 | FRED | Daily | Index | United States |
| ECB_DE_1Y | 1Y yield on Germany government bonds | ECB | Daily | Percent per annum | Germany |
| ECB_DE_2Y | 2Y yield on Germany government bonds | ECB | Daily | Percent per annum | Germany |
| ECB_DE_5Y | 5Y yield on Germany government bonds | ECB | Daily | Percent per annum | Germany |
| ECB_DE_10Y | 10Y yield on Germany government bonds | ECB | Daily | Percent per annum | Germany |
| ECB_DE_30Y | 30Y yield on Germany government bonds | ECB | Daily | Percent per annum | Germany |
| ECB_FR_1Y | 1Y yield on France government bonds | ECB | Daily | Percent per annum | France |
| ECB_FR_2Y | 2Y yield on France government bonds | ECB | Daily | Percent per annum | France |
| ECB_FR_5Y | 5Y yield on France government bonds | ECB | Daily | Percent per annum | France |
| ECB_FR_10Y | 10Y yield on France government bonds | ECB | Daily | Percent per annum | France |
| ECB_FR_30Y | 30Y yield on France government bonds | ECB | Daily | Percent per annum | France |
| ECB_IT_1Y | 1Y yield on Italy government bonds | ECB | Daily | Percent per annum | Italy |
| ECB_IT_2Y | 2Y yield on Italy government bonds | ECB | Daily | Percent per annum | Italy |
| ECB_IT_5Y | 5Y yield on Italy government bonds | ECB | Daily | Percent per annum | Italy |
| ECB_IT_10Y | 10Y yield on Italy government bonds | ECB | Daily | Percent per annum | Italy |
| ECB_IT_30Y | 30Y yield on Italy government bonds | ECB | Daily | Percent per annum | Italy |

## Detailed Variable Information

### Yield Curves

#### US_TREASURY_1M
- **Description:** 1-Month Treasury Constant Maturity Rate
- **Series ID:** DGS1MO
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 1M
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_3M
- **Description:** 3-Month Treasury Constant Maturity Rate
- **Series ID:** DGS3MO
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 3M
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_6M
- **Description:** 6-Month Treasury Constant Maturity Rate
- **Series ID:** DGS6MO
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 6M
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_1Y
- **Description:** 1-Year Treasury Constant Maturity Rate
- **Series ID:** DGS1
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 1Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_2Y
- **Description:** 2-Year Treasury Constant Maturity Rate
- **Series ID:** DGS2
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 2Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_3Y
- **Description:** 3-Year Treasury Constant Maturity Rate
- **Series ID:** DGS3
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 3Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_5Y
- **Description:** 5-Year Treasury Constant Maturity Rate
- **Series ID:** DGS5
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 5Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_7Y
- **Description:** 7-Year Treasury Constant Maturity Rate
- **Series ID:** DGS7
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 7Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_10Y
- **Description:** 10-Year Treasury Constant Maturity Rate
- **Series ID:** DGS10
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 10Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_20Y
- **Description:** 20-Year Treasury Constant Maturity Rate
- **Series ID:** DGS20
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 20Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### US_TREASURY_30Y
- **Description:** 30-Year Treasury Constant Maturity Rate
- **Series ID:** DGS30
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Tenor:** 30Y
- **Currency:** USD
- **Typical Range:** 0% to 15%
- **Validation Rules:** Non-negative, < 50%, Term structure consistency

#### ECB_DE_1Y
- **Description:** 1Y yield on Germany government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_1Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Germany
- **Tenor:** 1Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_DE_2Y
- **Description:** 2Y yield on Germany government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_2Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Germany
- **Tenor:** 2Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_DE_5Y
- **Description:** 5Y yield on Germany government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_5Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Germany
- **Tenor:** 5Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_DE_10Y
- **Description:** 10Y yield on Germany government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_10Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Germany
- **Tenor:** 10Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_DE_30Y
- **Description:** 30Y yield on Germany government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_30Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Germany
- **Tenor:** 30Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_FR_1Y
- **Description:** 1Y yield on France government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_1Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** France
- **Tenor:** 1Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_FR_2Y
- **Description:** 2Y yield on France government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_2Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** France
- **Tenor:** 2Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_FR_5Y
- **Description:** 5Y yield on France government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_5Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** France
- **Tenor:** 5Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_FR_10Y
- **Description:** 10Y yield on France government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_10Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** France
- **Tenor:** 10Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_FR_30Y
- **Description:** 30Y yield on France government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_30Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** France
- **Tenor:** 30Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_IT_1Y
- **Description:** 1Y yield on Italy government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_1Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Italy
- **Tenor:** 1Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_IT_2Y
- **Description:** 2Y yield on Italy government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_2Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Italy
- **Tenor:** 2Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_IT_5Y
- **Description:** 5Y yield on Italy government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_5Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Italy
- **Tenor:** 5Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_IT_10Y
- **Description:** 10Y yield on Italy government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_10Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Italy
- **Tenor:** 10Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

#### ECB_IT_30Y
- **Description:** 30Y yield on Italy government bonds
- **Series ID:** YC.B.U2.EUR.4F.G_N_*.SV_C_YM.SR_30Y
- **Source:** ECB
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** Italy
- **Tenor:** 30Y
- **Currency:** EUR
- **Typical Range:** -1% to 10%
- **Validation Rules:** Can be negative, < 20%, Term structure consistency

### Monetary Policy

#### FEDFUNDS
- **Description:** Federal Funds Rate
- **Series ID:** FEDFUNDS
- **Source:** FRED
- **Frequency:** Monthly
- **Unit:** Percent per annum
- **Country:** United States
- **Currency:** USD
- **Validation Rules:** Non-negative, < 25%

#### DFF
- **Description:** Daily Federal Funds Rate
- **Series ID:** DFF
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent per annum
- **Country:** United States
- **Currency:** USD
- **Validation Rules:** Non-negative, < 25%

### Inflation

#### CPIAUCSL
- **Description:** Consumer Price Index for All Urban Consumers: All Items
- **Series ID:** CPIAUCSL
- **Source:** FRED
- **Frequency:** Monthly
- **Unit:** Index (1982-84=100)
- **Country:** United States
- **Validation Rules:** Non-negative, < 1000 (index), < 50% (rate)

### Economic Activity

#### UNRATE
- **Description:** Unemployment Rate
- **Series ID:** UNRATE
- **Source:** FRED
- **Frequency:** Monthly
- **Unit:** Percent
- **Country:** United States
- **Currency:** USD
- **Validation Rules:** Non-negative, < 100% (unemployment), < 200 (index)

### Financial Markets

#### VIXCLS
- **Description:** CBOE Volatility Index: VIX
- **Series ID:** VIXCLS
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Percent
- **Country:** United States
- **Validation Rules:** Non-negative, Volatility < 200%

#### SP500
- **Description:** S&P 500
- **Series ID:** SP500
- **Source:** FRED
- **Frequency:** Daily
- **Unit:** Index
- **Country:** United States
- **Validation Rules:** Non-negative, Volatility < 200%
