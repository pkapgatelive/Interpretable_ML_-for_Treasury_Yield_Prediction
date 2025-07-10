# ❄️ Snowflake Streamlit Deployment Guide

## 🎯 **Deploying YieldCurveAI on Snowflake**

Snowflake's Streamlit platform provides enterprise-grade hosting with advanced security, data integration, and scalability features. This guide will help you deploy your YieldCurveAI application on Snowflake.

---

## 🏢 **Enterprise Benefits**

### ✅ **Enterprise Security & Compliance**
- **SOC 2 Type II** certified infrastructure
- **Role-based access control** for app sharing
- **Data governance** and audit trails
- **Network security policies**

### ✅ **Seamless Data Integration**
- **Direct access** to Snowflake data warehouses
- **Real-time data pipelines** 
- **Native SQL integration**
- **No data movement required**

### ✅ **Enterprise Support & Management**
- **Fully managed infrastructure**
- **Auto-scaling compute resources**
- **Enterprise-grade SLA**
- **24/7 technical support**

---

## 📋 **Prerequisites**

1. **Snowflake Account** (Enterprise or Business Critical tier recommended)
2. **ACCOUNTADMIN** or **SYSADMIN** privileges
3. **YieldCurveAI application code** (ready for upload)
4. **Snowflake CLI** or **Web Interface** access

---

## 🚀 **Deployment Steps**

### **Step 1: Prepare Your Snowflake Environment**

#### 1.1 Create Database and Schema
```sql
-- Connect to Snowflake as ACCOUNTADMIN/SYSADMIN
USE ROLE ACCOUNTADMIN;

-- Create database for YieldCurve application
CREATE DATABASE IF NOT EXISTS YIELDCURVE_DB;

-- Create schemas for different data types
CREATE SCHEMA IF NOT EXISTS YIELDCURVE_DB.STREAMLIT_APPS;
CREATE SCHEMA IF NOT EXISTS YIELDCURVE_DB.ML_MODELS;
CREATE SCHEMA IF NOT EXISTS YIELDCURVE_DB.FEATURES;
CREATE SCHEMA IF NOT EXISTS YIELDCURVE_DB.RAW_DATA;
```

#### 1.2 Set Up User Roles and Permissions
```sql
-- Create role for YieldCurve app users
CREATE ROLE IF NOT EXISTS YIELDCURVE_USERS;

-- Grant necessary permissions
GRANT USAGE ON DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_USERS;
GRANT USAGE ON ALL SCHEMAS IN DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_USERS;
GRANT SELECT ON ALL TABLES IN DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_USERS;

-- Grant role to users (replace with actual usernames)
GRANT ROLE YIELDCURVE_USERS TO USER "DR_KAPILA_MALLAH";
GRANT ROLE YIELDCURVE_USERS TO USER "MR_PAPPU_KAPGATE";
GRANT ROLE YIELDCURVE_USERS TO USER "DR_ERIC_KATOVAI";
```

### **Step 2: Upload Application to Snowflake**

#### 2.1 Using Snowflake Web Interface

1. **Login to Snowflake** Web UI
2. **Navigate to** "Streamlit" in the left sidebar
3. **Click "Create Streamlit App"**
4. **Configure the app:**
   - **App Name:** `YieldCurveAI`
   - **Warehouse:** Create or select appropriate warehouse
   - **Database:** `YIELDCURVE_DB`
   - **Schema:** `STREAMLIT_APPS`

#### 2.2 Upload Your Application Code

**Option A: Direct Upload**
1. **Copy your `YieldCurveAI.py` content**
2. **Paste into the Snowflake Streamlit editor**
3. **Save and run**

**Option B: Git Integration**
1. **Connect your GitHub repository**
2. **Select the main branch**
3. **Set main file as `YieldCurveAI.py`**

### **Step 3: Configure Data Sources**

#### 3.1 Create Sample Data Tables (if needed)
```sql
-- Create model metrics table
CREATE TABLE YIELDCURVE_DB.ML_MODELS.MODEL_METRICS (
    model_name STRING,
    rmse FLOAT,
    mae FLOAT,
    r2 FLOAT,
    mape FLOAT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Insert sample data
INSERT INTO YIELDCURVE_DB.ML_MODELS.MODEL_METRICS VALUES
('elastic_net', 0.0234, 0.0187, 0.892, 2.34, TRUE, CURRENT_TIMESTAMP()),
('ridge', 0.0267, 0.0203, 0.876, 2.67, TRUE, CURRENT_TIMESTAMP()),
('lasso', 0.0289, 0.0221, 0.856, 2.89, TRUE, CURRENT_TIMESTAMP()),
('random_forest', 0.0245, 0.0195, 0.885, 2.45, TRUE, CURRENT_TIMESTAMP()),
('gradient_boosting', 0.0238, 0.0189, 0.889, 2.38, TRUE, CURRENT_TIMESTAMP()),
('svr', 0.0278, 0.0215, 0.864, 2.78, TRUE, CURRENT_TIMESTAMP());
```

#### 3.2 Create Features Table
```sql
-- Create features table for real-time data
CREATE TABLE YIELDCURVE_DB.FEATURES.PROCESSED_FEATURES (
    date DATE,
    fed_funds_rate FLOAT,
    cpi_yoy FLOAT,
    unemployment_rate FLOAT,
    vix FLOAT,
    yield_spread_10y_2y FLOAT,
    yield_level FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);
```

### **Step 4: Modify Application for Snowflake Integration**

#### 4.1 Add Snowflake Connection Code
Add this to the top of your `YieldCurveAI.py`:

```python
# Snowflake integration
try:
    from snowflake.snowpark.context import get_active_session
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

class YieldCurveAI:
    def __init__(self):
        # ... existing code ...
        
        # Initialize Snowflake session
        if SNOWFLAKE_AVAILABLE:
            try:
                self.session = get_active_session()
                st.sidebar.success("❄️ Connected to Snowflake")
            except:
                self.session = None
        else:
            self.session = None
```

#### 4.2 Add Data Loading from Snowflake
```python
def load_model_metrics_from_snowflake(self):
    """Load model metrics from Snowflake tables."""
    if self.session:
        try:
            metrics_df = self.session.sql("""
                SELECT model_name, rmse, mae, r2, mape 
                FROM YIELDCURVE_DB.ML_MODELS.MODEL_METRICS
                WHERE is_active = TRUE
            """).to_pandas()
            
            metrics = {}
            for _, row in metrics_df.iterrows():
                metrics[row['MODEL_NAME']] = {
                    'rmse': row['RMSE'],
                    'mae': row['MAE'],
                    'r2': row['R2'],
                    'mape': row['MAPE']
                }
            return metrics
        except Exception as e:
            st.warning(f"Could not load from Snowflake: {str(e)}")
    
    # Fallback to demo data
    return self._get_demo_metrics()
```

---

## 🔒 **Security Configuration**

### **Role-Based Access Control**

#### 1. Create User Groups
```sql
-- Analysts (read-only access)
CREATE ROLE IF NOT EXISTS YIELDCURVE_ANALYSTS;
GRANT USAGE ON DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_ANALYSTS;
GRANT SELECT ON ALL TABLES IN DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_ANALYSTS;

-- Data Scientists (read/write for models)
CREATE ROLE IF NOT EXISTS YIELDCURVE_DATA_SCIENTISTS;
GRANT USAGE ON DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_DATA_SCIENTISTS;
GRANT ALL PRIVILEGES ON SCHEMA YIELDCURVE_DB.ML_MODELS TO ROLE YIELDCURVE_DATA_SCIENTISTS;

-- Administrators (full access)
CREATE ROLE IF NOT EXISTS YIELDCURVE_ADMINS;
GRANT ALL PRIVILEGES ON DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_ADMINS;
```

#### 2. Configure App Sharing
1. **Navigate to your Streamlit app**
2. **Click "Share" button**
3. **Add users by email or role:**
   - **Viewers:** `YIELDCURVE_ANALYSTS`
   - **Editors:** `YIELDCURVE_DATA_SCIENTISTS`
   - **Owners:** `YIELDCURVE_ADMINS`

---

## ⚡ **Performance Optimization**

### **Warehouse Configuration**

#### 1. Create Dedicated Warehouse
```sql
-- Create warehouse for Streamlit apps
CREATE WAREHOUSE IF NOT EXISTS STREAMLIT_WH
WITH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for YieldCurveAI Streamlit application';

-- Grant usage to appropriate roles
GRANT USAGE ON WAREHOUSE STREAMLIT_WH TO ROLE YIELDCURVE_USERS;
```

#### 2. Optimize Queries
```python
# Use efficient SQL queries in your app
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_latest_features():
    if session:
        return session.sql("""
            SELECT * FROM YIELDCURVE_DB.FEATURES.PROCESSED_FEATURES
            WHERE date >= CURRENT_DATE - INTERVAL '30 DAYS'
            ORDER BY date DESC
        """).to_pandas()
```

---

## 📊 **Data Pipeline Integration**

### **Real-time Data Ingestion**

#### 1. Create Data Pipeline
```sql
-- Create stream for real-time updates
CREATE STREAM feature_stream 
ON TABLE YIELDCURVE_DB.FEATURES.PROCESSED_FEATURES;

-- Create task for automated data processing
CREATE TASK update_features_task
    WAREHOUSE = STREAMLIT_WH
    SCHEDULE = 'USING CRON 0 9 * * * UTC'  -- Daily at 9 AM UTC
AS
    INSERT INTO YIELDCURVE_DB.FEATURES.PROCESSED_FEATURES
    SELECT * FROM external_data_source
    WHERE date = CURRENT_DATE;
```

#### 2. External Data Connections
```sql
-- Connect to external APIs (example)
CREATE EXTERNAL FUNCTION get_fed_data(symbol STRING)
RETURNS STRING
API_INTEGRATION = fred_api_integration
HEADERS = ('Content-Type' = 'application/json')
MAX_BATCH_ROWS = 100;
```

---

## 🚀 **Deployment Checklist**

### **Pre-Deployment**
- [ ] ✅ Snowflake account with appropriate tier
- [ ] ✅ Database and schemas created
- [ ] ✅ User roles and permissions configured
- [ ] ✅ Application code prepared
- [ ] ✅ Data tables created (if needed)

### **Deployment**
- [ ] ✅ Streamlit app created in Snowflake
- [ ] ✅ Code uploaded and tested
- [ ] ✅ Data connections verified
- [ ] ✅ Permissions tested
- [ ] ✅ Performance optimized

### **Post-Deployment**
- [ ] ✅ User access configured
- [ ] ✅ Monitoring set up
- [ ] ✅ Backup procedures established
- [ ] ✅ Documentation updated

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"Permission Denied" Error**
```sql
-- Check and grant necessary permissions
SHOW GRANTS TO ROLE YIELDCURVE_USERS;
GRANT USAGE ON DATABASE YIELDCURVE_DB TO ROLE YIELDCURVE_USERS;
```

#### **"Warehouse Not Found" Error**
```sql
-- Create or assign warehouse
GRANT USAGE ON WAREHOUSE STREAMLIT_WH TO ROLE YIELDCURVE_USERS;
```

#### **Data Not Loading**
```python
# Add error handling in your app
try:
    data = session.sql("SELECT * FROM table").to_pandas()
except Exception as e:
    st.error(f"Data loading error: {str(e)}")
    # Fallback to demo data
```

---

## 📈 **Enterprise Features**

### **Advanced Analytics**
- **Query optimization** with Snowflake's query planner
- **Automatic clustering** for large datasets
- **Time travel** for data versioning
- **Secure data sharing** across organizations

### **Compliance & Governance**
- **Data lineage tracking**
- **Audit logs** for all operations
- **GDPR compliance** features
- **SOX compliance** reporting

### **Integration Capabilities**
- **Partner Connect** for third-party tools
- **REST APIs** for custom integrations
- **Real-time streaming** with Kafka
- **ML model deployment** with Snowpark

---

## 📞 **Support & Resources**

### **Snowflake Resources**
- **Documentation:** [docs.snowflake.com](https://docs.snowflake.com)
- **Community:** [community.snowflake.com](https://community.snowflake.com)
- **Training:** [university.snowflake.com](https://university.snowflake.com)

### **YieldCurveAI Team**
- **Technical Lead:** Mr. Pappu Kapgate ([LinkedIn](https://www.linkedin.com/in/pkapgate))
- **Academic Oversight:** Dr. Kapila Mallah ([Hansraj Profile](https://www.hansrajcollege.ac.in/academics/departments/arts-and-commerce/economics/faculty-detail/64/))
- **Institutional Support:** Dr. Eric Katovai ([SINU Profile](https://www.sinu.edu.sb/executive-governance/vice-chancellor/pro-vice-chancellor-academic/))

---

## 🎯 **Next Steps**

1. **Contact Snowflake** for enterprise account setup
2. **Prepare your data** for migration to Snowflake
3. **Test the deployment** in a sandbox environment
4. **Train your team** on Snowflake features
5. **Deploy to production** with proper monitoring

---

**❄️ Ready to deploy on Snowflake? This enterprise-grade platform will provide the scalability, security, and integration capabilities your YieldCurveAI application needs for institutional use.** 