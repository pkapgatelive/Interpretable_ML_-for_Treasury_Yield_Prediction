# 🚀 YieldCurveAI Deployment Guide

This guide covers multiple deployment options for hosting your YieldCurveAI Streamlit application.

## 📋 **Table of Contents**
- [Streamlit Community Cloud](#streamlit-community-cloud) (Recommended)
- [Heroku](#heroku)
- [Google Cloud Platform](#google-cloud-platform)
- [AWS](#aws)
- [Azure](#azure)
- [Local Development](#local-development)

---

## 🎯 **Streamlit Community Cloud** (Recommended)

**Perfect for:** Academic projects, demos, public applications
**Cost:** Free
**Ease:** ⭐⭐⭐⭐⭐

### Prerequisites
1. GitHub repository with your code
2. Streamlit Community Cloud account

### Step-by-Step Deployment

#### 1. Prepare Your Repository
```bash
# Ensure your repo has these essential files:
YieldCurveAI.py                 # Main app file
requirements.txt               # Dependencies
config/profiles.yaml          # Team profiles (already created)
static/images/team/           # Team images (already created)
```

#### 2. Create Essential Files

**`.streamlit/config.toml`** (Already exists, but verify):
```toml
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#1f4e79"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f8f9fa"
textColor = "#262730"
```

**`requirements.txt`** (Minimal for deployment):
```txt
streamlit>=1.28.0
pandas>=1.3.0
numpy>=1.21.0
plotly>=5.6.0
scikit-learn>=1.0.0
PyYAML>=6.0
pathlib
```

#### 3. Deploy to Streamlit Cloud

1. **Login:** Go to [share.streamlit.io](https://share.streamlit.io)
2. **Connect GitHub:** Link your GitHub account
3. **Select Repository:** Choose your yield-curve-forecasting repo
4. **Configure:**
   - **Main file path:** `YieldCurveAI.py`
   - **Python version:** 3.9
5. **Deploy:** Click "Deploy!"

#### 4. Handle Data Files
Since your app needs trained models and data files, you have options:

**Option A: Mock Data Mode**
```python
# Add to YieldCurveAI.py
def load_demo_data():
    """Load demonstration data when real data isn't available."""
    # Create synthetic data for demo purposes
    return demo_features, demo_metrics
```

**Option B: Cloud Storage**
```python
# Add cloud storage integration
import requests

@st.cache_data
def load_models_from_cloud():
    """Load models from cloud storage like GitHub releases or S3."""
    try:
        # Download from GitHub releases or cloud storage
        model_url = "https://github.com/your-repo/releases/download/v1.0/models.zip"
        # Implementation for downloading and extracting
    except:
        st.warning("Using demo mode - upload your trained models for full functionality")
```

---

## 🌐 **Heroku**

**Perfect for:** Production applications, custom domains
**Cost:** $7+/month
**Ease:** ⭐⭐⭐⭐

### Deployment Steps

#### 1. Install Heroku CLI
```bash
# macOS
brew tap heroku/brew && brew install heroku

# Or download from heroku.com
```

#### 2. Create Heroku Files

**`Procfile`**:
```
web: streamlit run YieldCurveAI.py --server.port=$PORT --server.address=0.0.0.0
```

**`runtime.txt`**:
```
python-3.9.18
```

**`setup.sh`**:
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

#### 3. Deploy
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-yieldcurve-app

# Configure environment
heroku config:set \
  STREAMLIT_THEME_PRIMARY_COLOR="#1f4e79" \
  STREAMLIT_THEME_BACKGROUND_COLOR="#ffffff"

# Deploy
git add .
git commit -m "Deploy YieldCurveAI to Heroku"
git push heroku main

# Open app
heroku open
```

---

## ☁️ **Google Cloud Platform**

**Perfect for:** Enterprise applications, integration with Google services
**Cost:** Pay-as-you-go
**Ease:** ⭐⭐⭐

### Using Cloud Run

#### 1. Create `Dockerfile`
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD streamlit run YieldCurveAI.py \
    --server.address=0.0.0.0 \
    --server.port=8080 \
    --server.headless=true
```

#### 2. Deploy to Cloud Run
```bash
# Install Google Cloud CLI
curl https://sdk.cloud.google.com | bash

# Initialize project
gcloud init
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/yieldcurve-ai
gcloud run deploy yieldcurve-ai \
    --image gcr.io/YOUR_PROJECT_ID/yieldcurve-ai \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

---

## 🔶 **AWS**

**Perfect for:** Enterprise applications, AWS ecosystem integration
**Cost:** Pay-as-you-go
**Ease:** ⭐⭐

### Using AWS App Runner

#### 1. Create `apprunner.yaml`
```yaml
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  runtime-version: 3.9.18
  command: streamlit run YieldCurveAI.py --server.port=8000 --server.address=0.0.0.0
  network:
    port: 8000
  env:
    - name: STREAMLIT_SERVER_HEADLESS
      value: "true"
```

#### 2. Deploy via AWS Console
1. Go to AWS App Runner in AWS Console
2. Create service from source code
3. Connect your GitHub repository
4. Configure build settings using `apprunner.yaml`
5. Deploy!

---

## 🔷 **Azure**

**Perfect for:** Microsoft ecosystem integration
**Cost:** Pay-as-you-go
**Ease:** ⭐⭐⭐

### Using Azure Container Instances

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login and create resource group
az login
az group create --name yieldcurve-rg --location eastus

# Create container instance
az container create \
    --resource-group yieldcurve-rg \
    --name yieldcurve-ai \
    --image your-dockerhub/yieldcurve:latest \
    --dns-name-label yieldcurve-ai \
    --ports 8501
```

---

## 🛠️ **Local Development**

For testing before deployment:

```bash
# Clone your repository
git clone your-repo-url
cd yield-curve-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run YieldCurveAI.py
```

---

## 📊 **Deployment Comparison**

| Platform | Cost | Ease | Best For | Custom Domain |
|----------|------|------|----------|---------------|
| **Streamlit Cloud** | Free | ⭐⭐⭐⭐⭐ | Demos, Academic | ❌ |
| **Heroku** | $7+/month | ⭐⭐⭐⭐ | Small Production | ✅ |
| **Google Cloud** | Pay-per-use | ⭐⭐⭐ | Enterprise | ✅ |
| **AWS** | Pay-per-use | ⭐⭐ | Enterprise | ✅ |
| **Azure** | Pay-per-use | ⭐⭐⭐ | Microsoft Stack | ✅ |

---

## 🔧 **Production Considerations**

### 1. Environment Variables
Create `.streamlit/secrets.toml` for sensitive data:
```toml
[api_keys]
fred_api_key = "your-fred-api-key"
bloomberg_key = "your-bloomberg-key"

[database]
connection_string = "your-db-connection"
```

### 2. Model Storage
For production deployments with large model files:

**Option 1: Git LFS**
```bash
git lfs track "*.pkl"
git lfs track "*.joblib"
git add .gitattributes
```

**Option 2: Cloud Storage**
```python
# Download models at runtime
@st.cache_data
def download_models():
    import requests
    model_url = "https://your-storage-url/models.zip"
    # Download and extract logic
```

### 3. Performance Optimization
```python
# Add to YieldCurveAI.py
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_computation():
    # Your model predictions
    pass

# Use session state for user preferences
if 'user_settings' not in st.session_state:
    st.session_state.user_settings = {}
```

---

## 🚨 **Troubleshooting**

### Common Issues:

**"Module not found" errors:**
```bash
# Add missing dependencies to requirements.txt
echo "missing-package>=1.0.0" >> requirements.txt
```

**App won't start:**
```bash
# Check logs in deployment platform
# Ensure main file path is correct
# Verify Python version compatibility
```

**Large app size:**
```bash
# Reduce model file sizes
# Use model compression
# Implement lazy loading
```

---

## 📞 **Support**

For deployment issues:
1. Check platform-specific documentation
2. Review application logs
3. Test locally first
4. Use minimal requirements.txt for initial deployment

---

**🎯 Recommendation:** Start with **Streamlit Community Cloud** for quick deployment, then move to **Heroku** or **Google Cloud Run** for production use with custom domains. 