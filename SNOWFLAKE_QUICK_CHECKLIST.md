# ✅ Snowflake Deployment Quick Checklist

**Print this checklist and check off each step as you complete it!**

---

## 🎯 **Phase 1: Account Setup (15 minutes)**

- [ ] **1.1** Go to [signup.snowflake.com](https://signup.snowflake.com)
- [ ] **1.2** Choose **Standard** edition, **AWS** provider  
- [ ] **1.3** Fill registration form with your details
- [ ] **1.4** Verify email and create password
- [ ] **1.5** Login to your Snowflake account
- [ ] **1.6** Note your account URL for later

---

## 🎯 **Phase 2: Database Setup (10 minutes)**

- [ ] **2.1** Click **"Worksheets"** in left sidebar
- [ ] **2.2** Click **"+ Worksheet"** button
- [ ] **2.3** Copy-paste **Block 1** SQL code (database creation)
- [ ] **2.4** Click **"Run"** and wait for green checkmarks ✅
- [ ] **2.5** Copy-paste **Block 2** SQL code (model metrics table)
- [ ] **2.6** Click **"Run"** and wait for green checkmarks ✅
- [ ] **2.7** Copy-paste **Block 3** SQL code (features table)
- [ ] **2.8** Click **"Run"** and wait for green checkmarks ✅
- [ ] **2.9** Run verification query to check everything worked

---

## 🎯 **Phase 3: App Deployment (10 minutes)**

- [ ] **3.1** Click **"Streamlit"** in left sidebar
- [ ] **3.2** Click **"+ Streamlit App"** button
- [ ] **3.3** Fill out app configuration:
  - App Name: `YieldCurveAI`
  - Database: `YIELDCURVE_DB`
  - Schema: `STREAMLIT_APPS`
- [ ] **3.4** Create warehouse if needed (`STREAMLIT_WH`, X-Small)
- [ ] **3.5** Open `YieldCurveAI_Snowflake.py` file
- [ ] **3.6** Copy ALL code (Ctrl+A, Ctrl+C)
- [ ] **3.7** Paste into Snowflake editor (Ctrl+V)
- [ ] **3.8** Click **"Create"** button
- [ ] **3.9** Wait 1-2 minutes for deployment
- [ ] **3.10** Click **"Open App"** when ready

---

## 🎯 **Phase 4: Testing (10 minutes)**

- [ ] **4.1** Test **Team & Oversight** tab
  - [ ] See all 3 team profiles
  - [ ] Click "View Details" for each
- [ ] **4.2** Test **Enterprise Forecast** tab
  - [ ] Change Fed Funds Rate to 5.5%
  - [ ] Change CPI to 3.0%
  - [ ] Click "Generate Enterprise Forecast"
  - [ ] See yield curve chart and table
- [ ] **4.3** Test **Model Analytics** tab
  - [ ] See model performance table
  - [ ] See best model highlighted
- [ ] **4.4** Check Snowflake integration indicators:
  - [ ] "❄️ Snowflake Connected" badge in sidebar
  - [ ] "Enterprise" labels throughout app
  - [ ] Professional styling

---

## 🎯 **Phase 5: Sharing & Documentation (5 minutes)**

- [ ] **5.1** Go back to **"Streamlit"** section in Snowflake
- [ ] **5.2** Copy your app URL (looks like: `https://abc123.snowflakecomputing.com/streamlit/apps/YIELDCURVEAI`)
- [ ] **5.3** Test URL in new browser tab
- [ ] **5.4** Document URL for team sharing
- [ ] **5.5** Share with Dr. Kapila Mallah and Dr. Eric Katovai

---

## 🚨 **Common Issues & Quick Fixes**

### **❌ "Permission Denied"**
```sql
USE ROLE ACCOUNTADMIN;
GRANT USAGE ON DATABASE YIELDCURVE_DB TO ROLE PUBLIC;
GRANT USAGE ON ALL SCHEMAS IN DATABASE YIELDCURVE_DB TO ROLE PUBLIC;
GRANT SELECT ON ALL TABLES IN DATABASE YIELDCURVE_DB TO ROLE PUBLIC;
```

### **❌ "Warehouse Not Found"**
- Go to **Admin → Warehouses**
- Create: `STREAMLIT_WH`, X-Small size

### **❌ App Won't Load**
- Check warehouse is running (Admin → Warehouses)
- Check database exists (Data → Databases)
- Look for syntax errors in Streamlit logs

---

## 📋 **Final Success Checklist**

- [ ] ✅ App loads without errors
- [ ] ✅ All 3 navigation tabs work
- [ ] ✅ Forecast generates yield curve chart
- [ ] ✅ Team profiles display correctly
- [ ] ✅ Model analytics show performance metrics
- [ ] ✅ Snowflake branding appears throughout
- [ ] ✅ App URL accessible from external browser
- [ ] ✅ Team members notified with URL

---

## 🎉 **Congratulations!**

**Your YieldCurveAI is now live on Snowflake enterprise infrastructure!**

### **Your app URL:** `______________________________`

### **Share this with your team:**
- Dr. Kapila Mallah: Economic AI design
- Dr. Eric Katovai: Academic oversight  
- SINU stakeholders: Institutional showcase

---

**🚀 You've successfully deployed an enterprise-grade application!** 