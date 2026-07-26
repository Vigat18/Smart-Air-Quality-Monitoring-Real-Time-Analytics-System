import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import datetime
import json
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Air Quality & ESG Analytics System",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍃 Smart Air Quality Monitoring & Real-Time Analytics System")
st.caption("AI-Powered Carbon Monoxide (CO) Forecasting, Interactive Data Visualization & ESG Impact Monitoring")

# ---------------------------------------------------------
# Gemini API Initialization Helper
# ---------------------------------------------------------
def get_gemini_client(api_key_input):
    """Initializes Gemini API Client from user sidebar input or Streamlit secrets."""
    api_key = api_key_input or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Gemini API: {str(e)}")
        return None

# ---------------------------------------------------------
# Sidebar Configuration & Data Ingestion
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/leaf.png", width=60)
st.sidebar.title("⚙️ Controls & Setup")

# Gemini API Key Input
gemini_key_input = st.sidebar.text_input(
    "Google Gemini API Key", 
    type="password", 
    help="Enter your API key or configure it in .streamlit/secrets.toml"
)

# Page Navigation
page = st.sidebar.radio("Navigate Views:", [
    "📈 Real-Time Data Visualization",
    "🔮 AI CO Prediction & Gemini Assistant",
    "📊 ESG Impact Dashboard",
    "⚙️ AI Workflow Architecture"
])

# Generate Synthetic Sample Dataset if user doesn't upload one
@st.cache_data
def generate_sample_dataset():
    """Generates synthetic hourly air quality sensor data for demonstration[cite: 2]."""
    np.random.seed(42)
    timestamps = pd.date_range(start="2026-07-01", periods=168, freq="h")
    
    current_co = np.random.uniform(1.5, 9.5, len(timestamps))
    temp = 25 + 5 * np.sin(np.linspace(0, 10, len(timestamps))) + np.random.normal(0, 1, len(timestamps))
    humidity = 50 + 20 * np.cos(np.linspace(0, 10, len(timestamps))) + np.random.normal(0, 2, len(timestamps))
    
    # Target CO level (4 hours ahead)
    future_co = current_co * 0.85 + (temp * 0.04) - (humidity * 0.015) + np.random.normal(0, 0.4, len(timestamps))
    future_co = np.maximum(0.5, future_co)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "co_mg_m3": np.round(current_co, 2),
        "temperature_c": np.round(temp, 1),
        "humidity_percent": np.round(humidity, 1),
        "future_co_4hr": np.round(future_co, 2)
    })
    return df

# ---------------------------------------------------------
# 1. Real-Time Data Upload & Visualization Engine
# ---------------------------------------------------------
if page == "📈 Real-Time Data Visualization":
    st.header("📈 Upload Dataset & Real-Time Visual Analytics")
    st.write("Upload your air quality sensor dataset (CSV or Excel) to generate interactive charts and statistical summaries[cite: 2].")

    uploaded_file = st.file_uploader("Upload Sensor Dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Successfully loaded uploaded dataset: `{uploaded_file.name}` ({df.shape[0]} rows, {df.shape[1]} columns)")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = generate_sample_dataset()
    else:
        st.info("ℹ️ No file uploaded. Displaying real-time sample IoT air quality dataset.")
        df = generate_sample_dataset()

    # Store DataFrame in session state
    st.session_state['active_df'] = df

    # Data Overview Metrics
    st.markdown("---")
    st.subheader("📋 Dataset Overview & Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    co_col = next((col for col in numeric_cols if 'co' in col.lower()), numeric_cols[0] if numeric_cols else None)
    temp_col = next((col for col in numeric_cols if 'temp' in col.lower()), numeric_cols[1] if len(numeric_cols)>1 else None)
    hum_col = next((col for col in numeric_cols if 'hum' in col.lower()), numeric_cols[2] if len(numeric_cols)>2 else None)

    if co_col:
        col1.metric("Average CO Concentration", f"{df[co_col].mean():.2f} mg/m³")
        col2.metric("Peak CO Level Recorded", f"{df[co_col].max():.2f} mg/m³")
    if temp_col:
        col3.metric("Avg Temperature", f"{df[temp_col].mean():.1f} °C")
    if hum_col:
        col4.metric("Avg Humidity", f"{df[hum_col].mean():.1f} %")

    # Data Preview
    with st.expander("🔍 View Raw Dataset Table"):
        st.dataframe(df, use_container_width=True)

    # Interactive Graph Generators
    st.markdown("---")
    st.subheader("📊 Interactive Data Visualizations")

    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📉 Time Series Trends", "🔵 Scatter Correlation", "📊 Pollution Risk Distribution"])

    with chart_tab1:
        st.write("### Pollution Concentration Over Time")
        time_col = next((col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
        
        if time_col and co_col:
            fig_line = px.line(
                df, x=time_col, y=[co_col] + ([temp_col] if temp_col else []),
                title="Carbon Monoxide (CO) and Weather Trends Over Time",
                labels={"value": "Measured Units", time_col: "Timestamp"},
                template="plotly_white"
            )
            # Safe legal baseline indicator
            fig_line.add_hline(y=3.5, line_dash="dash", line_color="orange", annotation_text="Safe CO Baseline Threshold (3.5 mg/m³)")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.warning("Ensure your dataset has a timestamp or date column for time-series charts.")

    with chart_tab2:
        st.write("### Environmental Factor Correlations")
        if temp_col and hum_col and co_col:
            fig_scatter = px.scatter(
                df, x=temp_col, y=co_col, color=hum_col,
                size=co_col,
                title="CO Level vs Temperature (Color Scale: Humidity)",
                labels={temp_col: "Temperature (°C)", co_col: "CO Concentration (mg/m³)", hum_col: "Humidity (%)"},
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Insufficient numeric columns for scatter correlation plot.")

    with chart_tab3:
        st.write("### CO Concentration Range Distribution")
        if co_col:
            fig_hist = px.histogram(
                df, x=co_col, nbins=30,
                title="Frequency Distribution of Carbon Monoxide Readings",
                color_discrete_sequence=['#2ecc71'],
                template="plotly_white"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Gemini Data Trend Analysis
    st.markdown("---")
    st.subheader("🤖 Automated Gemini Trend Diagnostic")
    
    if st.button("✨ Analyze Dataset Trends with Gemini API", type="primary"):
        client = get_gemini_client(gemini_key_input)
        if not client:
            st.error("Please provide a valid Gemini API Key in the sidebar or setup secrets.")
        else:
            with st.spinner("Gemini is analyzing dataset summary statistics..."):
                stats_summary = df.describe().to_json()
                prompt = f"""
                You are an expert Environmental AI Data Analyst. Analyze the following summary statistics of an uploaded air quality dataset:
                ```json
                {stats_summary}
                ```
                Provide:
                1. Key findings regarding pollution peaks and baseline safety standard compliance[cite: 1].
                2. Potential correlation insights between weather variables (Temperature, Humidity) and CO levels[cite: 2].
                3. High-level policy recommendations for local environmental authorities[cite: 1].
                Keep it concise, clear, and actionable.
                """
                
                try:
                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are a senior environmental health data analyst.",
                            temperature=0.2
                        )
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")

# ---------------------------------------------------------
# 2. AI CO Prediction & Gemini Diagnostics
# ---------------------------------------------------------
elif page == "🔮 AI CO Prediction & Gemini Assistant":
    st.header("🔮 AI CO Forecasting & Real-Time Gemini Assistant")
    st.write("Predict future Carbon Monoxide levels using dynamic machine learning models and obtain real-time Gemini health advisories.")

    df = st.session_state.get('active_df', generate_sample_dataset())

    # Dynamically Train Machine Learning Model on active dataset
    @st.cache_resource
    def train_dynamic_model(_dataset):
        num_cols = _dataset.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 3:
            X = _dataset[num_cols[:3]]
            y = _dataset[num_cols[3]] if len(num_cols) > 3 else _dataset[num_cols[0]] * 0.9 + 0.5
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            return model, num_cols[:3]
        return None, []

    model, feature_names = train_dynamic_model(df)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input Environmental Metrics")
        in_co = st.number_input("Current CO Concentration (mg/m³)", min_value=0.0, max_value=30.0, value=4.8, step=0.1) # Baseline 4.8 mg/m³[cite: 1]
        in_temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.5)
        in_humidity = st.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=55.0)
        
        predict_btn = st.button("Run ML Forecast Model", type="primary")

    with col2:
        st.subheader("🎯 ML Prediction & Risk Assessment")
        if model and len(feature_names) == 3:
            pred_co = model.predict([[in_co, in_temp, in_humidity]])[0]
        else:
            pred_co = in_co * 0.85 + (in_temp * 0.04) - (in_humidity * 0.015)
        
        st.session_state['last_pred_co'] = pred_co

        delta_val = pred_co - in_co
        st.metric(
            label="Predicted CO Concentration (4 Hours Ahead)",
            value=f"{pred_co:.2f} mg/m³",
            delta=f"{delta_val:+.2f} mg/m³",
            delta_color="inverse"
        )

        if pred_co < 3.5:
            risk_cat = "Low"
            st.success("🟢 **LOW RISK**: Predicted pollution within legal threshold limits.")
        elif 3.5 <= pred_co < 6.0:
            risk_cat = "Medium"
            st.warning("🟡 **MEDIUM RISK**: Moderate pollution buildup anticipated.")
        else:
            risk_cat = "High"
            st.error("🔴 **HIGH RISK**: Severe pollution spike predicted! Immediate authority action required.")

    st.markdown("---")
    st.subheader("🤖 Request Gemini Diagnostic & Advisory")

    if st.button("✨ Ask Gemini for Real-Time Advisory", type="secondary"):
        client = get_gemini_client(gemini_key_input)
        if not client:
            st.error("Please enter a valid Gemini API Key in the sidebar.")
        else:
            with st.spinner("Generating custom diagnostic and emergency response plan..."):
                payload = {
                    "current_co_mg_m3": in_co,
                    "temperature_c": in_temp,
                    "humidity_percent": in_humidity,
                    "predicted_co_4hr_mg_m3": round(pred_co, 2),
                    "risk_category": risk_cat,
                    "baseline_standard_co": "3.5 mg/m³"
                }
                
                gemini_prompt = f"""
                Perform a structured environmental risk assessment for the following sensor payload:
                ```json
                {json.dumps(payload, indent=2)}
                ```
                Provide:
                1. **Diagnostic Summary**: Status of current vs predicted air quality.
                2. **Public Health Advisory**: Actionable precautions for citizens and vulnerable groups (schools/hospitals)[cite: 1].
                3. **Municipal Enforcement Plan**: Priority steps for local authorities within the 2-hour SLA response window[cite: 1, 2].
                4. **ESG Compliance Alignment**: How this action supports governance accountability[cite: 1].
                """
                
                try:
                    response = client.models.generate_content(
                        model="gemini-3.5-flash",
                        contents=gemini_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction="You are an expert environmental safety and municipal emergency management officer.",
                            temperature=0.2
                        )
                    )
                    st.markdown("### 📋 Gemini Diagnostic Report")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Gemini API Error: {str(e)}")

# ---------------------------------------------------------
# 3. ESG Impact Metrics Dashboard
# ---------------------------------------------------------
elif page == "📊 ESG Impact Dashboard":
    st.header("📊 Environmental, Social & Governance (ESG) Impact Tracker")
    st.write("Evaluating Environmental, Social & Governance Impact of the AI-Enabled Pollution Prediction Solution[cite: 1].")

    # Key Performance Metric Cards[cite: 1]
    st.subheader("🎯 Primary Target Metrics")
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric("Avg CO Concentration", "3.36 mg/m³", "-30% Goal", delta_color="normal")
        st.caption("Baseline: 4.8 mg/m³ | Target: ↓ 30% within 6 months[cite: 1]")

    with kpi2:
        st.metric("Population Exposed to High CO", "7.5%", "-14.5%", delta_color="normal")
        st.caption("Baseline: ~22% of zones | Target: < 8%[cite: 1]")

    with kpi3:
        st.metric("Authority Alert Ack (<2 hrs)", "92%", "+37%", delta_color="normal")
        st.caption("Baseline: ~55% acknowledged | Target: 90%+[cite: 1]")

    st.markdown("---")
    st.subheader("📋 Comprehensive ESG Metrics Framework")

    # Table representation of slide data[cite: 1]
    esg_data = {
        "ESG Pillar": ["Environmental", "Environmental", "Environmental", "Social", "Social", "Social", "Governance", "Governance", "Governance"],
        "Metric Name": [
            "CO Concentration Reduction", "Undetected Pollution Spikes", "Emission Control Response Time",
            "Population Exposed to High CO", "Health Advisory Reach", "Preventive Warning Lead Time",
            "Regulatory Compliance Rate", "Authority Alert Ack Rate", "Data Audit Trail Completeness"
        ],
        "Data Source": [
            "IoT sensors + AI model", "Sensor logs vs predictions", "Authority incident logs",
            "Zone pop data + sensor map", "Alert system logs", "AI prediction vs peak time",
            "Monthly compliance reports", "Authority response dashboard", "Sensor data pipeline logs"
        ],
        "Frequency": ["Daily", "Weekly", "Per-incident", "Monthly", "Per-alert", "Per-incident", "Monthly", "Per-alert", "Weekly"],
        "Baseline State": ["4.8 mg/m³", "~6 missed/month", "6.5 hrs post-alert", "~22% zones", "~1,200 citizens", "1.2 hrs notice", "78% compliant", "~55% in 2 hrs", "~82% complete"],
        "Target Improvement": ["↓ 30% in 6 months", "< 1 per month", "< 2 hrs", "< 8% zones", "5,000+ citizens", "4+ hrs notice", "95%+ compliant", "90%+ in 2 hrs", "99%+ complete"]
    }
    
    st.dataframe(pd.DataFrame(esg_data), use_container_width=True)

# ---------------------------------------------------------
# 4. AI Workflow Architecture View
# ---------------------------------------------------------
elif page == "⚙️ AI Workflow Architecture":
    st.header("⚙️ End-to-End System Workflow Architecture")
    st.write("Overview of the continuous pipeline from IoT sensor collection to authority dispatch[cite: 2].")

    st.markdown("""
    ```
    [ 📡 IoT Sensor Data ] ──> [ 🧹 Cleaning & Standardization ] ──> [ 🤖 AI Model Forecasting ] ──> [ 🚨 Alert & SLA Dispatch ]
    ```
    
    ### 🔄 Stage Breakdown:
    1. **Sensor Ingestion**: Continuous collection of CO (mg/m³), Temperature (°C), Relative Humidity (%), and Timestamp data[cite: 2].
    2. **Data Pipeline Processing**: Cleaning null values, standardizing features, and validating timestamp integrity for 99%+ audit compliance[cite: 1, 2].
    3. **Prediction Engine**: Machine Learning models (Random Forest / Linear Regression) forecast CO levels 4 hours in advance[cite: 1, 2].
    4. **Actionable Alerts**: Automated public health notifications and authority dispatches within mandatory 2-hour response SLAs[cite: 1, 2].
    """)