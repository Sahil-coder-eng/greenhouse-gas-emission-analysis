import streamlit as st
import joblib
import numpy as np
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# 🎯 Page Configuration
st.set_page_config(page_title="Emission Predictor", layout="wide")

# 🎨 Custom Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #2193b0;
            background-image: linear-gradient(to right, #6dd5ed, #2193b0);
            background-size: cover;
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        .stButton>button {
            background-color: #9c4dcc;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #7b35a9;
        }
    </style>
""", unsafe_allow_html=True)

# 💾 Load model and scaler
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("🌍 Supply Chain Emission Factor Estimator")
st.markdown("Use this interactive tool to estimate supply chain emission factors based on quality metrics.")

# 🎛️ Sidebar Input
st.sidebar.header("🔧 Source Configuration")
source_type = st.sidebar.radio("Select Source Type", ["Commodity", "Industry"])
source_value = 0 if source_type == "Commodity" else 1

combined_dq_live = np.mean([0.7, 0.6, 0.65, 0.55, 0.50])
st.sidebar.metric("📈 Live DQ Score", f"{combined_dq_live:.2f}")
st.sidebar.markdown("---")

# 📋 Input Form
with st.form("prediction_form"):
    st.subheader("🧮 Input Feature Values")
    col1, col2 = st.columns(2)

    with col1:
        f1 = st.number_input("Emission Factors Without Margins", min_value=0.0, max_value=10.0, value=0.50, step=0.01)
        f2 = st.number_input("Emission Factor Margins", min_value=0.0, max_value=5.0, value=0.25, step=0.01)
        dq_reliability = st.slider("DQ: Reliability 🔒", 0.0, 1.0, 0.7, 0.01)

    with col2:
        dq_temporal = st.slider("DQ: Temporal Correlation 🕒", 0.0, 1.0, 0.6, 0.01)
        dq_geo = st.slider("DQ: Geographical Correlation 🌍", 0.0, 1.0, 0.65, 0.01)
        dq_tech = st.slider("DQ: Technological Correlation ⚙️", 0.0, 1.0, 0.55, 0.01)
        dq_data = st.slider("DQ: Data Collection 📊", 0.0, 1.0, 0.50, 0.01)

    submitted = st.form_submit_button("🎯 Predict Emission Factor")

# 🚀 Prediction Logic
if submitted:
    combined_dq = np.mean([dq_reliability, dq_temporal, dq_geo, dq_tech, dq_data])
    input_data = np.array([[f1, f2, dq_reliability, dq_temporal, dq_geo, dq_tech, dq_data, combined_dq, source_value]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # 🌈 Color & Feedback
    if prediction < 1.0:
        color = "green"
        feedback = "🌟 **Low Emission!** Great job!"
    elif prediction < 2.0:
        color = "orange"
        feedback = "⚠️ **Moderate Emission** – review your sources."
    else:
        color = "red"
        feedback = "🔥 **High Emission Warning!** Consider optimization."

    st.markdown(f"<h3 style='color:{color}'>🔢 Estimated Emission Factor: {prediction:.4f}</h3>", unsafe_allow_html=True)
    st.markdown(feedback)
    st.progress(min(prediction / 2.0, 1.0))

    # 🟣 Updated Composite Score Display
    st.markdown(
        f"""
        <div style="margin-top: 20px; font-weight: bold; font-size: 18px; background-color: rgba(255,255,255,0.1); padding: 8px; border-radius: 8px;">
            📊 DQ Composite Score: <span style="color: white;">{combined_dq:.2f}</span> 
            <span style="color: #cccccc; font-size: 14px;">↗️ Derived from input sliders</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 📊 Plotly DQ Chart
    dq_scores = {
        "Reliability": dq_reliability,
        "Temporal": dq_temporal,
        "Geographical": dq_geo,
        "Technological": dq_tech,
        "Data Quality": dq_data,
    }
    fig = px.bar(
        x=list(dq_scores.keys()),
        y=list(dq_scores.values()),
        color=list(dq_scores.values()),
        color_continuous_scale='Blues',
        labels={'x': 'DQ Factors', 'y': 'Score'},
        title="📈 DQ Breakdown"
    )
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # 🎞️ Lottie Animation
    def load_lottie(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_anim = load_lottie("https://assets9.lottiefiles.com/packages/lf20_x62chJ.json")
    if lottie_anim:
        st_lottie(lottie_anim, height=150, key="animation")
    else:
        st.info("🎬 Animation could not be loaded at the moment.")

    st.success("✅ Prediction complete! Adjust inputs to explore further.")
         # 📄 Generate a text report
    