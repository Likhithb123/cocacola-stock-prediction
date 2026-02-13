import streamlit as st
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import warnings
import os

# ---------------- WARNINGS (OPTIONAL CLEANUP) ----------------
warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Coca-Cola Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "cocacola_model.pkl"
DATA_PATH = "Coca-Cola_stock_history.csv"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Train model on Streamlit Cloud
    df = pd.read_csv(DATA_PATH)

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)


# ---------------- LOAD IMAGES ----------------
banner = Image.open("assets/banner.jpg")
chart = Image.open("assets/chart.jpg")
bottle = Image.open("assets/bottle.jpg")

# ---------------- SESSION STATE ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# ---------------- HEADER (CONTROLLED BANNER WIDTH) ----------------
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.image(banner, width=900)

st.markdown(
    """
    <h1 style='text-align:center; color:#b11226;'>📈 Coca-Cola Stock Price Prediction</h1>
    <p style='text-align:center; font-size:17px; color:gray;'>
    Predict the closing price of Coca-Cola stock using market indicators
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.1, 1])

# ---------------- LEFT: INPUTS ----------------
with left:
    st.markdown("### 📊 Market Inputs")

    open_price = st.number_input("📊 Open Price (USD)", min_value=0.0)
    high_price = st.number_input("📈 High Price (USD)", min_value=0.0)
    low_price = st.number_input("📉 Low Price (USD)", min_value=0.0)
    volume = st.number_input("🔄 Trading Volume", min_value=0)

    if st.button("🔮 Predict Closing Price", use_container_width=True):
        # ✅ IMPORTANT: DataFrame with SAME feature names as training
        input_data = pd.DataFrame({
            "Open": [open_price],
            "High": [high_price],
            "Low": [low_price],
            "Volume": [volume]
        })

        st.session_state.prediction = model.predict(input_data)[0]

    # ✅ RESULT DISPLAY (NO FADING)
    if st.session_state.prediction is not None:
        st.success(
            f"💰 Predicted Closing Price: **${st.session_state.prediction:.2f}**"
        )

# ---------------- RIGHT: VISUALS ----------------
with right:
    st.markdown("### 📉 Market Insights")

    st.image(chart, caption="Stock Market Trend Visualization", use_container_width=True)

    # Centered bottle image (Streamlit-safe)
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        st.image(bottle, width=220, caption="Coca-Cola Brand Asset")

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built with ❤️ using Machine Learning & Streamlit | Finance Analytics Project")
