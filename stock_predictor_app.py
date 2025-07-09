import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from datetime import datetime, timedelta

# --- PAGE SETUP ---
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📈 Stock Price & Trend Predictor")

# --- RSI CALCULATION ---
def compute_RSI(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- CACHED FEATURE GENERATION ---
@st.cache_data
def generate_features(data):
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    data['RSI_14'] = compute_RSI(data['Close'])
    data['Daily_Return'] = data['Close'].pct_change()
    data['Target_Close'] = data['Close'].shift(-1)
    data['Target_Direction'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    return data

# --- CACHED MODEL TRAINING ---
@st.cache_resource
def train_models(data):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']
    X = data[features]
    y_reg = data['Target_Close']
    y_clf = data['Target_Direction']

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    reg_model.fit(X, y_reg)
    clf_model.fit(X, y_clf)

    return reg_model, clf_model, X.tail(1)

# --- UI ---
symbol = st.text_input("Enter NSE stock symbol (e.g. HDFCBANK.NS):", value="HDFCBANK.NS")
from datetime import date

target_date = st.date_input(
    "📅 Select the target date to predict (from last 180 days)", 
    value=date.today()
)


if st.button("Predict"):
    with st.spinner("🔄 Fetching data and predicting..."):
        try:
            df = yf.download(symbol, period="120d", interval="1d", progress=False)

            if df.empty or len(df) < 50:
                st.warning("⚠️ Not enough data available.")
            else:
                df = generate_features(df)
                reg, clf, last = train_models(df)

                price = reg.predict(last)[0]
                trend = clf.predict(last)[0]

                st.subheader("📊 Prediction Result")
                st.success(f"Predicted Price: ₹{price:.2f}")
                st.info(f"Predicted Trend: {'🔺 UP' if trend == 1 else '🔻 DOWN'}")

                st.subheader("📉 Past Price Chart")
                st.line_chart(df['Close'].tail(60))

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
