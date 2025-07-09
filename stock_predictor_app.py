# stock_predictor_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Feature Engineering Function
# -------------------------------
def generate_features(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    df = df.dropna()
    return df

# -------------------------------
# Model Training Function
# -------------------------------
def train_models(df):
    df = df.copy()
    df['Trend'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']
    X = df[features]
    y_reg = df['Close'].shift(-1).dropna()
    y_clf = df['Trend'].iloc[:-1]  # align

    X = X.iloc[:-1]  # align

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    reg.fit(X_scaled, y_reg)
    clf.fit(X_scaled, y_clf)

    return reg, clf, scaler

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📈 Stock Price & Trend Predictor")

symbol = st.text_input("Enter NSE stock symbol (e.g. HDFCBANK.NS):", value="HDFCBANK.NS")
target_date = st.date_input("📅 Select the target date to predict", value=date.today())

if st.button("Predict"):
    with st.spinner("🔄 Fetching data and predicting..."):
        try:
            # Download 1 year of data
            df = yf.download(symbol, period="1y", interval="1d", progress=False)

            if df.empty:
                st.warning("⚠️ No data available for this symbol.")
            else:
                df = generate_features(df)

                # Format date
                target_date_str = target_date.strftime('%Y-%m-%d')
                target_dt = pd.to_datetime(target_date_str)

                # Split data
                df_train = df[df.index < target_dt]
                df_target = df[df.index == target_dt]

                if df_target.empty:
                    st.error("❌ Data for selected date not available.")
                elif len(df_train) < 30:
                    st.error("❌ Not enough past data to train the model.")
                else:
                    # Train models on past data only
                    reg_model, clf_model, scaler = train_models(df_train)

                    # Prepare target features
                    X_target = df_target[['Open', 'High', 'Low', 'Close', 'Volume',
                                          'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']]
                    X_target_scaled = scaler.transform(X_target)

                    # Predict
                    predicted_price = reg_model.predict(X_target_scaled)[0]
                    predicted_trend = clf_model.predict(X_target_scaled)[0]

                    # Output
                    st.subheader(f"📊 Prediction for {target_date_str}")
                    st.success(f"Predicted Price: ₹{predicted_price:.2f}")
                    st.info(f"Predicted Trend: {'🔺 UP' if predicted_trend == 1 else '🔻 DOWN'}")

                    # Optional Chart
                    st.subheader("📉 Price Chart (Last 60 Days)")
                    st.line_chart(df['Close'].tail(60))

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
