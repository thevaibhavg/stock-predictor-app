import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Feature Engineering
# -------------------------------
def generate_features(df):
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Daily_Return'] = df['Close'].pct_change()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    return df

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📈 Stock Price & Trend Predictor")

symbol = st.text_input("Enter NSE stock symbol (e.g. HDFCBANK.NS):", value="HDFCBANK.NS")
target_date = st.date_input("📅 Select the target date to predict", value=date.today())

if st.button("Predict"):
    with st.spinner("🔄 Fetching data and preparing input..."):
        try:
            # Step 1: Download and clean data
            df = yf.download(symbol, period="1y", interval="1d", progress=False)

            if df is None or df.empty:
                st.error("❌ Invalid symbol or no data found.")
            else:
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
                df = generate_features(df)
                df = df.dropna()

                # Step 2: Date processing
                target_date_str = target_date.strftime('%Y-%m-%d')
                target_dt = pd.to_datetime(target_date_str)

                # Step 3: Filter data
                df_train = df[df.index < target_dt]
                df_target = df[df.index == target_dt]

                features = ['Open', 'High', 'Low', 'Close', 'Volume',
                            'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']

                # Step 4: Validation
                if df_target.empty:
                    st.error("❌ No market data available for the selected date.")
                elif len(df_train) < 30:
                    st.error("❌ Not enough data to train the model. Try a later date.")
                elif df_train.isnull().values.any() or df_target.isnull().values.any():
                    st.error("❌ Missing values found.")
                elif df_train[features].nunique().min() <= 1:
                    st.error("❌ Not enough variation in training data.")
                else:
                    st.success(f"✅ Data validated. Training on {len(df_train)} records...")

                    # Step 5: Train models
                    df_train = df_train.copy()
                    df_train['Trend'] = np.where(df_train['Close'].shift(-1) > df_train['Close'], 1, 0)

                    X = df_train[features].iloc[:-1]
                    y_reg = df_train['Close'].shift(-1).dropna()
                    y_clf = df_train['Trend'].iloc[:-1]

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)

                    reg_model.fit(X_scaled, y_reg)
                    clf_model.fit(X_scaled, y_clf)

                    # Step 6: Predict
                    X_target = df_target[features]
                    X_target_scaled = scaler.transform(X_target)

                    predicted_price = reg_model.predict(X_target_scaled)[0]
                    predicted_trend = clf_model.predict(X_target_scaled)[0]

                    # Step 7: Display result
                    st.subheader(f"📊 Prediction for {target_date_str}")
                    st.success(f"💰 Predicted Price: ₹{predicted_price:.2f}")
                    st.info(f"📈 Predicted Trend: {'🔺 UP' if predicted_trend == 1 else '🔻 DOWN'}")

                    # Step 8: Show chart
                    st.subheader("📉 Close Price - Last 60 Days")
                    st.line_chart(df['Close'].tail(60))

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
