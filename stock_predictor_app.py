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

# Step 1: Download Data
df = yf.download(symbol, period="1y", interval="1d", progress=False)

required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

if df is None or df.empty:
    st.error("❌ No data returned. Please check the stock symbol or try again later.")
    st.stop()

# Step 2: Validate Columns
if not all(col in df.columns for col in required_cols):
    st.error("❌ Downloaded data is missing required columns. Please try a different stock.")
    st.stop()

# Step 3: Clean & Feature Engineer
df = df.dropna(subset=required_cols)
df = generate_features(df)
df = df.dropna()

# Step 4: Set Date Picker
min_date = df.index.min().date()
max_date = df.index.max().date()
target_date = st.date_input("📅 Select the target date to predict", value=max_date, min_value=min_date, max_value=max_date)

# Step 5: On Predict Button Click
if st.button("Predict"):
    with st.spinner("🔄 Processing..."):
        try:
            target_date_str = target_date.strftime('%Y-%m-%d')
            target_dt = pd.to_datetime(target_date_str)

            df_train = df[df.index < target_dt]
            df_target = df[df.index == target_dt]

            features = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']
            
            # Validation
            missing_cols = [col for col in required_cols if col not in df_target.columns or df_target[col].isnull().any()]

            if df_target.empty or len(df_target) == 0:
                st.error("❌ No market data available for the selected date (weekend or holiday).")
            elif missing_cols:
                st.error(f"❌ Selected date is missing data: {missing_cols}")
            elif len(df_train) < 30:
                st.error("❌ Not enough historical data to train the model.")
            elif df_train.isnull().values.any() or df_target.isnull().values.any():
                st.error("❌ Missing values found in the data.")
            elif df_train[features].nunique().min() <= 1:
                st.error("❌ Not enough variation in training data.")
            else:
                st.success(f"✅ Data validated. Training on {len(df_train)} records...")

                # Training
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

                # Prediction
                X_target = df_target[features]
                X_target_scaled = scaler.transform(X_target)

                predicted_price = reg_model.predict(X_target_scaled)[0]
                predicted_trend = clf_model.predict(X_target_scaled)[0]

                # Display Results
                st.subheader(f"📊 Prediction for {target_date_str}")
                st.success(f"💰 Predicted Price: ₹{predicted_price:.2f}")
                st.info(f"📈 Predicted Trend: {'🔺 UP' if predicted_trend == 1 else '🔻 DOWN'}")

                # Chart
                st.subheader("📉 Close Price - Last 60 Days")
                st.line_chart(df['Close'].tail(60))

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
