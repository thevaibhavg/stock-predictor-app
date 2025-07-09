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

# Fix MultiIndex error
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
else:
    df.columns = df.columns.str.strip()

# Define required columns
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

# Step 2: Validate DataFrame
if df is None or df.empty:
    st.error("❌ No data returned. Please check the stock symbol or try again later.")
    st.stop()

# Step 3: Filter only existing required columns
existing_cols = [col for col in required_cols if col in df.columns]
if not existing_cols:
    st.error("❌ None of the required OHLCV columns exist in the dataset.")
    st.write("Returned columns:", list(df.columns))
    st.stop()

try:
    df = df.dropna(subset=existing_cols)
except KeyError as e:
    st.error(f"❌ Unexpected column issue: {e}")
    st.stop()

# Step 4: Feature Engineering
df = generate_features(df)
df = df.dropna()

# Step 5: Date Picker
min_date = df.index.min().date()
max_date = df.index.max().date()
target_date = st.date_input("📅 Select the target date to predict", value=max_date, min_value=min_date, max_value=max_date)

# Step 6: Predict Button
if st.button("Predict"):
    with st.spinner("🔄 Processing..."):
        try:
            target_dt = pd.to_datetime(target_date)
            df_train = df[df.index < target_dt]
            df_target = df[df.index == target_dt]

            features = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']

            # Extra validations
            if df_target.empty:
                st.error("❌ No market data available for the selected date (weekend or holiday).")
            elif df_train.empty or len(df_train) < 30:
                st.error("❌ Not enough historical data to train the model.")
            elif df_train[features].isnull().any().any() or df_target[features].isnull().any().any():
                st.error("❌ Missing values in features. Cannot proceed.")
            elif df_train[features].nunique().min() <= 1:
                st.error("❌ Not enough variation in training data.")
            else:
                # Add trend label
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

                X_target = df_target[features]
                X_target_scaled = scaler.transform(X_target)

                predicted_price = reg_model.predict(X_target_scaled)[0]
                predicted_trend = clf_model.predict(X_target_scaled)[0]

                st.subheader(f"📊 Prediction for {target_date.strftime('%Y-%m-%d')}")
                st.success(f"💰 Predicted Price: ₹{predicted_price:.2f}")
                st.info(f"📈 Predicted Trend: {'🔺 UP' if predicted_trend == 1 else '🔻 DOWN'}")

                st.subheader("📉 Close Price - Last 60 Days")
                st.line_chart(df['Close'].tail(60))

        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
