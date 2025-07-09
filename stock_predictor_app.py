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
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    return df

# -------------------------------
# Model Training
# -------------------------------
def train_models(df):
    df = df.copy()
    df['Trend'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']
    
    X = df[features].iloc[:-1]  # drop last row for alignment
    y_reg = df['Close'].shift(-1).dropna()
    y_clf = df['Trend'].iloc[:-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    reg_model.fit(X_scaled, y_reg)
    clf_model.fit(X_scaled, y_clf)

    return reg_model, clf_model, scaler

# -------------------------------
# Streamlit UI & Logic
# -------------------------------
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📈 Stock Price & Trend Predictor")

symbol = st.text_input("Enter NSE stock symbol (e.g. HDFCBANK.NS):", value="HDFCBANK.NS")
target_date = st.date_input("📅 Select the target date to predict", value=date.today())

if st.button("Predict"):
    with st.spinner("🔄 Fetching data and preparing input..."):
        try:
            # 1. Download 1 year of daily data
            df = yf.download(symbol, period="1y", interval="1d", progress=False)

            # 2. Remove rows with missing values (weekends/holidays)
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

            # 3. Add indicators
            df = generate_features(df)
            df = df.dropna()  # drop indicator-related NaNs

            # 4. Handle target date
            target_date_str = target_date.strftime('%Y-%m-%d')
            target_dt = pd.to_datetime(target_date_str)

            # 5. Filter past data and target row
            df_train = df[df.index < target_dt]
            df_target = df[df.index == target_dt]

            # 6. Validate
            if df_target.empty:
                st.error("❌ No data available for selected date (weekend/holiday).")
            elif len(df_train) < 30:
                st.error("❌ Not enough past data to train the model. Try a later date.")
            elif df_train.isnull().values.any() or df_target.isnull().values.any():
                st.error("❌ Missing values found. Try a different date.")
            else:
                st.success(f"✅ Training on {len(df_train)} past records...")

                # 7. Train models
                reg_model, clf_model, scaler = train_models(df_train)

                # 8. Prepare input
                features = ['Open', 'High', 'Low', 'Close', 'Volume',
                            'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']
                X_target = df_target[features]
                X_target_scaled = scaler.transform(X_target)

                # 9. Predict
                predicted_price = reg_model.predict(X_target_scaled)[0]
                predicted_trend = clf_model.predict(X_target_scaled)[0]

                # 10. Display results
                st.subheader(f"📊 Prediction for {target_date_str}")
                st.success(f"💰 Predicted Price: ₹{predicted_price:.2f}")
                st.info(f"📈 Trend: {'🔺 UP' if predicted_trend == 1 else '🔻 DOWN'}")

                # Optional chart
                st.subheader("📉 Close Price - Last 60 Days")
                st.line_chart(df['Close'].tail(60))

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
