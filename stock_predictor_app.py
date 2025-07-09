import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ========== Streamlit UI ==========
st.set_page_config(page_title="📈 Stock Predictor", layout="centered")
st.title("📈 Stock Price & Trend Predictor")

symbol = st.text_input("Enter NSE stock symbol (e.g. HDFCBANK.NS):", value="HDFCBANK.NS")

df = yf.download(symbol, period="1y", interval="1d", progress=False)

# Fix for MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
else:
    df.columns = df.columns.str.strip()

# # Correct dynamic mapping
col_map = {}
for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
    match = [col for col in df.columns if col.lower().startswith(base.lower())]
    col_map[base] = match[0] if match else None
    
required = ['Open', 'High', 'Low', 'Close', 'Volume']
if not all(k in col_map for k in required):
    st.error("❌ Required columns not found. Check the symbol or try later.")
    st.write("Returned columns:", list(df.columns))
    st.stop()

# Drop missing values
df = df.dropna(subset=col_map.values())

# ========== Feature Engineering ==========
def generate_features(df):
    close = col_map['Close']
    df['MA_5'] = df[close].rolling(5).mean()
    df['MA_20'] = df[close].rolling(20).mean()
    df['Daily_Return'] = df[close].pct_change()

    delta = df[close].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    return df

df = generate_features(df).dropna()

# ========== Date Selection ==========
min_date = df.index.min().date()
max_date = df.index.max().date()
target_date = st.date_input("📅 Select date to predict", value=max_date, min_value=min_date, max_value=max_date)

# ========== Prediction ==========
if st.button("Predict"):
    with st.spinner("🔄 Predicting..."):
        try:
            target_dt = pd.to_datetime(target_date)
            df_train = df[df.index < target_dt]
            df_target = df[df.index == target_dt]

            features = list(col_map.values()) + ['MA_5', 'MA_20', 'RSI_14', 'Daily_Return']

            if df_target.empty:
                st.error("No data on selected date.")
                st.stop()

            df_train = df_train.copy()
            df_train['Trend'] = np.where(df_train[col_map['Close']].shift(-1) > df_train[col_map['Close']], 1, 0)

            X = df_train[features].iloc[:-1]
            y_reg = df_train[col_map['Close']].shift(-1).dropna()
            y_clf = df_train['Trend'].iloc[:-1]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
            clf_model = RandomForestClassifier(n_estimators=100, random_state=42)

            reg_model.fit(X_scaled, y_reg)
            clf_model.fit(X_scaled, y_clf)

            X_target = df_target[features]
            X_target_scaled = scaler.transform(X_target)

            price = reg_model.predict(X_target_scaled)[0]
            trend = clf_model.predict(X_target_scaled)[0]

            st.subheader(f"📊 Prediction for {target_date}")
            st.success(f"💰 Price: ₹{price:.2f}")
            st.info(f"📈 Trend: {'🔺 UP' if trend == 1 else '🔻 DOWN'}")
            st.line_chart(df[col_map['Close']].tail(60))

        except Exception as e:
            st.error(f"❌ Error: {e}")
