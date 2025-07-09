import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# ---------------------- Page Setup ------------------------
st.set_page_config(page_title="Stock Predictor", layout="wide")

st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
        }
        .subtitle {
            font-size: 20px;
            color: #444;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 18px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- Sidebar Menu ----------------------
with st.sidebar:
    selected = option_menu(
        "Navigation", ["📊 Predict", "ℹ️ About"],
        icons=["bar-chart-line", "info-circle"],
        menu_icon="cast", default_index=0)

# ---------------------- Title ----------------------
st.markdown('<div class="title">📈 Stock Price & Trend Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart predictions using ML & technical indicators</div>', unsafe_allow_html=True)

# ---------------------- About Page ----------------------
if selected == "ℹ️ About":
    st.markdown("""
        This is a professional-grade stock prediction app using:
        - Machine learning (Random Forest)
        - Technical indicators (MA, RSI, Returns)
        - Interactive visualizations
        - Extendable to include news sentiment and LSTM models
    """)
    st.stop()

# ---------------------- Prediction Page ----------------------

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    symbol = st.text_input("Enter NSE Symbol (e.g. HDFCBANK.NS):", value="HDFCBANK.NS")
with col2:
    st.image("https://companiesmarketcap.com/img/company-logos/512/HDFCBANK.NS.png", width=80)

df = yf.download(symbol, period="1y", interval="1d", progress=False)

if df.empty:
    st.error("No data found. Please check the symbol.")
    st.stop()

# Clean column names
if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
else:
    df.columns = df.columns.str.strip()

# Column Mapping using startswith
col_map = {}
for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
    match = [col for col in df.columns if col.lower().startswith(base.lower())]
    col_map[base] = match[0] if match else None

if not all(col_map.values()):
    st.error("Some required columns missing.")
    st.write("Returned columns:", list(df.columns))
    st.stop()

# Drop missing rows
df = df.dropna(subset=col_map.values())

# Feature Engineering
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

min_date = df.index.min().date()
max_date = df.index.max().date()

target_date = st.date_input("📅 Select prediction date", value=max_date, min_value=min_date, max_value=max_date)

# Tabs
tab1, tab2 = st.tabs(["📈 Prediction", "📉 Historical Chart"])

with tab1:
    if st.button("🚀 Predict"):
        with st.spinner("Training model and predicting..."):
            try:
                target_dt = pd.to_datetime(target_date)
                df_train = df[df.index < target_dt]
                df_target = df[df.index == target_dt]

                features = list(col_map.values()) + ['MA_5', 'MA_20', 'RSI_14', 'Daily_Return']

                if df_target.empty:
                    st.error("No trading data for selected date.")
                elif len(df_train) < 30:
                    st.error("Insufficient data to train.")
                else:
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

                    col1, col2 = st.columns(2)
                    col1.metric(label="💰 Predicted Price", value=f"₹{price:.2f}")
                    col2.metric(label="📈 Predicted Trend", value="🔺 UP" if trend == 1 else "🔻 DOWN")

            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.subheader("🔍 Close Price (Last 60 Days)")
    fig = plot_data(df.tail(60), col_map['Close'])  # Only show last 60 days
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Plot Function ----------------------
def plot_data(df, close_col):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df[close_col],
        mode='lines+markers',
        name='Close Price',
        line=dict(color='blue')))

    fig.update_layout(
        title='📉 Stock Close Price (Last 60 Days)',
        xaxis_title='Date',
        yaxis_title='Price (INR)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
        template='plotly_white'
    )

    return fig
