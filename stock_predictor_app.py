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

# ---------------------- CSS Styling ------------------------
st.markdown("""
    <style>
        .title { font-size: 36px; font-weight: bold; color: #0E76A8; }
        .subtitle { font-size: 18px; color: #444; }
        .stButton>button {
            background-color: #0E76A8;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 18px;
        }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------- Sidebar Navigation ------------------------
with st.sidebar:
    selected = option_menu(
        "📋 Main Menu",
        ["📊 Predict", "📉 Chart", "ℹ️ About"],
        icons=["bar-chart-line", "graph-up", "info-circle"],
        default_index=0
    )

# ---------------------- Global Stock Data Setup ------------------------
symbol = "HDFCBANK.NS"
df = yf.download(symbol, period="1y", interval="1d", progress=False)

if df.empty:
    st.error("No data found.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
else:
    df.columns = df.columns.str.strip()

col_map = {}
for base in ['Open', 'High', 'Low', 'Close', 'Volume']:
    match = [col for col in df.columns if col.lower().startswith(base.lower())]
    col_map[base] = match[0] if match else None

if not all(col_map.values()):
    st.error("Some required OHLCV columns are missing.")
    st.write("Columns:", list(df.columns))
    st.stop()

df = df.dropna(subset=col_map.values())

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

# ---------------------- Hero Header ------------------------
st.markdown("""
    <div style="background-color:#0E76A8;padding:20px;border-radius:10px">
        <h1 style="color:white;text-align:center;">📈 Stock Predictor App</h1>
        <p style="color:white;text-align:center;">Smart machine learning predictions with interactive charts</p>
    </div>
    <br>
""", unsafe_allow_html=True)

# ---------------------- About Page ------------------------
if selected == "ℹ️ About":
    st.markdown("""
    ### ℹ️ About the App
    - Built with ❤️ using **Streamlit**, **Plotly**, and **Machine Learning**
    - Predicts future stock **price** and **trend direction**
    - Features include:
        - Interactive charts
        - Technical indicators
        - Professional UI
    """)
    st.stop()

# ---------------------- Prediction Section ------------------------
if selected == "📊 Predict":
    col1, col2 = st.columns([3, 1])
    with col1:
        user_symbol = st.text_input("Enter NSE Symbol (e.g. HDFCBANK.NS):", value=symbol)
    with col2:
        st.write("")

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("📅 Select prediction date", value=max_date, min_value=min_date, max_value=max_date)
    with col2:
        st.write("")

    with st.expander("⚙️ Advanced Options"):
        use_scaler = st.checkbox("Use Standard Scaler", value=True)
        n_estimators = st.slider("Number of Trees", 50, 300, 100, step=50)

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
                    df_train['Trend'] = np.where(
                        df_train[col_map['Close']].shift(-1) > df_train[col_map['Close']], 1, 0
                    )

                    X = df_train[features].iloc[:-1]
                    y_reg = df_train[col_map['Close']].shift(-1).dropna()
                    y_clf = df_train['Trend'].iloc[:-1]

                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X) if use_scaler else X

                    reg_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    clf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

                    reg_model.fit(X_scaled, y_reg)
                    clf_model.fit(X_scaled, y_clf)

                    X_target = df_target[features]
                    X_target_scaled = scaler.transform(X_target) if use_scaler else X_target

                    price = reg_model.predict(X_target_scaled)[0]
                    trend = clf_model.predict(X_target_scaled)[0]

                    col1, col2 = st.columns(2)
                    col1.metric("💰 Predicted Price", f"₹{price:.2f}")
                    col2.metric("📈 Predicted Trend", "🔺 UP" if trend == 1 else "🔻 DOWN")

            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------- Chart Tab ------------------------
if selected == "📉 Chart":
    def plot_data(df, close_col):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df[close_col],
            mode='lines+markers',
            name='Close Price',
            line=dict(color='blue')))

        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_5'],
            mode='lines', name='MA 5',
            line=dict(color='green', dash='dash')))

        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_20'],
            mode='lines', name='MA 20',
            line=dict(color='orange', dash='dot')))

        fig.update_layout(
            title='📉 Stock Close Price (Last 60 Days)',
            xaxis_title='Date',
            yaxis_title='Price (INR)',
            margin=dict(l=20, r=20, t=30, b=20),
            height=500,
            template='plotly_white'
        )

        return fig

    st.subheader("📊 Historical Chart")
    st.plotly_chart(plot_data(df.tail(60), col_map['Close']), use_container_width=True)

# ---------------------- Footer ------------------------
st.markdown("""
    <hr>
    <center>
        Made with ❤️ by <a href="https://github.com/yourusername" target="_blank">Your Name</a> | Powered by Streamlit
    </center>
""", unsafe_allow_html=True)
