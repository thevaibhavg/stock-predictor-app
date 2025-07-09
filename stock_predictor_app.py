import streamlit as st
import yfinance as yf 
import pandas as pd 
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
import plotly.graph_objects as go 
from streamlit_option_menu import option_menu

st.set_page_config(page_title=" Predictor", layout="wide")

#---------------------- Styling ----------------------

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
    } footer {visibility: hidden;}
    </style> 
    """, unsafe_allow_html=True)

#---------------------- Sidebar ----------------------

with st.sidebar: 
    selected = option_menu(
        "📋 Main Menu", 
        ["📉 Chart", "ℹ️ About"], 
        icons=["bar-chart-line", "graph-up", "info-circle"],
        default_index=0 
    )

#---------------------- Data ----------------------

symbol = "HDFCBANK.NS" 
df = yf.download(symbol, period="1y", interval="1d", progress=False)

if df.empty:
    st.error("No data found.") 
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]
else:
    df.columns = df.columns.astype(str)

required_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] 
if not all(col in df.columns for col in required_cols): 
    st.error("Missing required columns: " + str(required_cols))
    st.write("Available columns:", list(df.columns))
    st.stop()

def generate_features(df): 
    df['MA_5'] = df['Close'].rolling(5).mean() 
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['Daily_Return'] = df['Close'].pct_change() 
    delta = df['Close'].diff() 
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    return df

df = generate_features(df).dropna()

#---------------------- Header ----------------------

st.markdown("""
<div style="background-color:#0E76A8;padding:20px;border-radius:10px"> 
<h1 style="color:white;text-align:center;">📈 Predictor App</h1>
<p style="color:white;text-align:center;">Smart machine learning predictions with interactive charts</p>
</div> 
<br> 
""", unsafe_allow_html=True)

#---------------------- About ----------------------

if selected == "ℹ️ About": 
    st.markdown("""
    ### ℹ️ About the App
    - Built with ❤️ using Streamlit, Plotly, and Machine Learning
    - Predicts future stock price and trend direction
    - Features include:
            - Interactive charts 
            - Technical indicators 
            - Professional UI
            """)
    st.stop()

#---------------------- Predict ----------------------

if selected == "📊 Predict":
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol = st.text_input("Enter NSE Symbol (e.g. HDFCBANK.NS):", value=symbol)

min_date = df.index.min().date()
max_date = df.index.max().date()

col1, col2 = st.columns(2)
with col1:
    target_date = st.date_input("📅 Select prediction date", value=max_date, min_value=min_date, max_value=max_date)

with st.expander("⚙️ Advanced Options"):
    use_scaler = st.checkbox("Use Standard Scaler", value=True)
    n_estimators = st.slider("Number of Trees", 50, 300, 100, step=50)

if st.button("🚀 Predict"):
    with st.spinner("Training model and predicting..."):
        try:
            target_dt = pd.to_datetime(target_date)
            df_train = df[df.index < target_dt]
            df_target = df[df.index == target_dt]

            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'RSI_14', 'Daily_Return']

            if df_target.empty:
                st.error("No trading data for selected date.")
            elif len(df_train) < 30:
                st.error("Insufficient data to train.")
            else:
                df_train = df_train.copy()
                df_train['Trend'] = np.where(df_train['Close'].shift(-1) > df_train['Close'], 1, 0)

                X = df_train[features].iloc[:-1]
                y_reg = df_train['Close'].shift(-1).dropna()
                y_clf = df_train['Trend'].iloc[:-1]

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X) if use_scaler else X

                reg_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                clf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

                reg_model.fit(X_scaled, y_reg)
                clf_model.fit(X_scaled, y_clf)

                X_target = df_target[features]
                X_target_scaled = scaler.transform(X_target) if use_scaler else X_target

                try:
                    # --- Predict Price and Trend ---
                    price = reg_model.predict(X_target_scaled)[0]
                    trend = clf_model.predict(X_target_scaled)[0]

                    # --- Predictive Confidence Score ---
                    trend_proba = clf_model.predict_proba(X_target_scaled)[0][1]
                    confidence_pct = trend_proba * 100

                    col1, col2 = st.columns(2)
                    col1.metric("💰 Predicted Price", f"₹{price:.2f}")
                    col2.metric("📈 Predicted Trend", "🔺 UP" if trend == 1 else "🔻 DOWN")

                    st.markdown("#### 🔍 Prediction Confidence")
                    st.progress(int(confidence_pct))
                    st.info(f"Confidence: {confidence_pct:.2f}% that the trend is {'UP' if trend == 1 else 'DOWN'}")
                
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        
        except Exception as e:
            st.error(f"Error: {e}")

#---------------------- Chart ----------------------

if selected == "📉 Chart": 
     def plot_data(df): 
         fig = go.Figure()
         fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines+markers', name='Close', line=dict(color='blue')))
         fig.add_trace(go.Scatter(x=df.index, y=df['MA_5'], mode='lines', name='MA 5', line=dict(color='green', dash='dash')))
         fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], mode='lines', name='MA 20', line=dict(color='orange', dash='dot')))
         fig.update_layout(title='📉 Stock Close Price (Last 60 Days)', xaxis_title='Date', yaxis_title='Price (INR)', height=500, template='plotly_white')
         return fig

st.subheader("📊 Historical Chart")
st.plotly_chart(plot_data(df.tail(60)), use_container_width=True)

#---------------------- Footer ----------------------

st.markdown("""
<hr> 
<center> 
Made with ❤️ by <a href="https://github.com/vaibhavgupta20" target="_blank">vaibhavgupta20</a> | Powered by Streamlit
</center>
""", unsafe_allow_html=True)

