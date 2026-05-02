import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import traceback
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="F-Aloodah — Lahore AQI Forecast",
    page_icon="logo.png",
    layout="wide"
)

st.markdown("""
<style>

    [data-testid="stSidebar"] {
        background-color: #0c524a;
    }
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("lahore_air_quality_final_dataset.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_aqi_category(pm25):
    if pm25 <= 50:    return "Good",           "🟢"
    elif pm25 <= 100: return "Moderate",       "🟡"
    elif pm25 <= 200: return "Unhealthy",      "🟠"
    elif pm25 <= 300: return "Very Unhealthy", "🔴"
    else:             return "Hazardous",      "🟣"

def cat_num(v):
    if v <= 50:    return 0
    elif v <= 100: return 1
    elif v <= 200: return 2
    elif v <= 300: return 3
    else:          return 4

# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    feature_cols = ["T2M", "RH2M", "WS2M", "pm25_lag_1", "pm25_lag_7", "month", "smog_season"]
    X = df[feature_cols]
    y = df["pm25"]

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression().fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train_sc, y_train)
    y_pred_rf = rf.predict(X_test_sc)

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train_sc, y_train)
    y_pred_xgb = xgb.predict(X_test_sc)

    # LSTM
    features_lstm = ["T2M", "RH2M", "WS2M", "pm25_lag_1", "pm25_lag_7"]
    lstm_scaler = MinMaxScaler()
    data_scaled = lstm_scaler.fit_transform(df[features_lstm + ["pm25"]].values)

    SEQ = 7
    Xs, ys = [], []
    for i in range(len(data_scaled) - SEQ):
        Xs.append(data_scaled[i:i+SEQ, :-1])
        ys.append(data_scaled[i+SEQ, -1])
    Xs, ys = np.array(Xs), np.array(ys)

    sp2 = int(0.8 * len(Xs))
    X_tr_s, X_te_s = Xs[:sp2], Xs[sp2:]
    y_tr_s, y_te_s = ys[:sp2], ys[sp2:]

    lstm_model = Sequential([
        LSTM(50, input_shape=(SEQ, X_tr_s.shape[2])),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_tr_s, y_tr_s, epochs=50, batch_size=16,
                   validation_data=(X_te_s, y_te_s), verbose=0)

    y_pred_lstm_sc = lstm_model.predict(X_te_s, verbose=0)
    dummy       = np.zeros((len(y_pred_lstm_sc), len(features_lstm)))
    y_pred_lstm = lstm_scaler.inverse_transform(np.hstack([dummy, y_pred_lstm_sc]))[:, -1]
    y_test_lstm = lstm_scaler.inverse_transform(np.hstack([dummy, y_te_s.reshape(-1,1)]))[:, -1]

    def metrics(yt, yp):
        mae  = round(mean_absolute_error(yt, yp), 2)
        rmse = round(np.sqrt(mean_squared_error(yt, yp)), 2)
        acc  = round(accuracy_score([cat_num(v) for v in yt], [cat_num(v) for v in yp]), 3)
        return mae, rmse, acc

    results = {
        "Linear Regression": (*metrics(y_test, y_pred_lr),  y_pred_lr,   y_test.values),
        "Random Forest":     (*metrics(y_test, y_pred_rf),  y_pred_rf,   y_test.values),
        "XGBoost":           (*metrics(y_test, y_pred_xgb), y_pred_xgb,  y_test.values),
        "LSTM":              (*metrics(y_test_lstm, y_pred_lstm), y_pred_lstm, y_test_lstm),
    }

    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
    return results, feat_imp, scaler, rf, lr, xgb, feature_cols

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
from PIL import Image
logo = Image.open("logo.png")

col1, col2 = st.columns([1, 1])
with col1:
        st.title("F-ALOODAH")
        st.caption("An air quality prediction project by Eesha & Reesha")
 
with col2:
   st.image(logo, width=150)
st.markdown("---")

# ─────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────
try:
    with st.spinner("Loading data and training models — takes about 1 minute..."):
        df = load_data()
        results, feat_imp, scaler, rf, lr, xgb, feature_cols = train_models(df)
    st.success("Models ready!")
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("Predict AQI")
    st.markdown("Adjust weather conditions below:")

    temp  = st.slider("Temperature (°C)",  -5.0, 45.0, 18.0, 0.5)
    humid = st.slider("Humidity (%)",        10.0, 100.0, 60.0, 1.0)
    wind  = st.slider("Wind Speed (m/s)",     0.0, 10.0, 2.5, 0.1)
    lag1  = st.slider("Yesterday's PM2.5",    0.0, 500.0, 120.0, 1.0)
    lag7  = st.slider("Last Week's PM2.5",    0.0, 500.0, 100.0, 1.0)
    month = st.selectbox("Month", list(range(1, 13)),
                         format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                                "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
                         index=10)
    smog = 1 if month in [10, 11, 12, 1, 2] else 0
    st.info(f"Smog season: {'Yes (Oct–Feb)' if smog else 'No'}")

    st.markdown("---")
    st.subheader("Model")
    pred_model = st.radio("Choose model:", ["Linear Regression", "Random Forest", "XGBoost"])

# ─────────────────────────────────────────────
# LIVE PREDICTION
# ─────────────────────────────────────────────
model_map = {"Linear Regression": lr, "Random Forest": rf, "XGBoost": xgb}
inp    = np.array([[temp, humid, wind, lag1, lag7, month, smog]])
inp_sc = scaler.transform(inp)
pred   = float(model_map[pred_model].predict(inp_sc)[0])
pred   = max(0, pred)
cat, emoji = get_aqi_category(pred)

st.subheader("Live Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Predicted PM2.5", f"{pred:.1f} µg/m³")
c2.metric("AQI Category", f"{emoji} {cat}")
c3.metric("Model Used", pred_model)
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["Historical Trends", "Model Comparison"])

# ── TAB 1 ─────────────────────────────────────
with tab1:
    st.subheader("PM2.5 Trend Over Time")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(df["date"], df["pm25"], color="#1f77b4", linewidth=0.9)
    ax.fill_between(df["date"], df["pm25"], alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Daily PM2.5 Levels — Lahore (2022–2023)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close()

    st.subheader("Monthly Distribution — Smog Season Highlighted")
    fig2, ax2 = plt.subplots(figsize=(12, 3.5))
    month_groups = [df[df["date"].dt.month == m]["pm25"].values for m in range(1, 13)]
    bp = ax2.boxplot(month_groups, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
    for patch, m in zip(bp["boxes"], range(1, 13)):
        patch.set_facecolor("#e57373" if m in [10,11,12,1,2] else "#64b5f6")
        patch.set_alpha(0.6)
    ax2.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax2.set_xlabel("Month")
    ax2.set_ylabel("PM2.5 (µg/m³)")
    ax2.set_title("Monthly PM2.5 — 🔴 Smog Season (Oct–Feb)  🔵 Off-Season")
    st.pyplot(fig2)
    plt.close()

# ── TAB 2 ─────────────────────────────────────
with tab2:
    st.subheader("Model Performance Metrics")
    summary = pd.DataFrame({
        "Model":          list(results.keys()),
        "MAE ↓":          [results[m][0] for m in results],
        "RMSE ↓":         [results[m][1] for m in results],
        "AQI Accuracy ↑": [results[m][2] for m in results],
    }).set_index("Model")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Actual vs Predicted PM2.5")
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fig3, axes = plt.subplots(2, 2, figsize=(14, 7))
    for ax, (name, color) in zip(axes.flatten(), zip(results.keys(), colors)):
        yt = results[name][4]
        yp = results[name][3]
        n  = min(len(yt), len(yp), 120)
        ax.plot(yt[:n], color="grey",  linewidth=1,   label="Actual",    alpha=0.7)
        ax.plot(yp[:n], color=color,   linewidth=1.2, label="Predicted", alpha=0.9)
        ax.set_title(f"{name}  |  MAE {results[name][0]}  RMSE {results[name][1]}")
        ax.set_ylabel("PM2.5")
        ax.legend(fontsize=8)
    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.subheader("MAE & RMSE Bar Chart")
    names = list(results.keys())
    maes  = [results[m][0] for m in names]
    rmses = [results[m][1] for m in names]
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 3.5))
    bars1 = ax4a.bar(names, maes,  color=colors, alpha=0.8)
    ax4a.set_title("MAE (lower = better)"); ax4a.set_ylabel("MAE")
    ax4a.bar_label(bars1, fmt="%.1f", fontsize=9)
    ax4a.set_xticklabels(names, rotation=15, ha="right")
    bars2 = ax4b.bar(names, rmses, color=colors, alpha=0.8)
    ax4b.set_title("RMSE (lower = better)"); ax4b.set_ylabel("RMSE")
    ax4b.bar_label(bars2, fmt="%.1f", fontsize=9)
    ax4b.set_xticklabels(names, rotation=15, ha="right")
    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close()

st.markdown("---")
st.caption("F-ALOODAH · Eesha & Reesha · DS Project · Data: OpenAQ + NASA POWER")