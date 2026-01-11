import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# -------------------- Config & Styling --------------------
st.set_page_config(page_title="WHR GA2 – Happiness Dashboard", layout="wide")
st.markdown("""
<style>
[data-testid="stTabs"] {
    justify-content: flex-end; /* align tabs to the right */
}
[data-testid="stMetric"] {background: #f8f9fa; padding: 12px; border-radius: 12px;}
h1, h2, h3 {letter-spacing: 0.2px;}
</style>
""", unsafe_allow_html=True)

# -------------------- Constants --------------------
TARGET_COL = "Life evaluation (3-year average)"
FEATURE_COLS = [
    "Explained by: Log GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
    "Explained by: Generosity",
    "Explained by: Perceptions of corruption",
]
DEFAULT_CSV = "WHRDATAFIGURE25_cleaned.csv"

# -------------------- Load & Train --------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    coef_df = pd.DataFrame({"Variable": FEATURE_COLS, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    return model, (r2, rmse, mae), coef_df, results_df

def find_closest_by_score(df, pred_score):
    tmp = df[["Country name", "Year", TARGET_COL]].copy()
    tmp["Difference"] = (tmp[TARGET_COL] - pred_score).abs()
    return tmp.sort_values("Difference").head(5)

def find_closest_by_profile(df, input_df):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    input_scaled = scaler.transform(input_df[FEATURE_COLS])
    distances = np.sqrt(((X_scaled - input_scaled) ** 2).sum(axis=1))
    tmp = df[["Country name", "Year", TARGET_COL]].copy()
    tmp["Profile distance"] = distances
    return tmp.sort_values("Profile distance").head(5)

# -------------------- Load Data --------------------
df = load_data(DEFAULT_CSV)
model, (r2, rmse, mae), coef_df, results_df = train_model(df)

# -------------------- Header --------------------
st.title("🌍 World Happiness Prediction Dashboard")
st.caption("Explore how policy indicators shape happiness scores (2020–2024)")

# -------------------- Prediction Section --------------------
st.subheader("🎛️ Adjust policy levers and predict happiness score")
cols = st.columns(3)
inputs = {}
icons = ["💰", "❤️", "🏥", "🕊️", "🎁", "⚖️"]

for i, feat in enumerate(FEATURE_COLS):
    with cols[i % 3]:
        label = f"{icons[i]} {feat.replace('Explained by: ', '')}"
        inputs[feat] = st.slider(label, float(df[feat].min()), float(df[feat].max()), float(df[feat].mean()))

input_df = pd.DataFrame([inputs])
pred = float(model.predict(input_df)[0])
rank_est = (df[TARGET_COL] > pred).sum() + 1
delta = pred - df[TARGET_COL].mean()

# KPI Card
st.subheader("📈 Predicted Happiness Score")
kpi = st.columns(3)
kpi[0].metric("Score", f"{pred:.2f}")
kpi[1].metric("Estimated Rank", f"{rank_est} / {df['Country name'].nunique()}")
kpi[2].metric("vs Global Avg", f"{delta:+.2f}")

# Comparisons
st.subheader("🧭 Country Comparisons")
by_score = find_closest_by_score(df, pred)
by_profile = find_closest_by_profile(df, input_df)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**A) Closest by happiness score**")
    st.dataframe(by_score, use_container_width=True)
with c2:
    st.markdown("**B) Closest by socioeconomic profile**")
    st.dataframe(by_profile, use_container_width=True)

top = by_score.iloc[0]
st.info(f"✨ This score is most similar to **{top['Country name']} ({int(top['Year'])})**, "
        f"with an actual score of **{top[TARGET_COL]:.2f}**.")

# World Map
st.subheader("🗺️ Global Happiness Map")
fig = px.choropleth(df, locations="Country name", locationmode="country names",
                    color=TARGET_COL, color_continuous_scale="Viridis",
                    title="Global Happiness Scores")
st.plotly_chart(fig, use_container_width=True)

# -------------------- Tabs for Model & About --------------------
tab1, tab2 = st.tabs(["🧠 Model", "ℹ️ About"])

with tab1:
    st.subheader("Model Performance (Hold-out Test Set)")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.3f}")
    m2.metric("RMSE", f"{rmse:.3f}")
    m3.metric("MAE", f"{mae:.3f}")

    st.divider()
    st.subheader("Feature Importance")
    st.bar_chart(coef_df.set_index("Variable"))

    st.divider()
    st.subheader("Actual vs Predicted")
    st.line_chart(results_df.reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("About This Application")
    st.markdown("""
**Course:** WQD7001 — Principles of Data Science (GA2)  
**Dataset:** World Happiness Report (2020–2024)  
**Model:** XGBoost Regressor  
**Purpose:** Simulate how changes in GDP, health, freedom, and other factors affect happiness scores.  

**SDGs:**  
- SDG 3: Good Health & Well-being  
- SDG 10: Reduced Inequalities
""")
