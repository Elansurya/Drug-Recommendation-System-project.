import os
import warnings
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# CONFIGURATION

MODEL_PATH    = r"C:\project\Drug Recommendation System\models\drug_model.pkl"
FEATURE_ORDER = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
ENCODING_MAP  = {
    "Sex":         {"Female": 0, "Male": 1},
    "BP":          {"High": 0, "Low": 1, "Normal": 2},
    "Cholesterol": {"High": 0, "Normal": 1},
}

# PAGE CONFIG

st.set_page_config(
    page_title="Drug Recommendation System",
    page_icon="💊",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .block-container {
    background-color: #0A0A0A !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden !important; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 750px !important;
}

/* All text white */
p, span, div, h1, h2, h3, h4, label,
.stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #FFFFFF !important;
}

/* Input labels */
[data-testid="stNumberInput"] label p,
[data-testid="stSelectbox"] label p {
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* Number input box */
[data-testid="stNumberInput"] input {
    background-color: #1C1C1C !important;
    color: #FFFFFF !important;
    border: 1.5px solid #333333 !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background-color: #1C1C1C !important;
    color: #FFFFFF !important;
    border: 1.5px solid #333333 !important;
    border-radius: 8px !important;
}
[data-testid="stSelectbox"] svg { fill: #FFFFFF !important; }

/* Dropdown list */
[data-baseweb="menu"] { background-color: #1C1C1C !important; }
[data-baseweb="menu"] li { color: #FFFFFF !important; background-color: #1C1C1C !important; }
[data-baseweb="menu"] li:hover { background-color: #2A2A2A !important; }

/* +/- buttons */
[data-testid="stNumberInput"] button {
    background-color: #2A2A2A !important;
    color: #FFFFFF !important;
    border-color: #333333 !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #00C896, #007A5E) !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    padding: 0.75rem !important;
    box-shadow: 0 4px 20px rgba(0,200,150,0.30) !important;
    margin-top: 0.5rem !important;
}
.stButton > button p { color: #000000 !important; font-weight: 700 !important; }
.stButton > button:hover {
    background: linear-gradient(135deg, #00E0A8, #00C896) !important;
    transform: translateY(-1px) !important;
}

/* st.success box */
[data-testid="stAlert"] {
    background-color: #0D2B22 !important;
    border: 2px solid #00C896 !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}
[data-testid="stAlert"] p {
    color: #FFFFFF !important;
    font-size: 1rem !important;
}

/* st.metric */
[data-testid="stMetric"] {
    background-color: #111111 !important;
    border: 1px solid #222222 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] p  { color: #888888 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"]    { color: #FFFFFF !important; font-size: 1.3rem !important; font-weight: 700 !important; }

/* Divider */
hr { border-color: #1E1E1E !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: #00C896 !important;
}
[data-testid="stProgress"] {
    background-color: #1C1C1C !important;
    border-radius: 99px !important;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH} — run model_training.py first.")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

model = load_model()


# HELPER: Build Encoded DataFrame

def build_input_dataframe(age, sex, bp, cholesterol, na_to_k):
    df = pd.DataFrame([{
        "Age": age, "Sex": sex,
        "BP": bp, "Cholesterol": cholesterol, "Na_to_K": na_to_k
    }])
    for col, mapping in ENCODING_MAP.items():
        df[col] = df[col].map(mapping)
    return df[FEATURE_ORDER]


# HERO BANNER  (this HTML is static — no f-string variables, renders fine)

st.markdown("""
<div style="background:linear-gradient(135deg,#00C896,#003D30);
            border-radius:16px; padding:2rem 2.4rem 1.8rem;
            margin-bottom:1.8rem; box-shadow:0 8px 40px rgba(0,200,150,0.2);">
    <div style="font-size:2rem; font-weight:800; color:#FFFFFF; margin-bottom:0.4rem;">
        💊 Drug Recommendation System
    </div>
    <div style="font-size:0.98rem; color:rgba(255,255,255,0.80);">
        Enter patient vitals below to receive an AI-powered drug recommendation
    </div>
</div>
""", unsafe_allow_html=True)


# SECTION HEADER

st.markdown("""
<div style="font-size:0.7rem; font-weight:700; letter-spacing:0.14em;
            text-transform:uppercase; color:#00C896; margin-bottom:0.8rem;">
    🧑‍⚕️ Patient Information
</div>
""", unsafe_allow_html=True)
st.divider()

# INPUT FIELDS

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age (years)", min_value=0, max_value=120,
        value=35, step=1, help="Patient age (0–120)"
    )
    bp = st.selectbox(
        "Blood Pressure (BP)",
        options=["High", "Low", "Normal"],
        help="Patient blood pressure level"
    )
    na_to_k = st.number_input(
        "Na_to_K Ratio", min_value=0.1, max_value=100.0,
        value=15.0, step=0.1, format="%.2f",
        help="Sodium-to-Potassium ratio in blood"
    )

with col2:
    sex = st.selectbox(
        "Sex", options=["Female", "Male"],
        help="Patient biological sex"
    )
    cholesterol = st.selectbox(
        "Cholesterol", options=["High", "Normal"],
        index=1, help="Patient cholesterol level"
    )

st.write("")


# PREDICT BUTTON

predict_clicked = st.button("💊  Predict Recommended Drug")


if predict_clicked:

    input_df       = build_input_dataframe(age, sex, bp, cholesterol, na_to_k)
    predicted_drug = model.predict(input_df)[0]
    probabilities  = model.predict_proba(input_df)[0]

    all_scores = {
        drug: round(float(prob) * 100, 2)
        for drug, prob in zip(model.classes_, probabilities)
    }
    all_scores_sorted = dict(
        sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    )
    confidence = all_scores_sorted[predicted_drug]

    st.write("")

    # ── SUCCESS MESSAGE ────────────────────────────────────────────────────
    st.success(f"✅  Recommended Drug:  **{predicted_drug}**   |   Confidence: **{confidence:.1f}%**")

    st.write("")

    # ── PATIENT SUMMARY (native metrics) ──────────────────────────────────
    st.markdown("""
    <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.14em;
                text-transform:uppercase; color:#00C896; margin-bottom:0.6rem;">
        Patient Summary
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Age",         str(age))
    m2.metric("Sex",         sex)
    m3.metric("BP",          bp)
    m4.metric("Cholesterol", cholesterol)
    m5.metric("Na_to_K",     f"{na_to_k:.2f}")

    st.write("")
    st.divider()

    # ── CONFIDENCE BREAKDOWN (native progress bars) ────────────────────────
    st.markdown("""
    <div style="font-size:0.7rem; font-weight:700; letter-spacing:0.14em;
                text-transform:uppercase; color:#00C896; margin-bottom:0.8rem;">
        📊 Prediction Confidence Breakdown
    </div>
    """, unsafe_allow_html=True)

    for drug, score in all_scores_sorted.items():
        c_label, c_bar, c_pct = st.columns([2, 6, 1])

        with c_label:
            if drug == predicted_drug:
                st.markdown(f"**:green[{drug}]**")
            else:
                st.write(drug)

        with c_bar:
            # st.progress expects a float 0.0–1.0
            st.progress(score / 100)

        with c_pct:
            if drug == predicted_drug:
                st.markdown(f"**:green[{score:.1f}%]**")
            else:
                st.markdown(f"<span style='color:#888;font-size:0.85rem'>{score:.1f}%</span>",
                            unsafe_allow_html=True)

    st.write("")


# FOOTER

st.divider()
st.markdown("""
<div style="text-align:center; color:#444444; font-size:0.78rem;">
    Drug Recommendation System &nbsp;·&nbsp; Powered by Decision Tree Classifier<br>
    For educational purposes only — not a substitute for medical advice.
</div>
""", unsafe_allow_html=True)