import os
import time
import threading
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from lime import lime_tabular
from streamlit.runtime.scriptrunner import add_script_run_ctx

st.set_page_config(
    page_title="HTN/DM Risk Prediction Model",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "XGBoost.pkl")
LIME_BG_PATH = os.path.join(BASE_DIR, "lime_background_train.csv")

FEATURE_NAMES = ["family_DM", "age", "BMI", "HR", "smoke"]
MODEL_AUC = 0.853


@st.cache_resource
def load_model():
    m = joblib.load(MODEL_PATH)
    m.predict_proba(np.zeros((1, len(FEATURE_NAMES))))
    return m


@st.cache_resource
def load_lime_explainer():
    bg = pd.read_csv(LIME_BG_PATH)
    return lime_tabular.LimeTabularExplainer(
        training_data=bg.values,
        mode="classification",
        feature_names=bg.columns.tolist(),
        class_names=["Negative", "Positive"],
        random_state=42,
        verbose=False,
        discretize_continuous=False,
    )


model = load_model()
explainer = load_lime_explainer()

st.markdown(
    """
    <style>
    [data-testid="stToolbar"], [data-testid="stHeader"], [data-testid="stDecoration"] {
        display: none !important;
    }
    #MainMenu {display: none !important;}
    footer, [data-testid="stFooter"], [data-testid="stBottom"],
    [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"],
    [data-testid="stMainMenu"] {
        display: none !important;
    }
    .block-container {padding-top: .6rem; padding-bottom: .6rem; max-width: 1500px;}
    [data-testid="stAppViewContainer"] {background: #f4f7fa;}

    .header-title {
        text-align: center;
        font-family: "Times New Roman", Georgia, serif;
        font-size: 2.1rem; font-weight: 700; color: #16222e;
        letter-spacing: .01em; margin-bottom: .1rem; line-height: 1.2;
    }
    .header-sub {
        text-align: center;
        font-family: "Times New Roman", Georgia, serif;
        font-size: 1.2rem; color: #4d6479; margin-bottom: .05rem;
    }
    .header-meta {
        text-align: center; font-size: 1.02rem; color: #7c8b99;
        margin-bottom: .3rem;
    }
    .header-rule {
        border: none; border-top: 2px solid #1f4e79; width: 300px;
        margin: .05rem auto .4rem auto;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e2e9f1;
        border-radius: 14px;
        padding: 1.15rem 1.5rem;
        box-shadow: 0 2px 12px rgba(31, 78, 121, .08);
    }
    .section-title {
        font-size: 1.18rem; letter-spacing: .09em; text-transform: uppercase;
        color: #1f4e79; font-weight: 700;
        border-bottom: 2px solid #1f4e79;
        padding-bottom: .35rem; margin-bottom: .7rem;
    }

    .risk-card {
        border-radius: 12px; padding: .9rem 1.3rem; margin-top: .2rem;
        border: 1px solid transparent;
    }
    .risk-low {
        border-left: 6px solid #2a9d8f; background: #f4fbf9; border-color: #d6ede7;
    }
    .risk-high {
        border-left: 6px solid #c0392b; background: #fdf6f5; border-color: #f1d8d4;
    }
    .risk-label {font-size: 1.2rem; color: #3b4a58;}
    .prob-caption {font-size: 1.08rem; color: #3b4a58; margin-top: .15rem;}
    .prob-num {font-size: 2.8rem; font-weight: 700; font-family: "Times New Roman", serif;}
    .risk-low .prob-num {color: #1f7a6f;}
    .risk-high .prob-num {color: #b03a2e;}

    .summary {font-size: 1.02rem; color: #6d7b89; margin-top: .45rem;}

    .note {
        font-size: 1.05rem; color: #55636f; margin-top: .6rem;
        line-height: 1.6;
    }

    .footer-left {
        font-size: .98rem; color: #8594a2; text-align: left;
        border-top: 1px solid #e8ecf1; padding-top: .45rem; margin-top: .6rem;
        line-height: 1.55;
    }

    .stNumberInput label, .stRadio label {font-size: 1.18rem; font-weight: 500; color: #2c3a47;}
    .stNumberInput div[data-baseweb="input"] {
        max-height: 2.9rem; border-radius: 8px; border-color: #c9d6e2; background: #ffffff;
    }
    .stNumberInput div[data-baseweb="input"]:focus-within {
        border-color: #1f4e79; box-shadow: 0 0 0 1px #1f4e79;
    }
    .stNumberInput div[data-baseweb="input"] input {font-size: 1.2rem; font-weight: 500;}
    .stRadio > div {gap: .1rem;}
    .stRadio div[role="radio"][aria-checked="true"] {
        background: #1f4e79; border-color: #1f4e79;
    }
    .stButton button {
        font-size: 1.2rem; height: 3rem; font-weight: 600;
        border-radius: 8px; border: none;
        box-shadow: 0 2px 8px rgba(31, 78, 121, .25);
    }
    .stButton button:hover {box-shadow: 0 4px 14px rgba(31, 78, 121, .35);}
    .stInfo {
        background: #f0f6fb; border: 1px solid #d5e3f0;
        border-left: 4px solid #1f4e79; border-radius: 8px;
        padding: .9rem 1.05rem; font-size: 1.05rem; color: #33475b;
    }
    .stInfo p {margin: 0;}
    .stInfo code {background: #e3edf5; color: #1f4e79; border-radius: 4px; padding: .1em .35em;}
    [data-testid="stTooltipIcon"] svg {fill: #8ba0b5;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='header-title'>Machine Learning Model for Predicting Onset of "
    "Hypertension & Diabetes Mellitus</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='header-sub'>XGBoost-based Risk Stratification &middot; "
    "5 Clinical Features</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<div class='header-meta'>Test AUC = {MODEL_AUC} &middot; "
    "For research reference only, not for clinical diagnosis</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr class='header-rule'>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1.12], gap="medium")

with left_col:
    st.markdown(
        "<div class='panel'>"
        "<div class='section-title'>Variables</div>",
        unsafe_allow_html=True,
    )

    family_dm = st.radio(
        "Family History of Diabetes Mellitus (family_DM)",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        index=0,
        horizontal=True,
        help="No = without family history of diabetes; Yes = with family history of diabetes",
    )

    age = st.number_input(
        "Age (age)",
        min_value=18, max_value=110, value=45, step=1,
        help="Age in years (18-110)",
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0, max_value=100.0, value=24.0, step=0.1,
        help="Body Mass Index, kg/m² (0-100)",
    )

    hr = st.number_input(
        "Heart Rate (HR)",
        min_value=0, max_value=300, value=72, step=1,
        help="Resting heart rate, beats per minute (0-300)",
    )

    smoke = st.radio(
        "Smoking Status (smoke)",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        index=0,
        horizontal=True,
        help="No = not smoking; Yes = currently smoking",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='footer-left'>Model: XGBoost &middot; Features: family_DM, age, BMI, HR, "
        "smoke &middot; Test AUC = 0.853 &middot; "
        "For research reference only, not medical advice.</div>",
        unsafe_allow_html=True,
    )

    note_ph = st.empty()

with right_col:
    st.markdown(
        "<div class='panel'>"
        "<div class='section-title'>Prediction Result</div>",
        unsafe_allow_html=True,
    )

    predict_clicked = st.button(
        "Predict", type="primary", use_container_width=True
    )

    if predict_clicked:
        t_pred0 = time.time()
        features = np.array([[float(family_dm), float(age), float(bmi),
                              float(hr), float(smoke)]])
        proba = model.predict_proba(features)[0][1]
        t_pred = time.time() - t_pred0
        proba_pct = proba * 100.0

        if proba >= 0.5:
            risk_group, risk_class = "High Risk", "risk-high"
        else:
            risk_group, risk_class = "Low Risk", "risk-low"

        st.markdown(
            f"""
            <div class='risk-card {risk_class}'>
                <div class='risk-label'>
                    Risk grouping for Diabetes: <b>{risk_group}</b>
                </div>
                <div class='prob-caption'>Probability of Diabetes:</div>
                <div class='prob-num'>{proba_pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='summary'>Input features &mdash; "
            f"family_DM: {family_dm}, age: {age}, BMI: {bmi:.1f}, "
            f"HR: {hr}, smoke: {smoke}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='section-title'>LIME Explanation</div>",
            unsafe_allow_html=True,
        )

        progress_ph = st.empty()
        bar = progress_ph.progress(0, text="Generating LIME explanation, please wait... 0%")
        stop_evt = threading.Event()

        def _tick():
            try:
                for pct in range(3, 97, 3):
                    if stop_evt.is_set():
                        break
                    bar.progress(pct, text=f"Generating LIME explanation, please wait... {pct}%")
                    time.sleep(0.10)
            except Exception:
                pass

        tick_th = threading.Thread(target=_tick, daemon=True)
        add_script_run_ctx(tick_th)
        tick_th.start()

        t_lime = None
        t_lime0 = time.time()
        try:
            exp = explainer.explain_instance(
                data_row=features[0],
                predict_fn=model.predict_proba,
                num_features=9,
                num_samples=1500,
            )
            lime_html = exp.as_html()
            resize_js = (
                "<script>"
                "(function(){try{"
                "var frs=window.parent.document.querySelectorAll('iframe');"
                "var fr=frs[frs.length-1];"
                "if(fr&&fr.style){fr.style.height=(document.body.scrollHeight+28)+'px';}"
                "}catch(e){}})();"
                "</script>"
            )
            lime_html = lime_html.replace("</body>", resize_js + "</body>")
            mobile_css = (
                "<style>"
                "::-webkit-scrollbar{display:none;width:0;height:0;}"
                "*{scrollbar-width:none;-ms-overflow-style:none;}"
                "@media (max-width:500px){body{zoom:.8;}}"
                "</style>"
            )
            lime_html = lime_html.replace("</head>", mobile_css + "</head>")
            st.components.v1.html(lime_html, height=430, scrolling=False)
            t_lime = time.time() - t_lime0
        except Exception as e:
            st.error(f"LIME explanation failed: {e}")
        finally:
            stop_evt.set()
            bar.progress(100, text="LIME explanation complete")
            time.sleep(0.3)
            progress_ph.empty()
            tick_th.join(timeout=1)

        lime_time_txt = f"{t_lime:.2f} s" if t_lime is not None else "failed"
        note_ph.markdown(
            "<div class='note'>Feature contributions toward 'Positive' (Diabetes): "
            "positive weights increase the predicted probability, negative decrease it.<br>"
            f"Computation time &mdash; prediction: {t_pred*1000:.0f} ms &middot; "
            f"LIME explanation: {lime_time_txt}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Enter patient features on the left, "
            "then click **Predict** to view risk."
        )

    st.markdown("</div>", unsafe_allow_html=True)
