from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# ECOSORT AI — Smart Waste Segregation
# Requires ecosort_best.keras in the same folder as app.py.
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "ecosort_best.keras"
HISTORY_PATH = APP_DIR / "prediction_history.csv"

st.set_page_config(
    page_title="EcoSort AI | Smart Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Styling -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --eco-green: #16794b;
        --eco-dark: #103b2b;
        --eco-mint: #e8f5ed;
        --eco-bg: #f7faf7;
    }

    .stApp {
        background: var(--eco-bg);
        font-family: 'DM Sans', sans-serif;
        color: #20352a;
    }

    [data-testid="stSidebar"] {
        background: #103b2b;
    }

    [data-testid="stSidebar"] * {
        color: #f5fff8 !important;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif;
        color: #103b2b !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #334b3c;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #103b2b 0%, #16794b 65%, #56a875 100%);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 2.55rem;
        margin: 0.2rem 0 0.5rem 0;
    }

    .hero p {
        color: #e4f6e9 !important;
        font-size: 1.05rem;
        max-width: 760px;
    }

    .pill {
        display: inline-block;
        background: #d6f3df;
        color: #135c39;
        padding: 5px 11px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #dcebe0;
        border-radius: 18px;
        padding: 1.5rem;
        color: #20352a !important;
        box-shadow: 0 4px 16px rgba(16, 59, 43, 0.06);
    }

    .result-card h2 {
        color: #103b2b !important;
        font-size: 28px;
        margin-bottom: 12px;
    }

    .result-card p {
        color: #52665a !important;
        font-size: 15px;
    }

    .result-card .score {
        color: #103b2b;
        font-size: 42px;
        font-weight: 800;
        margin-top: 8px;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dcebe0;
        padding: 16px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #52665a !important;
    }

    div[data-testid="stMetricValue"] {
        color: #103b2b !important;
    }

    div.stButton > button[kind="primary"] {
        background: #16794b;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #105c37;
        color: #ffffff;
        border: none;
    }

    div.stButton > button {
        border-radius: 10px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stProgress"] > div > div {
        background-color: #16794b;
    }

    .small-muted {
        color: #65776b;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------- Model and history helpers -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Place ecosort_best.keras beside app.py."
        )
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def append_history(record):
    file_exists = HISTORY_PATH.exists()
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def read_history():
    if HISTORY_PATH.exists():
        try:
            return pd.read_csv(HISTORY_PATH)
        except (pd.errors.EmptyDataError, OSError):
            return pd.DataFrame()
    return pd.DataFrame()


def predict_image(loaded_model, image):
    resized = image.convert("RGB").resize((224, 224))
    array = np.asarray(resized, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    # Model has an internal Rescaling(1./255) layer.
    recyclable_score = float(
        loaded_model.predict(array, verbose=0).reshape(-1)[0]
    )
    recyclable_score = float(np.clip(recyclable_score, 0.0, 1.0))
    organic_score = 1.0 - recyclable_score

    # Dataset mapping: O=0 (Organic), R=1 (Recyclable).
    label = "Recyclable" if recyclable_score >= 0.5 else "Organic"
    confidence = recyclable_score if label == "Recyclable" else organic_score

    return label, confidence, organic_score, recyclable_score


def save_prediction(uploaded_name, label, confidence, organic_score, recyclable_score):
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "filename": uploaded_name,
        "prediction": label,
        "score_percent": round(confidence * 100, 2),
        "organic_score_percent": round(organic_score * 100, 2),
        "recyclable_score_percent": round(recyclable_score * 100, 2),
    }
    append_history(record)


def hero(tag, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <span class="pill">{tag}</span>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------- Sidebar navigation -----------------------------
st.sidebar.markdown("# ♻️ EcoSort AI")
st.sidebar.caption("Smart campus waste segregation")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Home & Predict", "Evaluation Dashboard", "Prediction History", "Waste Guide"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Course Project**")
st.sidebar.caption("Environment, Sustainability & Campus Life")
st.sidebar.caption("CNN image classification")
st.sidebar.markdown("---")
st.sidebar.caption("EcoSort AI • CNN-based image classification")

# ----------------------------- Home and prediction -----------------------------
if page == "Home & Predict":
    hero(
        "AI FOR A CLEANER CAMPUS",
        "Sort smarter. Waste less.",
        "Upload a waste image and let EcoSort AI estimate whether it belongs to the "
        "Organic or Recyclable category. Better sorting can support a more sustainable campus.",
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Test accuracy", "86.55%")
    metric2.metric("Macro F1-score", "86.47%")
    metric3.metric("Test ROC-AUC", "94.08%")
    st.caption("Recorded evaluation results from the project's 2,513-image test set.")

    st.markdown("## Try the classifier")
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        image = None

        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
                st.image(
                    image,
                    caption="Image ready for analysis",
                    width="stretch",
                )
            except Exception:
                st.error("This file could not be read as an image.")

        if image is None:
            st.info(
                "Upload a clear image of one waste item. Good lighting and a simple "
                "background may help."
            )

        classify = st.button(
            "♻️ Classify waste",
            type="primary",
            disabled=image is None,
            width="stretch",
        )

    with right:
        st.markdown("### How it works")
        st.markdown(
            """
            1. **Upload** a JPG, PNG, or WEBP image.
            2. **Analyze** it with the trained CNN.
            3. **Review** the predicted category and model score.
            4. **Check local guidance** before disposal.
            """
        )
        st.warning(
            "The model predicts only two broad image classes. It may misclassify "
            "unfamiliar objects, mixed waste, or unclear images."
        )

    if classify and image is not None and uploaded is not None:
        try:
            with st.spinner("Analyzing image..."):
                loaded_model = load_model()
                label, confidence, organic_score, recyclable_score = predict_image(
                    loaded_model, image
                )

            st.markdown("---")
            st.markdown("## Prediction result")
            result_image_col, result_col = st.columns([0.8, 1.2], gap="large")

            with result_image_col:
                st.image(
                    image,
                    caption="Classified image",
                    width="stretch",
                )

            with result_col:
                symbol = "🌱" if label == "Organic" else "♻️"
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div style="color:#16794b;font-size:13px;font-weight:700;
                                    letter-spacing:1px;margin-bottom:12px;">
                            CLASSIFICATION RESULT
                        </div>
                        <h2>{symbol} {label} Waste</h2>
                        <p>Model score for the predicted class</p>
                        <div class="score">{confidence * 100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(confidence)
                st.caption(
                    "This is the model's output score, not a calibrated probability "
                    "or a guarantee of correctness."
                )
                st.write(f"🌱 Organic score: **{organic_score * 100:.1f}%**")
                st.write(f"♻️ Recyclable score: **{recyclable_score * 100:.1f}%**")

                if label == "Organic":
                    st.info(
                        "If this is food or garden waste, use the campus organic/compost "
                        "bin where accepted. Keep plastic and other contaminants out."
                    )
                else:
                    st.info(
                        "Check the item's material and local recycling rules. Empty and "
                        "rinse containers where appropriate; not every plastic item is recyclable."
                    )

            # Save once for each explicit classification click.
            save_prediction(
                uploaded.name,
                label,
                confidence,
                organic_score,
                recyclable_score,
            )
            st.success("Prediction added to history.")

        except Exception as exc:
            st.error(f"Could not run the model: {exc}")
            st.info(
                "Check that ecosort_best.keras is beside app.py and that TensorFlow is installed."
            )

# ----------------------------- Evaluation dashboard -----------------------------
elif page == "Evaluation Dashboard":
    hero(
        "MODEL PERFORMANCE",
        "Evaluation dashboard",
        "Summary of EcoSort AI's recorded test-set performance.",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test accuracy", "86.55%")
    col2.metric("Macro precision", "86.38%")
    col3.metric("Macro recall", "86.81%")
    col4.metric("Macro F1-score", "86.47%")
    st.metric("ROC-AUC", "94.08%")
    st.caption(
        "These are the recorded metrics from the saved model evaluation. Re-run "
        "evaluation if you replace or retrain the model."
    )

    st.markdown("### Per-class metrics")
    metrics_df = pd.DataFrame(
        {
            "Class": ["Organic", "Recyclable"],
            "Precision": [0.9073, 0.8204],
            "Recall": [0.8451, 0.8912],
            "F1-score": [0.8751, 0.8543],
            "Support": [1401, 1112],
        }
    ).set_index("Class")

    st.dataframe(
        metrics_df.style.format(
            {
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1-score": "{:.2%}",
            }
        ),
        width="stretch",
    )

    st.markdown("### Precision, recall and F1-score")
    st.bar_chart(metrics_df[["Precision", "Recall", "F1-score"]], width="stretch")

    st.markdown("### Interpretation")
    st.write(
        "- Organic precision is 90.73%: when the model predicts Organic, it is correct "
        "relatively often on the test set."
    )
    st.write(
        "- Recyclable recall is 89.12%: the model detects about 89% of the recyclable "
        "examples in the test set."
    )
    st.write(
        "- ROC-AUC of 94.08% indicates good ranking separation across thresholds; "
        "it does not mean 94.08% classification accuracy."
    )
    st.warning(
        "The saved confusion-matrix image came from an earlier evaluation and may not "
        "match these latest metrics, so it is intentionally not shown here."
    )

# ----------------------------- Prediction history -----------------------------
elif page == "Prediction History":
    hero(
        "RECENT ACTIVITY",
        "Prediction history",
        "Review classifications saved by this app.",
    )

    history_df = read_history()
    if history_df.empty:
        st.info(
            "No predictions have been saved yet. Go to Home & Predict and classify an image."
        )
    else:
        st.metric("Saved predictions", len(history_df))
        st.dataframe(
            history_df.iloc[::-1],
            width="stretch",
            hide_index=True,
        )
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download history as CSV",
            data=csv_data,
            file_name="ecosort_prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )

        if st.button("Clear prediction history"):
            HISTORY_PATH.unlink(missing_ok=True)
            st.success("Prediction history cleared.")
            st.rerun()

    st.caption(
        "History is stored in prediction_history.csv beside the app. Some cloud hosts "
        "may reset local files when the app restarts or redeploys."
    )

# ----------------------------- Waste guide -----------------------------
elif page == "Waste Guide":
    hero(
        "DISPOSE RESPONSIBLY",
        "Waste sorting guide",
        "Use the prediction as a starting point, then follow local campus and municipal rules.",
    )

    organic_col, recyclable_col = st.columns(2, gap="large")

    with organic_col:
        st.markdown("## 🌱 Organic waste")
        st.markdown(
            """
            **Examples:** fruit and vegetable scraps, leftover food, coffee grounds, leaves.

            **Good practice**
            - Use the designated compost or organic bin if available.
            - Keep plastic wrappers, glass, and metal out.
            - Follow campus rules for cooked food, meat, dairy, and liquids.
            """
        )

    with recyclable_col:
        st.markdown("## ♻️ Potentially recyclable")
        st.markdown(
            """
            **Examples:** clean paper and cardboard, accepted plastic bottles, metal cans,
            and glass containers.

            **Good practice**
            - Empty containers and rinse when required.
            - Keep paper dry and free of food contamination.
            - Check material rules with your local collection service.
            - Batteries, e-waste, medical waste, and chemicals need separate handling.
            """
        )

    st.warning(
        "Not every item in the model's Recyclable class is accepted by every recycling "
        "facility. Follow your campus's actual waste policy."
    )
