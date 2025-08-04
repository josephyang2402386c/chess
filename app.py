# imports
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Chess Outcome Predictor", page_icon="♟️")

# paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH   = BASE_DIR / "chess_winner_model.pkl" 
COLS_PATH    = BASE_DIR / "column_order.txt"
CLASSES_PATH = BASE_DIR / "winner_classes.txt"

# load
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH) as f:
        cols = [line.strip() for line in f]
    classes = np.loadtxt(CLASSES_PATH, dtype=str)
    return model, cols, classes

model, COLS, WINNER_CLASSES = load_artifacts()

# opening choices
OPENING_COLS = [c for c in COLS if c.startswith("opening_name_")]
OPENING_NAMES = ["Other"] + [c.replace("opening_name_", "") for c in OPENING_COLS if c != "opening_name_Other"]
OPENING_NAMES = ["Other"] + [n for n in OPENING_NAMES if n != "Other"] 

# stacked bar
def render_prob_bar(labels, probs):
    """
    Draw a single horizontal stacked bar for class probabilities (values 0..1).
    """
    fig, ax = plt.subplots(figsize=(6, 1.3))
    left = 0.0
    for lab, p in zip(labels, probs):
        ax.barh([0], [p], left=left, label=f"{lab} ({p*100:.1f}%)")
        left += p

    ax.set_xlim(0, 1)
    ax.set_yticks([])                      
    ax.set_xlabel("Probability")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.6), frameon=False)
    return fig

# ui
st.title("♟️ Chess Outcome Predictor")
st.write(
    "Enter **pre-game** information and the model will predict the winner (white / black / draw). "
    "The model uses player ratings, opening info, and time control to evaluate the probability of each game result"
)

col1, col2 = st.columns(2)
with col1:
    white_rating = st.number_input("White rating", min_value=0, max_value=3000, value=1000, step=50)
    black_rating = st.number_input("Black rating", min_value=0, max_value=3000, value=1000, step=50)
with col2:
    opening_ply = st.number_input("Opening devation (higher number means more moves before deviation from book moves)", min_value=1, max_value=20, value=6)
    base_time = st.number_input("Base time (minutes)", min_value=0, max_value=180, value=15)
    increment = st.number_input("Increment (seconds)", min_value=0, max_value=60, value=5)

opening_name = st.selectbox("Opening name", options=OPENING_NAMES, index=0)

# prediction
if st.button("Predict winner"):
    # check input
    if white_rating <= 0 or black_rating <= 0:
        st.error("Ratings must be positive.")
        st.stop()

    # preprocess
    elo_diff = int(white_rating) - int(black_rating)
    avg_elo = (int(white_rating) + int(black_rating)) / 2.0

    base_row = {
        "opening_ply": int(opening_ply),
        "elo_diff": elo_diff,
        "avg_elo": avg_elo,
        "base_time": int(base_time),
        "increment": int(increment),
    }

    # one hot encoding for openings
    row = base_row.copy()
    for col in OPENING_COLS:
        name = col.replace("opening_name_", "")
        row[col] = 1 if name == opening_name else 0

    # dataframe
    X_new = pd.DataFrame([row]).reindex(columns=COLS, fill_value=0)

    # result
    proba = model.predict_proba(X_new)[0]              
    labels = list(WINNER_CLASSES)                       
    pred_idx = int(np.argmax(proba))
    pred_label = str(labels[pred_idx])

    st.success(f"Predicted winner: **{pred_label}**")

    # table
    proba_df = pd.DataFrame({"Class": labels, "Probability (%)": (proba * 100)})
    st.write("Class probabilities:")
    st.dataframe(proba_df.style.format({"Probability (%)": "{:.1f}"}), use_container_width=True)

    # bar
    fig = render_prob_bar(labels, proba)
    st.pyplot(fig)

    # process
    with st.expander("Show model inputs (after preprocessing)"):
        st.dataframe(X_new)

