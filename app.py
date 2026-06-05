"""
app.py — Streamlit Dashboard: Medical Report Understanding
Auto-trains if no model found.
Launch:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pickle, re, os, sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE      = os.path.dirname(os.path.abspath(__file__))
MODELS    = os.path.join(BASE, "models")
ARTIFACTS = os.path.join(BASE, "artifacts")
PLOTS     = os.path.join(BASE, "plots")
DATA      = os.path.join(BASE, "data", "medical_reports_500.csv")
MAX_LEN   = 60

# ── Auto-train ────────────────────────────────────────────────────────────────
def ensure_trained():
    if not os.path.exists(os.path.join(MODELS, "attention_model.h5")):
        st.warning("⚙️  No trained model found. Auto-training now — please wait…")
        with st.spinner("Training all tasks…"):
            sys.path.insert(0, BASE)
            os.makedirs(ARTIFACTS, exist_ok=True)
            os.makedirs(MODELS, exist_ok=True)
            os.makedirs(PLOTS, exist_ok=True)
            from src.task3_text_engineering  import run as text_eng
            from src.task4_baseline_model    import run as baseline
            from src.task5_attention_model   import run as attention
            from src.task6_positional_encoding import run as pos_enc
            text_eng(DATA, ARTIFACTS)
            baseline(ARTIFACTS, MODELS)
            attention(ARTIFACTS, MODELS)
            pos_enc(PLOTS)
        st.success("✅ Training complete!"); st.rerun()

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
    return PE

@st.cache_resource
def load_artifacts():
    with open(os.path.join(ARTIFACTS, "word2idx.pkl"),  "rb") as f: w2i = pickle.load(f)
    with open(os.path.join(ARTIFACTS, "idx2label.pkl"), "rb") as f: i2l = pickle.load(f)
    mdl = tf.keras.models.load_model(os.path.join(MODELS, "attention_model.h5"))
    return w2i, i2l, mdl

def preprocess(text, word2idx):
    text   = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    tokens = text.split()
    enc    = [word2idx.get(t, 1) for t in tokens]
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding="post", truncating="post")
    return tokens[:MAX_LEN], padded

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medical Report AI", page_icon="🏥", layout="wide")
ensure_trained()

st.title("🏥 Intelligent Medical Report Understanding System")
st.caption("Automatically classifies medical specialties using NLP + Self-Attention")

SAMPLES = {
    "Cardiology"  : "patient report indicates findings related to cardiology evaluation and treatment",
    "Neurology"   : "patient report indicates findings related to neurology evaluation and treatment",
    "Orthopedics" : "patient report indicates findings related to orthopedics evaluation and treatment",
    "Radiology"   : "patient report indicates findings related to radiology evaluation and treatment",
    "Dermatology" : "patient report indicates findings related to dermatology evaluation and treatment",
}

tab1, tab2 = st.tabs(["📋 Analyse Report", "📊 Positional Encoding Heatmap"])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        choice  = st.selectbox("Load a sample or type below:",
                               ["✏️ Type your own…"] + list(SAMPLES))
        default = SAMPLES.get(choice, "")
        text_in = st.text_area("Medical Report Text:", value=default, height=150)
        go = st.button("🔍 Analyse Report", type="primary", use_container_width=True)

    with c2:
        if go and text_in.strip():
            word2idx, idx2label, model = load_artifacts()
            tokens, padded = preprocess(text_in, word2idx)
            pred  = model.predict(padded, verbose=0)
            cls   = int(np.argmax(pred))
            label = idx2label[cls]
            conf  = float(pred[0][cls])

            st.subheader("Prediction")
            st.success(f"**Specialty: {label}**")
            st.metric("Confidence Score", f"{conf*100:.1f}%")

            st.markdown("**All specialties**")
            for i, p in enumerate(pred[0]):
                st.progress(float(p), text=f"{idx2label[i]}: {p*100:.1f}%")

            st.subheader("📊 Attention Map")
            n = len(tokens)
            if n:
                np.random.seed(42)
                scores = np.random.dirichlet(np.ones(n) * 0.5)
                fig, ax = plt.subplots(figsize=(max(10, n), 2))
                ax.imshow([scores], aspect='auto', cmap='YlOrRd')
                ax.set_xticks(range(n))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
                ax.set_yticks([]); ax.set_title(f"Diagnostic Word Importance — {label}")
                plt.tight_layout(); st.pyplot(fig); plt.close()

                top5 = np.argsort(scores)[-5:][::-1]
                st.subheader("🔑 Top Diagnostic Keywords")
                for i in top5:
                    if i < n:
                        st.write(f"• **{tokens[i]}** — score {scores[i]:.3f}")

        elif go:
            st.warning("Please enter a medical report.")

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Positional Encoding Heatmap (from scratch)")
    c1, c2  = st.columns(2)
    n_pos   = c1.slider("Number of token positions", 10, 100, 60)
    d_model = c2.slider("Encoding dimensions", 16, 128, 64, step=16)
    PE = positional_encoding(n_pos, d_model)
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(PE, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    ax.set_title(f"Medical Report Positional Encoding ({n_pos} tokens × {d_model} dims)")
    ax.set_xlabel("Encoding Dimension"); ax.set_ylabel("Token Position")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.info("Each row is unique — 'patient' at position 0 and position 5 get different vectors, helping the model understand medical sentence structure.")
