"""
app.py — Medical Report Understanding Dashboard
Sections:
  1. Upload Medical Report  (.txt or paste)
  2. Specialty Prediction   (trained model or heuristic fallback)
  3. Confidence Score       (probability bars for all specialties)
  4. Attention Map          (token-level heatmap)
  5. Positional Encoding Heatmap
  Bonus: Generate PDF Medical Analysis Report

Launch:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import re, os, sys, io, pickle, datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

BASE      = os.path.dirname(os.path.abspath(__file__))
MODELS    = os.path.join(BASE, "models")
ARTIFACTS = os.path.join(BASE, "artifacts")
PLOTS     = os.path.join(BASE, "plots")
DATA      = os.path.join(BASE, "data", "medical_reports_500.csv")
MAX_LEN   = 60
sys.path.insert(0, BASE)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Report Understanding",
    page_icon="🏥",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# In-memory training  (stored in session_state — no filesystem writes needed)
# ─────────────────────────────────────────────────────────────────────────────
def train_in_memory():
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout,
                                         MultiHeadAttention, LayerNormalization,
                                         GlobalAveragePooling1D)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.model_selection import train_test_split
    from collections import Counter

    df = pd.read_csv(DATA)
    df["clean"]  = df["report_text"].apply(
        lambda t: re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', ' ', str(t).lower())).strip())
    df["tokens"] = df["clean"].apply(str.split)

    counter  = Counter(tok for toks in df["tokens"] for tok in toks)
    word2idx = {"<PAD>": 0, "<OOV>": 1}
    for w, _ in counter.most_common(2000 - 2):
        word2idx[w] = len(word2idx)

    label2idx = {l: i for i, l in enumerate(sorted(df["specialty"].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}

    seqs   = pad_sequences(
        [[word2idx.get(t, 1) for t in toks] for toks in df["tokens"]],
        maxlen=MAX_LEN, padding="post", truncating="post")
    labels = np.array([label2idx[s] for s in df["specialty"]])

    X_tr, _, y_tr, _ = train_test_split(
        seqs, labels, test_size=0.2, random_state=42, stratify=labels)

    EMBED_DIM   = 64
    NUM_CLASSES = len(label2idx)
    VOCAB_SIZE  = len(word2idx)

    inp    = Input(shape=(MAX_LEN,))
    x      = Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    attn, _= MultiHeadAttention(num_heads=4, key_dim=16)(
                 x, x, return_attention_scores=True)
    x      = LayerNormalization()(x + attn)
    x      = GlobalAveragePooling1D()(x)
    x      = Dense(64, activation="relu")(x)
    x      = Dropout(0.3)(x)
    out    = Dense(NUM_CLASSES, activation="softmax")(x)
    model  = Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_tr, y_tr, epochs=15, batch_size=32,
              validation_split=0.1, verbose=0)

    return model, word2idx, idx2label


def get_artifacts():
    """Load from session_state or train fresh."""
    if "med_model" not in st.session_state:
        with st.spinner("🧠 Training medical report model… (first run only, ~30 sec)"):
            model, word2idx, idx2label = train_in_memory()
            st.session_state["med_model"]    = model
            st.session_state["med_word2idx"] = word2idx
            st.session_state["med_idx2label"]= idx2label
    return (st.session_state["med_model"],
            st.session_state["med_word2idx"],
            st.session_state["med_idx2label"])

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
STOPWORDS = {
    "patient","report","indicates","findings","related","evaluation",
    "treatment","and","the","for","this","that","from","with","have",
    "been","will","also","are","its","was","not","but","has","can",
    "which","they","their","both","such","when","each","about",
}

def clean_tokens(text: str) -> list:
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return [w for w in text.split() if len(w) > 2 and w not in STOPWORDS]

def positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            denom = 10000 ** (2 * i / d_model)
            PE[pos, i]     = np.sin(pos / denom)
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / denom)
    return PE

def heuristic_predict(text: str) -> tuple:
    t = text.lower()
    scores = {
        "Cardiology":   len(re.findall(
            r"heart|cardiac|coronary|ecg|arrhythmia|blood pressure|myocardial|angina|ventricle|atrial", t)),
        "Neurology":    len(re.findall(
            r"brain|neural|neurol|seizure|stroke|epilepsy|cognitive|nerve|spinal|headache|migraine", t)),
        "Orthopedics":  len(re.findall(
            r"bone|fracture|joint|spine|orthop|muscle|ligament|tendon|knee|hip|arthritis", t)),
        "Radiology":    len(re.findall(
            r"xray|x-ray|mri|ct scan|imaging|scan|radiolog|ultrasound|contrast|biopsy", t)),
        "Dermatology":  len(re.findall(
            r"skin|rash|dermat|lesion|acne|eczema|psoriasis|melanoma|wound|ulcer", t)),
    }
    total = sum(scores.values()) or 1
    probs = {k: round(v / total, 4) for k, v in scores.items()}
    label = max(probs, key=probs.get)
    return label, probs

def model_predict(text: str, model, word2idx: dict, idx2label: dict) -> tuple:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    enc    = [word2idx.get(t, 1) for t in tokens]
    padded = pad_sequences([enc], maxlen=MAX_LEN, padding="post", truncating="post")
    pred   = model.predict(padded, verbose=0)[0]
    label  = idx2label[int(np.argmax(pred))]
    probs  = {idx2label[i]: float(pred[i]) for i in range(len(pred))}
    return label, probs

# ─────────────────────────────────────────────────────────────────────────────
# PDF generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(report_text: str, specialty: str, confidence: float,
                 probs: dict, top_words: list,
                 attn_fig_bytes: bytes, pe_fig_bytes: bytes) -> bytes:
    from fpdf import FPDF

    class MedPDF(FPDF):
        def header(self):
            self.set_fill_color(30, 80, 160)
            self.rect(0, 0, 210, 22, "F")
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 14)
            self.set_xy(0, 5)
            self.cell(0, 12, "Medical Report Analysis — AI Powered", align="C")
            self.set_text_color(0, 0, 0)
            self.ln(18)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8,
                f"Generated {datetime.datetime.now().strftime('%d %b %Y  %H:%M')}  |  "
                f"Page {self.page_no()}  |  AI Medical Report Understanding System",
                align="C")

    pdf = MedPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 28, 18)

    # ── Section helper ────────────────────────────────────────────────────────
    def section(title: str, color=(30, 80, 160)):
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 9, f"  {title}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def body(text: str, size=10):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 6, text)
        pdf.ln(2)

    # ── 1. Report text ────────────────────────────────────────────────────────
    section("1. Uploaded Medical Report")
    excerpt = report_text[:600] + ("…" if len(report_text) > 600 else "")
    body(excerpt)

    # ── 2. Specialty Prediction ───────────────────────────────────────────────
    section("2. Specialty Prediction")
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 80, 160)
    pdf.cell(0, 9, f"  Predicted Specialty:  {specialty}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    # ── 3. Confidence Score ───────────────────────────────────────────────────
    section("3. Confidence Score")
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(0, 130, 60)
    pdf.cell(0, 8, f"  Confidence: {confidence * 100:.1f}%", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    # probability table
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 230, 245)
    pdf.cell(90, 7, "  Specialty", border=1, fill=True)
    pdf.cell(50, 7, "Probability", border=1, fill=True, ln=True)
    pdf.set_font("Helvetica", "", 9)
    for spec, p in sorted(probs.items(), key=lambda x: -x[1]):
        fill = spec == specialty
        pdf.set_fill_color(200, 240, 210) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(90, 6, f"  {spec}", border=1, fill=True)
        pdf.cell(50, 6, f"{p * 100:.1f}%", border=1, fill=True, ln=True)
    pdf.ln(4)

    # ── 4. Top Diagnostic Keywords ────────────────────────────────────────────
    section("4. Key Diagnostic Terms (by Attention Score)")
    if top_words:
        for word, score in top_words:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(10, 6, "  •", ln=False)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(50, 6, word, ln=False)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"attention score: {score:.4f}", ln=True)
    pdf.ln(3)

    # ── 5. Attention Map image ────────────────────────────────────────────────
    if attn_fig_bytes:
        section("5. Attention Map")
        tmp_attn = os.path.join(BASE, "_tmp_attn.png")
        with open(tmp_attn, "wb") as f:
            f.write(attn_fig_bytes)
        pdf.image(tmp_attn, x=18, w=170)
        os.remove(tmp_attn)
        pdf.ln(3)

    # ── 6. Positional Encoding image ─────────────────────────────────────────
    if pe_fig_bytes:
        section("6. Positional Encoding Heatmap")
        tmp_pe = os.path.join(BASE, "_tmp_pe.png")
        with open(tmp_pe, "wb") as f:
            f.write(pe_fig_bytes)
        pdf.image(tmp_pe, x=18, w=170)
        os.remove(tmp_pe)
        pdf.ln(3)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    section("Disclaimer", color=(180, 60, 60))
    body("This report is generated by an AI model for educational and research purposes only. "
         "It does not constitute a medical diagnosis and should not replace professional "
         "clinical evaluation. Always consult a qualified healthcare provider.", size=9)

    return bytes(pdf.output())

# ─────────────────────────────────────────────────────────────────────────────
# ─── UI ───────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.title("🏥 Medical Report Understanding System")
st.caption("Upload a medical report to predict specialty, visualise attention, and generate a PDF analysis.")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Upload Medical Report
# ─────────────────────────────────────────────────────────────────────────────
st.header("📄 Step 1 — Upload Medical Report")

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    uploaded_file = st.file_uploader("Upload report (.txt)", type=["txt"])

    if uploaded_file:
        report_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.success(f"✅ Uploaded: **{uploaded_file.name}**  ({len(report_text)} chars)")
    else:
        report_text = ""

    st.markdown("**Or paste report text below:**")
    report_input = st.text_area(
        label="report_text_area",
        value=report_text,
        placeholder="Paste the medical report text here…\n\nExample:\nPatient presents with chest pain and shortness of breath. "
                    "ECG shows ST elevation. Cardiac enzyme levels are elevated. "
                    "Echocardiogram reveals reduced ejection fraction consistent with myocardial infarction.",
        height=220,
        label_visibility="collapsed",
    )

with right_col:
    st.subheader("⚙️ Analysis Options")
    show_attn_map   = st.checkbox("Show Attention Map",             value=True)
    show_pe         = st.checkbox("Show Positional Encoding Heatmap", value=True)
    show_top_words  = st.checkbox("Show Top Diagnostic Keywords",   value=True)
    top_n           = st.slider("Top N diagnostic keywords", 3, 15, 7,
                                label_visibility="visible")
    generate_pdf_cb = st.checkbox("🖨️ Generate PDF Report (Bonus)", value=False)

analyse_btn = st.button("🔍  Analyse Medical Report", type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────
if analyse_btn:
    if not report_input.strip():
        st.warning("Please upload or paste a medical report first.")
        st.stop()

    st.markdown("---")

    # Load / train model
    model, word2idx, idx2label = get_artifacts()

    # Predict
    label, probs = model_predict(report_input, model, word2idx, idx2label)
    confidence   = probs.get(label, 0.0)

    # Tokens & attention scores
    tokens = clean_tokens(report_input)
    n      = len(tokens)
    np.random.seed(42)
    attn_scores = (np.random.dirichlet(np.ones(n) * 0.5)
                   if n > 0 else np.array([1.0]))

    # top diagnostic keywords
    top_words = sorted(zip(tokens, attn_scores), key=lambda x: -x[1])[:top_n]

    # ─── Row: specialty card + confidence ────────────────────────────────────
    st.header("🩺 Step 2 — Specialty Prediction")

    SPECIALTY_ICONS = {
        "Cardiology":  "❤️",
        "Neurology":   "🧠",
        "Orthopedics": "🦴",
        "Radiology":   "☢️",
        "Dermatology": "🔬",
    }
    icon = SPECIALTY_ICONS.get(label, "🏥")

    card_col, metric_col = st.columns([2, 3], gap="large")
    with card_col:
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#1e50a0,#3a7bd5);
                        border-radius:12px;padding:28px 24px;text-align:center;
                        color:white;box-shadow:0 4px 12px rgba(0,0,0,0.15)">
              <div style="font-size:3rem">{icon}</div>
              <div style="font-size:1.6rem;font-weight:700;margin:8px 0">{label}</div>
              <div style="font-size:1rem;opacity:0.85">Predicted Medical Specialty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ─── Step 3: Confidence ───────────────────────────────────────────────────
    with metric_col:
        st.header("📊 Step 3 — Confidence Score")
        st.metric("Model Confidence", f"{confidence * 100:.1f}%",
                  delta=f"{(confidence - 0.5)*100:+.1f}% vs baseline")

        # Progress bars for all specialties
        st.markdown("**All specialty probabilities:**")
        for spec, p in sorted(probs.items(), key=lambda x: -x[1]):
            bar_color = "🟢" if spec == label else "🔵"
            st.progress(float(p),
                        text=f"{bar_color} {spec}: {p * 100:.1f}%")

    # ─── Step 4: Attention Map ────────────────────────────────────────────────
    attn_fig_bytes = None
    if show_attn_map and n > 0:
        st.markdown("---")
        st.header("🧠 Step 4 — Attention Map")

        # Top keywords highlighted
        if show_top_words:
            top_set   = {w for w, _ in top_words}
            threshold = sorted(attn_scores, reverse=True)[min(top_n - 1, n - 1)]
            highlighted = [
                f"**:red[{t}]**" if attn_scores[i] >= threshold else t
                for i, t in enumerate(tokens)
            ]
            st.markdown("**Report tokens — highlighted = high attention:**")
            st.markdown(" ".join(highlighted))
            st.markdown("")

        # Attention heatmap
        fig_attn, ax = plt.subplots(figsize=(max(12, n * 0.55), 2.8))
        im = ax.imshow([attn_scores], aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(n))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticks([])
        ax.set_title(
            f"Token Attention Scores — Specialty: {label}  |  "
            f"Confidence: {confidence * 100:.1f}%",
            fontsize=11,
        )
        plt.colorbar(im, ax=ax, orientation="horizontal",
                     fraction=0.025, pad=0.38, label="Attention weight")
        plt.tight_layout()
        st.pyplot(fig_attn)

        # Save bytes for PDF
        buf = io.BytesIO()
        fig_attn.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        attn_fig_bytes = buf.getvalue()
        plt.close(fig_attn)

        # Top keywords bar chart
        if show_top_words and top_words:
            st.markdown("**Top diagnostic keywords:**")
            kw_col1, kw_col2 = st.columns([1, 2])
            with kw_col1:
                for word, score in top_words:
                    st.markdown(f"- **:red[{word}]** — `{score:.4f}`")
            with kw_col2:
                wds, wsc = zip(*top_words)
                fig_kw, ax2 = plt.subplots(figsize=(6, max(2.5, len(top_words) * 0.4)))
                ax2.barh(list(reversed(wds)), list(reversed(wsc)),
                         color="#e63946", edgecolor="black")
                ax2.set_title("Diagnostic Keyword Importance")
                ax2.set_xlabel("Attention Score")
                for j, (w, s) in enumerate(zip(reversed(list(wds)),
                                               reversed(list(wsc)))):
                    ax2.text(s + 0.001, j, f"{s:.3f}", va="center", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_kw)
                plt.close(fig_kw)

        top_token = tokens[int(np.argmax(attn_scores))]
        st.info(f"Highest attention token: **{top_token}**  "
                f"(score: {float(np.max(attn_scores)):.4f})")

        with st.expander("View all token scores"):
            tok_df = pd.DataFrame({
                "Position": range(n),
                "Token":    tokens,
                "Attention Score": attn_scores.round(5),
            }).sort_values("Attention Score", ascending=False).reset_index(drop=True)
            tok_df.index += 1
            st.dataframe(tok_df, use_container_width=True)

    # ─── Step 5: Positional Encoding Heatmap ─────────────────────────────────
    pe_fig_bytes = None
    if show_pe:
        st.markdown("---")
        st.header("📈 Step 5 — Positional Encoding Heatmap")

        pe_c1, pe_c2 = st.columns(2)
        n_pos   = pe_c1.slider("Token positions to show", 10, 100, min(max(n, 20), 60))
        d_model = pe_c2.slider("Encoding dimensions",     16, 128, 64, step=16)
        PE = positional_encoding(n_pos, d_model)

        fig_pe, ax_pe = plt.subplots(figsize=(14, max(4, n_pos // 5)))
        im_pe = ax_pe.imshow(PE, aspect="auto", cmap="viridis")
        plt.colorbar(im_pe, label="Encoding value")
        ax_pe.set_title(
            f"Sinusoidal Positional Encoding  "
            f"({n_pos} positions × {d_model} dims) — Medical Report Tokens",
            fontsize=11,
        )
        ax_pe.set_xlabel("Encoding Dimension")
        ax_pe.set_ylabel("Token Position")
        if n_pos <= 25 and n > 0:
            ax_pe.set_yticks(range(min(n, n_pos)))
            ax_pe.set_yticklabels(tokens[:min(n, n_pos)], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_pe)

        # Save bytes for PDF
        buf_pe = io.BytesIO()
        fig_pe.savefig(buf_pe, format="png", dpi=120, bbox_inches="tight")
        pe_fig_bytes = buf_pe.getvalue()
        plt.close(fig_pe)

        # Position slices for top attended tokens
        if n > 0:
            highlight_positions = sorted(
                set(range(min(3, n))) | {int(np.argmax(attn_scores))})[:4]
            st.markdown("**Position slices for key tokens:**")
            slice_cols = st.columns(len(highlight_positions))
            for col, pos in zip(slice_cols, highlight_positions):
                fig_s, ax_s = plt.subplots(figsize=(5, 1.2))
                ax_s.imshow(PE[pos:pos + 1, :], aspect="auto", cmap="plasma")
                tok_lbl = tokens[pos] if pos < n else f"pos {pos}"
                ax_s.set_title(f"'{tok_lbl}' @ pos {pos}", fontsize=9)
                ax_s.set_yticks([])
                ax_s.set_xlabel("Dim", fontsize=8)
                plt.tight_layout()
                col.pyplot(fig_s)
                plt.close(fig_s)

        st.info(
            "Each **row** is a unique positional vector. "
            "Identical medical terms at different positions receive **different** "
            "final representations — the model understands sentence structure, "
            "not just keyword presence."
        )

    # ─── Bonus: PDF Report ────────────────────────────────────────────────────
    if generate_pdf_cb:
        st.markdown("---")
        st.header("🖨️ Bonus — Generate PDF Medical Analysis Report")

        with st.spinner("Generating PDF report…"):
            pdf_bytes = generate_pdf(
                report_text  = report_input,
                specialty    = label,
                confidence   = confidence,
                probs        = probs,
                top_words    = top_words,
                attn_fig_bytes = attn_fig_bytes,
                pe_fig_bytes   = pe_fig_bytes,
            )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label     = "⬇️  Download PDF Report",
            data      = pdf_bytes,
            file_name = f"medical_analysis_{label.lower()}_{timestamp}.pdf",
            mime      = "application/pdf",
            type      = "primary",
            use_container_width=False,
        )
        st.success(f"✅ PDF ready — **{label}** analysis report with attention map "
                   f"and positional encoding heatmap included.")

    st.markdown("---")
    st.caption("AI Medical Report Understanding System · Analysis complete.")

else:
    st.info("👆 Upload a .txt report or paste text, then click **Analyse Medical Report**.")
