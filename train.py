"""
train.py — run ALL tasks for Medical Report System.
Usage:  python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

BASE      = os.path.dirname(__file__)
DATA      = os.path.join(BASE, "data",      "medical_reports_500.csv")
ARTIFACTS = os.path.join(BASE, "artifacts")
MODELS    = os.path.join(BASE, "models")
PLOTS     = os.path.join(BASE, "plots")

for d in [ARTIFACTS, MODELS, PLOTS]:
    os.makedirs(d, exist_ok=True)

from src.task1_medical_text_analysis import run as text_analysis
from src.task2_medical_vocabulary    import run as vocab
from src.task3_text_engineering      import run as text_eng
from src.task4_baseline_model        import run as baseline
from src.task5_attention_model       import run as attention
from src.task6_positional_encoding   import run as pos_enc
from src.task7_diagnostic_importance import run as diag

if __name__ == "__main__":
    print("\n[1/7] Medical Text Analysis")
    text_analysis(DATA, PLOTS)

    print("\n[2/7] Medical Vocabulary Builder")
    vocab(DATA, ARTIFACTS)

    print("\n[3/7] Text Engineering")
    text_eng(DATA, ARTIFACTS)

    print("\n[4/7] Baseline Model")
    baseline(ARTIFACTS, MODELS)

    print("\n[5/7] Attention Model")
    attention(ARTIFACTS, MODELS)

    print("\n[6/7] Positional Encoding")
    pos_enc(PLOTS)

    print("\n[7/7] Diagnostic Importance")
    diag(ARTIFACTS, MODELS, PLOTS)

    print("\n✅  All tasks complete.")
    print("   Launch dashboard:  streamlit run app.py")
