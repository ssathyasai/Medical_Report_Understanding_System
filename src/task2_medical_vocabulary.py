"""Task 2: Medical Vocabulary Builder"""
import pandas as pd
import re, json, os
from collections import Counter

def run(data_path="data/medical_reports_500.csv", artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    all_text = " ".join(df["report_text"].astype(str)).lower()
    all_text = re.sub(r'[^a-z\s]', ' ', all_text)
    words    = all_text.split()
    stop     = {"patient","report","indicates","findings","related","evaluation",
                "treatment","and","the","for","this","that","from","with","have",
                "been","will","also","are","its","was","not","but","has","can"}
    med_words = [w for w in words if w not in stop and len(w) >= 4]
    freq = Counter(med_words)

    with open(f"{artifacts_dir}/medical_vocab.json", "w") as f:
        json.dump(dict(freq.most_common(300)), f, indent=2)

    print("=" * 55)
    print("TASK 2 — Medical Vocabulary Builder")
    print("=" * 55)
    print(f"Unique medical terms : {len(freq)}")
    print("\nTop 30:")
    for w, c in freq.most_common(30):
        print(f"  {w:22s}: {c}")
    print(f"\nVocabulary saved → {artifacts_dir}/medical_vocab.json")
    return freq

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
