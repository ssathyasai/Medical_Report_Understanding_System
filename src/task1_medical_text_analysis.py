"""Task 1: Medical Text Analysis"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import re, os

def run(data_path="data/medical_reports_500.csv", plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    print("=" * 55)
    print("TASK 1 — Medical Text Analysis")
    print("=" * 55)
    print(f"Total reports   : {len(df)}")
    print(f"\nSpecialty distribution:\n{df['specialty'].value_counts().to_string()}")

    # Most common medical terms
    all_text = " ".join(df["report_text"].astype(str)).lower()
    words    = re.findall(r'\b[a-z]{4,}\b', all_text)
    stop     = {"patient","report","indicates","findings","related","evaluation",
                "treatment","this","that","with","from","have","been","will","also"}
    words    = [w for w in words if w not in stop]
    top25    = Counter(words).most_common(25)
    print("\nTop 25 medical terms:")
    for w, c in top25:
        print(f"  {w:20s}: {c}")

    # Plot 1: Specialty distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    df["specialty"].value_counts().plot(kind="bar", ax=ax,
                                        color="steelblue", edgecolor="black")
    ax.set_title("Medical Specialty Distribution")
    ax.set_xlabel("Specialty"); ax.set_ylabel("Count")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(f"{plots_dir}/specialty_distribution.png"); plt.close()

    # Plot 2: Top medical terms
    wlabels, wcounts = zip(*top25[:20])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(wlabels, wcounts, color="coral", edgecolor="black")
    ax.set_title("Top 20 Medical Terms")
    ax.set_xlabel("Term"); ax.set_ylabel("Frequency")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(f"{plots_dir}/medical_terms.png"); plt.close()

    print(f"\nPlots saved → {plots_dir}/")
    return df

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
