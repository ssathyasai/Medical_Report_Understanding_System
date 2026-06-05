"""Task 6: Positional Encoding — token & sentence position heatmaps"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
    return PE

def run(plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    MAX_LEN = 60
    D_MODEL = 64
    PE = positional_encoding(MAX_LEN, D_MODEL)

    print("=" * 55)
    print("TASK 6 — Positional Encoding (Medical Reports)")
    print("=" * 55)
    print(f"PE shape: {PE.shape}")
    for p in [0, 1, 2, 3]:
        print(f"  Position {p}: {PE[p, :6].round(4)}")

    # Full heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(PE, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    ax.set_title("Medical Report — Positional Encoding Heatmap (all 60 token positions)")
    ax.set_xlabel("Encoding Dimension"); ax.set_ylabel("Token Position")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/pe_full_heatmap.png"); plt.close()

    # Per-position slices (sentence / token positions)
    sample_report = "patient report indicates findings related to cardiology evaluation and treatment"
    tokens = sample_report.split()
    for pos in [0, 1, 2, 3, 5]:
        fig, ax = plt.subplots(figsize=(12, 1.8))
        ax.imshow(PE[pos:pos+1, :], aspect='auto', cmap='plasma')
        label = tokens[pos] if pos < len(tokens) else str(pos)
        ax.set_title(f"Token Position {pos}  ('{label}')")
        ax.set_xlabel("Encoding Dimension"); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/pe_token_{pos}.png"); plt.close()
        print(f"  Saved: pe_token_{pos}.png")

    print(f"\nAll PE heatmaps saved → {plots_dir}/")
    return PE

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
