"""Task 3-prep: Text Engineering — tokenise, pad, build vocab"""
import pandas as pd
import numpy as np
import re, pickle, os
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 2000
MAX_LEN    = 60

def clean(text):
    text = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    return re.sub(r'\s+', ' ', text).strip()

def run(data_path="data/medical_reports_500.csv", artifacts_dir="artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df["clean"]  = df["report_text"].apply(clean)
    df["tokens"] = df["clean"].apply(str.split)

    counter  = Counter(t for toks in df["tokens"] for t in toks)
    word2idx = {"<PAD>": 0, "<OOV>": 1}
    for w, _ in counter.most_common(VOCAB_SIZE - 2):
        word2idx[w] = len(word2idx)

    label2idx = {l: i for i, l in enumerate(sorted(df["specialty"].unique()))}
    idx2label = {v: k for k, v in label2idx.items()}

    sequences = pad_sequences(
        [[word2idx.get(t, 1) for t in toks] for toks in df["tokens"]],
        maxlen=MAX_LEN, padding="post", truncating="post")
    labels = np.array([label2idx[s] for s in df["specialty"]])

    np.save(f"{artifacts_dir}/sequences.npy", sequences)
    np.save(f"{artifacts_dir}/labels.npy",    labels)
    for name, obj in [("word2idx", word2idx),
                      ("label2idx", label2idx),
                      ("idx2label", idx2label)]:
        with open(f"{artifacts_dir}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    print(f"Vocab: {len(word2idx)} | Sequences: {sequences.shape} | Labels: {label2idx}")
    return sequences, labels, word2idx, label2idx, idx2label

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
