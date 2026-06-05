"""Task 4: Baseline Model — Embedding → Dense → Output"""
import numpy as np, pickle, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

MAX_LEN   = 60
EMBED_DIM = 64

def run(artifacts_dir="artifacts", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    sequences = np.load(f"{artifacts_dir}/sequences.npy")
    labels    = np.load(f"{artifacts_dir}/labels.npy")
    with open(f"{artifacts_dir}/word2idx.pkl",  "rb") as f: word2idx  = pickle.load(f)
    with open(f"{artifacts_dir}/label2idx.pkl", "rb") as f: label2idx = pickle.load(f)

    X_tr, X_te, y_tr, y_te = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels)

    model = Sequential([
        Embedding(len(word2idx), EMBED_DIM, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(64, activation="relu"), Dropout(0.3),
        Dense(len(label2idx), activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()
    model.fit(X_tr, y_tr, epochs=15, batch_size=32, validation_split=0.1, verbose=1)

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    names  = [k for k, _ in sorted(label2idx.items(), key=lambda x: x[1])]
    print(classification_report(y_te, y_pred, target_names=names, zero_division=0))

    model.save(f"{models_dir}/baseline_model.h5")
    print(f"Baseline model saved → {models_dir}/baseline_model.h5")
    return model

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
