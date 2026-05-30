"""
src/train_pipeline.py
=====================
Full ML training pipeline for the Complaint Auto-Routing System.

Steps:
  1. Load data/complaints.csv
  2. Generate embeddings (paraphrase-multilingual-mpnet-base-v2)
  3. Train Officer Routing model  (RandomForestClassifier)
  4. Train Priority model         (RandomForestClassifier)
  5. Train ETA model              (GradientBoostingRegressor)
  6. Build FAISS similarity index (IndexFlatIP, cosine similarity)
  7. Save all models + evaluation_report.json
"""

import os, sys, json, time, pickle
import numpy as np
import pandas as pd
import joblib
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, classification_report, confusion_matrix
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "complaints.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
EMBED_PATH = os.path.join(MODEL_DIR, "complaints_metadata.pkl")
EVAL_PATH  = os.path.join(BASE_DIR, "evaluation_report.json")
CM_PATH    = os.path.join(MODEL_DIR, "confusion_matrix.png")

os.makedirs(MODEL_DIR, exist_ok=True)

EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE      = 64
RANDOM_STATE    = 42


# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("\n[1/7] Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"  ✗ Dataset not found at {DATA_PATH}")
        print("  → Run: python data/data_generator.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    required = {"complaint_text", "officer", "priority", "eta_days"}
    missing  = required - set(df.columns)
    if missing:
        print(f"  ✗ Missing columns: {missing}")
        sys.exit(1)

    df = df.dropna(subset=["complaint_text", "officer", "priority", "eta_days"])
    df["complaint_text"] = df["complaint_text"].astype(str).str.strip()
    print(f"  ✓ Loaded {len(df)} complaints | "
          f"Officers: {df['officer'].nunique()} | Priorities: {df['priority'].nunique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
def generate_embeddings(texts: list) -> np.ndarray:
    print(f"\n[2/7] Generating embeddings with '{EMBEDDING_MODEL}'...")
    print(f"  (First run downloads ~420MB — cached locally for all future runs)")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Batch encoding with progress bar
    all_embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc="  Encoding", unit="batch", ncols=80):
        batch = texts[i:i + BATCH_SIZE]
        emb   = model.encode(batch, show_progress_bar=False,
                              convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings).astype("float32")
    elapsed    = time.time() - t0
    print(f"  ✓ Embeddings shape: {embeddings.shape}  ({elapsed:.1f}s)")
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
def encode_labels(df: pd.DataFrame):
    print("\n[3/7] Encoding labels...")
    le_officer  = LabelEncoder()
    le_priority = LabelEncoder()

    y_officer  = le_officer.fit_transform(df["officer"])
    y_priority = le_priority.fit_transform(df["priority"])
    y_eta      = df["eta_days"].values.astype(float)

    print(f"  Officers  : {list(le_officer.classes_)}")
    print(f"  Priorities: {list(le_priority.classes_)}")

    joblib.dump(le_officer,  os.path.join(MODEL_DIR, "label_encoder_officer.joblib"))
    joblib.dump(le_priority, os.path.join(MODEL_DIR, "label_encoder_priority.joblib"))
    print("  ✓ Label encoders saved")
    return y_officer, y_priority, y_eta, le_officer, le_priority


# ══════════════════════════════════════════════════════════════════════════════
def train_officer_model(X_train, X_test, y_train, y_test, le_officer):
    print("\n[4/7] Training Officer Routing model (RandomForest)...")
    t0  = time.time()
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds   = clf.predict(X_test)
    acc     = accuracy_score(y_test, preds)
    f1      = f1_score(y_test, preds, average="weighted")
    elapsed = time.time() - t0

    print(f"  ✓ Accuracy: {acc:.4f}  |  Weighted F1: {f1:.4f}  ({elapsed:.1f}s)")
    print(classification_report(y_test, preds,
                                 target_names=le_officer.classes_, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le_officer.classes_,
                yticklabels=le_officer.classes_, ax=ax)
    ax.set_title("Officer Routing — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=120)
    plt.close()
    print(f"  ✓ Confusion matrix saved → {CM_PATH}")

    joblib.dump(clf, os.path.join(MODEL_DIR, "officer_model.joblib"))
    print(f"  ✓ Model saved → models/officer_model.joblib")
    return {"officer_accuracy": round(acc, 4), "officer_f1_weighted": round(f1, 4)}


# ══════════════════════════════════════════════════════════════════════════════
def train_priority_model(X_train, X_test, y_train, y_test, le_priority):
    print("\n[5/7] Training Priority Prediction model (RandomForest)...")
    t0  = time.time()
    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds   = clf.predict(X_test)
    acc     = accuracy_score(y_test, preds)
    f1      = f1_score(y_test, preds, average="weighted")
    elapsed = time.time() - t0

    print(f"  ✓ Accuracy: {acc:.4f}  |  Weighted F1: {f1:.4f}  ({elapsed:.1f}s)")
    print(classification_report(y_test, preds,
                                 target_names=le_priority.classes_, zero_division=0))

    joblib.dump(clf, os.path.join(MODEL_DIR, "priority_model.joblib"))
    print(f"  ✓ Model saved → models/priority_model.joblib")
    return {"priority_accuracy": round(acc, 4), "priority_f1_weighted": round(f1, 4)}


# ══════════════════════════════════════════════════════════════════════════════
def train_eta_model(X_train, X_test, y_train, y_test):
    print("\n[6/7] Training ETA Prediction model (GradientBoosting)...")
    t0  = time.time()
    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=RANDOM_STATE
    )
    reg.fit(X_train, y_train)
    preds   = reg.predict(X_test)
    mae     = mean_absolute_error(y_test, preds)
    rmse    = mean_squared_error(y_test, preds) ** 0.5
    elapsed = time.time() - t0

    print(f"  ✓ MAE: {mae:.3f} days  |  RMSE: {rmse:.3f} days  ({elapsed:.1f}s)")

    joblib.dump(reg, os.path.join(MODEL_DIR, "eta_model.joblib"))
    print(f"  ✓ Model saved → models/eta_model.joblib")
    return {"eta_mae_days": round(mae, 3), "eta_rmse_days": round(rmse, 3)}


# ══════════════════════════════════════════════════════════════════════════════
def build_faiss_index(embeddings: np.ndarray, texts: list,
                       officers: list, priorities: list, etas: list):
    print("\n[7/7] Building FAISS similarity index (IndexFlatIP)...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          # Inner product on L2-normalized = cosine
    index.add(embeddings)

    faiss.write_index(index, os.path.join(MODEL_DIR, "faiss_index.bin"))
    print(f"  ✓ FAISS index saved  ({index.ntotal} vectors, dim={dim})")

    # Save metadata for retrieval display
    metadata = {
        "texts":      texts,
        "officers":   officers,
        "priorities": priorities,
        "etas":       etas,
    }
    with open(EMBED_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  ✓ Metadata saved → {EMBED_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
def compute_recall_at_k(X_test, y_officer_test, k=5):
    """Recall@K: top-K FAISS results contain at least one same-officer complaint."""
    index = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))
    with open(EMBED_PATH, "rb") as f:
        meta = pickle.load(f)

    train_officers = np.array(meta["officers"])
    hits = 0
    for i, query in enumerate(X_test):
        q = query.reshape(1, -1).astype("float32")
        _, indices = index.search(q, k + 1)
        retrieved  = [train_officers[idx] for idx in indices[0] if idx < len(train_officers)][:k]
        if y_officer_test[i] in retrieved:
            hits += 1

    recall = hits / len(X_test)
    print(f"\n  Recall@{k} (Similarity Search): {recall:.4f}")
    return {f"similarity_recall_at_{k}": round(recall, 4)}


# ══════════════════════════════════════════════════════════════════════════════
def main():
    total_start = time.time()
    print("=" * 60)
    print("  Complaint Auto-Routing — Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. Embed
    texts      = df["complaint_text"].tolist()
    embeddings = generate_embeddings(texts)

    # 3. Encode labels
    y_officer, y_priority, y_eta, le_officer, le_priority = encode_labels(df)

    # 4. Train/test split (80/20 stratified on officer)
    idx        = np.arange(len(embeddings))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, stratify=y_officer, random_state=RANDOM_STATE
    )
    X_train, X_test = embeddings[train_idx], embeddings[test_idx]
    print(f"\n  Train: {len(train_idx)} | Test: {len(test_idx)}")

    # 5. Train models
    metrics = {}
    metrics.update(train_officer_model(
        X_train, X_test, y_officer[train_idx], y_officer[test_idx], le_officer))
    metrics.update(train_priority_model(
        X_train, X_test, y_priority[train_idx], y_priority[test_idx], le_priority))
    metrics.update(train_eta_model(
        X_train, X_test, y_eta[train_idx], y_eta[test_idx]))

    # 6. Build FAISS (on full dataset for richer retrieval)
    build_faiss_index(
        embeddings,
        texts,
        df["officer"].tolist(),
        df["priority"].tolist(),
        df["eta_days"].tolist(),
    )

    # 7. Recall@K
    metrics.update(compute_recall_at_k(X_test, y_officer[test_idx], k=5))

    # 8. Save evaluation report
    report = {
        "model":   EMBEDDING_MODEL,
        "dataset": DATA_PATH,
        "train_samples": int(len(train_idx)),
        "test_samples":  int(len(test_idx)),
        "metrics": metrics,
        "training_time_sec": round(time.time() - total_start, 1),
    }
    with open(EVAL_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Officer  F1  : {metrics['officer_f1_weighted']:.4f}")
    print(f"  Priority F1  : {metrics['priority_f1_weighted']:.4f}")
    print(f"  ETA MAE      : {metrics['eta_mae_days']:.3f} days")
    print(f"  Recall@5     : {metrics['similarity_recall_at_5']:.4f}")
    print(f"\n  Report saved → {EVAL_PATH}")
    print(f"  Total time   : {report['training_time_sec']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
