"""
src/evaluation.py
==================
Standalone evaluation script for the Complaint Auto-Routing System.
Loads trained models, runs 80/20 stratified split evaluation, and reports:
  - Officer Routing  : Accuracy, Weighted F1, confusion matrix
  - Priority         : Accuracy, Weighted F1
  - ETA              : MAE, RMSE
  - Similarity       : Recall@5
Saves evaluation_report.json and prints a formatted summary table.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import joblib
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, classification_report, confusion_matrix
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "complaints.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
EVAL_PATH  = os.path.join(BASE_DIR, "evaluation_report.json")
EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
RANDOM_STATE = 42
BATCH_SIZE   = 64


def load_artifacts():
    """Load all trained models and metadata."""
    print("[1/5] Loading trained models...")
    required = [
        "officer_model.joblib", "priority_model.joblib", "eta_model.joblib",
        "label_encoder_officer.joblib", "label_encoder_priority.joblib",
        "faiss_index.bin", "complaints_metadata.pkl"
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(MODEL_DIR, f))]
    if missing:
        print(f"\n  ✗ Missing model files: {missing}")
        print("  → Run: python src/train_pipeline.py")
        sys.exit(1)

    officer_model   = joblib.load(os.path.join(MODEL_DIR, "officer_model.joblib"))
    priority_model  = joblib.load(os.path.join(MODEL_DIR, "priority_model.joblib"))
    eta_model       = joblib.load(os.path.join(MODEL_DIR, "eta_model.joblib"))
    le_officer      = joblib.load(os.path.join(MODEL_DIR, "label_encoder_officer.joblib"))
    le_priority     = joblib.load(os.path.join(MODEL_DIR, "label_encoder_priority.joblib"))
    faiss_index     = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))

    with open(os.path.join(MODEL_DIR, "complaints_metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    print("  ✓ All models loaded")
    return officer_model, priority_model, eta_model, le_officer, le_priority, faiss_index, meta


def prepare_data():
    """Load dataset and generate embeddings."""
    print("\n[2/5] Loading dataset and generating embeddings...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["complaint_text", "officer", "priority", "eta_days"])
    texts = df["complaint_text"].astype(str).tolist()

    model = SentenceTransformer(EMBED_MODEL)
    all_emb = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Encoding", unit="batch", ncols=80):
        batch = texts[i:i + BATCH_SIZE]
        emb   = model.encode(batch, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
        all_emb.append(emb)

    embeddings = np.vstack(all_emb).astype("float32")
    print(f"  ✓ {len(texts)} embeddings generated | shape: {embeddings.shape}")
    return df, embeddings, texts


def evaluate_classification(model, le, X_test, y_test, task_name: str) -> dict:
    """Evaluate a classification model and return metrics dict."""
    preds   = model.predict(X_test)
    acc     = accuracy_score(y_test, preds)
    f1      = f1_score(y_test, preds, average="weighted", zero_division=0)
    f1_mac  = f1_score(y_test, preds, average="macro",    zero_division=0)

    print(f"\n  ── {task_name} ──────────────────────────────")
    print(f"    Accuracy (weighted) : {acc:.4f}")
    print(f"    F1 Score (weighted) : {f1:.4f}")
    print(f"    F1 Score (macro)    : {f1_mac:.4f}")
    print()
    print(classification_report(y_test, preds,
                                 target_names=le.classes_, zero_division=0))
    return {
        f"{task_name}_accuracy":    round(acc,    4),
        f"{task_name}_f1_weighted": round(f1,     4),
        f"{task_name}_f1_macro":    round(f1_mac, 4),
    }


def plot_confusion_matrix(model, le, X_test, y_test, title: str, save_path: str):
    preds = model.predict(X_test)
    cm    = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_,
                yticklabels=le.classes_, ax=ax)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Confusion matrix → {save_path}")


def evaluate_regression(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = mean_squared_error(y_test, preds) ** 0.5

    print(f"\n  ── ETA Prediction ────────────────────────────")
    print(f"    MAE  : {mae:.3f} days")
    print(f"    RMSE : {rmse:.3f} days")
    return {"eta_mae_days": round(mae, 3), "eta_rmse_days": round(rmse, 3)}


def evaluate_recall_at_k(faiss_index, X_test, y_officer_test,
                          meta: dict, k: int = 5) -> dict:
    """Recall@K: does the top-K retrieved set contain a complaint from same officer?"""
    train_officers = np.array(meta["officers"])
    hits = 0
    for i in tqdm(range(len(X_test)), desc=f"  Recall@{k}", ncols=80):
        q = X_test[i].reshape(1, -1)
        _, indices = faiss_index.search(q, k + 5)   # slight oversampling
        retrieved  = [train_officers[idx]
                      for idx in indices[0]
                      if 0 <= idx < len(train_officers)][:k]
        if y_officer_test[i] in retrieved:
            hits += 1

    recall = hits / len(X_test)
    print(f"\n  ── Similarity Retrieval ──────────────────────")
    print(f"    Recall@{k} : {recall:.4f}  ({hits}/{len(X_test)} queries)")
    return {f"similarity_recall_at_{k}": round(recall, 4)}


def print_summary(metrics: dict):
    print("\n" + "=" * 58)
    print("  EVALUATION SUMMARY")
    print("=" * 58)
    rows = [
        ("Officer Routing",    "Accuracy",  f"{metrics.get('officer_accuracy','-'):.4f}"),
        ("",                   "F1 (wt.)",  f"{metrics.get('officer_f1_weighted','-'):.4f}"),
        ("Priority Pred.",     "Accuracy",  f"{metrics.get('priority_accuracy','-'):.4f}"),
        ("",                   "F1 (wt.)",  f"{metrics.get('priority_f1_weighted','-'):.4f}"),
        ("ETA Regression",     "MAE",       f"{metrics.get('eta_mae_days','-'):.3f} days"),
        ("",                   "RMSE",      f"{metrics.get('eta_rmse_days','-'):.3f} days"),
        ("Similarity Search",  "Recall@5",  f"{metrics.get('similarity_recall_at_5','-'):.4f}"),
    ]
    for task, metric, value in rows:
        print(f"  {task:<22} {metric:<12} {value}")
    print("=" * 58)


def main():
    print("=" * 58)
    print("  Complaint Auto-Routing — Evaluation Script")
    print("=" * 58)

    officer_model, priority_model, eta_model, \
        le_officer, le_priority, faiss_index, meta = load_artifacts()

    df, embeddings, texts = prepare_data()

    y_officer  = le_officer.transform(df["officer"].tolist())
    y_priority = le_priority.transform(df["priority"].tolist())
    y_eta      = df["eta_days"].values.astype(float)

    idx        = np.arange(len(embeddings))
    _, test_idx = train_test_split(idx, test_size=0.2,
                                   stratify=y_officer, random_state=RANDOM_STATE)
    X_test = embeddings[test_idx]
    print(f"\n[3/5] Evaluating on {len(test_idx)} held-out test samples")

    metrics = {}
    metrics.update(evaluate_classification(
        officer_model, le_officer, X_test, y_officer[test_idx], "officer"))
    plot_confusion_matrix(
        officer_model, le_officer, X_test, y_officer[test_idx],
        "Officer Routing — Confusion Matrix",
        os.path.join(MODEL_DIR, "confusion_matrix_officer.png"))

    metrics.update(evaluate_classification(
        priority_model, le_priority, X_test, y_priority[test_idx], "priority"))
    plot_confusion_matrix(
        priority_model, le_priority, X_test, y_priority[test_idx],
        "Priority Prediction — Confusion Matrix",
        os.path.join(MODEL_DIR, "confusion_matrix_priority.png"))

    print("\n[4/5] ETA Regression")
    metrics.update(evaluate_regression(eta_model, X_test, y_eta[test_idx]))

    print("\n[5/5] Similarity Recall@5")
    metrics.update(evaluate_recall_at_k(faiss_index, X_test,
                                         le_officer.classes_[y_officer[test_idx]],
                                         meta, k=5))

    print_summary(metrics)

    report = {
        "embedding_model": EMBED_MODEL,
        "test_samples":    int(len(test_idx)),
        "metrics":         metrics,
    }
    with open(EVAL_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Report saved → {EVAL_PATH}")


if __name__ == "__main__":
    main()
