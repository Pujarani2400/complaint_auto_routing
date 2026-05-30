"""
src/inference_engine.py
========================
Singleton inference engine for the Complaint Auto-Routing System.
Loads all models once at startup; processes any complaint text into:
  - Assigned officer + confidence
  - Priority + confidence
  - ETA in days
  - Top-5 similar past complaints (FAISS cosine search)
  - XAI explanation (keyword analysis + domain evidence + historical evidence)
"""

import os
import re
import pickle
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# ── Officer domain keyword vocabulary (used by XAI engine) ─────────────────
OFFICER_KEYWORDS = {
    "Water Officer": [
        "water", "pipe", "pipeline", "burst", "leak", "leaking", "supply",
        "pressure", "tap", "main", "plumbing", "flood", "sewage", "dirty",
        "contamination", "drinking", "tank", "pump", "connection", "flow",
        "waterlogging", "main break", "no water", "water quality", "sediment",
    ],
    "Electrical Officer": [
        "light", "street light", "lamp", "electricity", "electric", "power",
        "outage", "transformer", "wiring", "wire", "signal", "traffic signal",
        "sparking", "flickering", "short circuit", "voltage", "meter",
        "blackout", "fuse", "generator", "load shedding", "substation",
    ],
    "Road Officer": [
        "pothole", "road", "street", "highway", "pavement", "sidewalk",
        "footpath", "crack", "cave-in", "manhole", "guardrail", "divider",
        "traffic", "gridlock", "intersection", "expressway", "broken road",
        "rough", "debris", "surface", "tarmac", "asphalt",
    ],
    "Sanitation Officer": [
        "garbage", "waste", "trash", "rubbish", "litter", "collection",
        "bin", "basket", "dumping", "illegal dump", "smell", "odor",
        "stench", "sweeping", "clean", "dirty", "refuse", "recycl",
        "overflowing bin", "sanitation", "mosquito", "dead animal",
    ],
    "Drainage Officer": [
        "drain", "drainage", "sewer", "sewage", "clog", "blocked",
        "flooding", "waterlog", "catch basin", "gutter", "manhole overflow",
        "stormwater", "pooling", "standing water", "backup", "overflow",
        "rain water", "basement flood", "grate",
    ],
}


class InferenceEngine:
    """Loads all models once; provides predict() for any complaint text."""

    _instance = None  # singleton

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir  = os.path.join(base_dir, "models")

        print("[InferenceEngine] Loading models...")

        # Embedding model
        self.embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        # Sklearn models
        self.officer_model   = joblib.load(os.path.join(model_dir, "officer_model.joblib"))
        self.priority_model  = joblib.load(os.path.join(model_dir, "priority_model.joblib"))
        self.eta_model       = joblib.load(os.path.join(model_dir, "eta_model.joblib"))

        # Label encoders
        self.le_officer  = joblib.load(os.path.join(model_dir, "label_encoder_officer.joblib"))
        self.le_priority = joblib.load(os.path.join(model_dir, "label_encoder_priority.joblib"))

        # FAISS index + metadata
        self.faiss_index = faiss.read_index(os.path.join(model_dir, "faiss_index.bin"))
        with open(os.path.join(model_dir, "complaints_metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)

        self._initialized = True
        print("[InferenceEngine] ✓ All models loaded and ready.")

    # ─────────────────────────────────────────────────────────────────────────
    def _embed(self, text: str) -> np.ndarray:
        """Generate L2-normalized 768-dim embedding for input text."""
        emb = self.embedder.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )
        return emb.astype("float32")

    # ─────────────────────────────────────────────────────────────────────────
    def _predict_officer(self, embedding: np.ndarray) -> tuple[str, float, dict]:
        """Return (officer_name, confidence%, all_probabilities)."""
        probs      = self.officer_model.predict_proba(embedding)[0]
        pred_idx   = int(np.argmax(probs))
        officer    = self.le_officer.classes_[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        all_probs  = {
            self.le_officer.classes_[i]: round(float(p) * 100, 1)
            for i, p in enumerate(probs)
        }
        return officer, round(confidence, 1), all_probs

    # ─────────────────────────────────────────────────────────────────────────
    def _predict_priority(self, embedding: np.ndarray) -> tuple[str, float]:
        """Return (priority_label, confidence%)."""
        probs      = self.priority_model.predict_proba(embedding)[0]
        pred_idx   = int(np.argmax(probs))
        priority   = self.le_priority.classes_[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        return priority, round(confidence, 1)

    # ─────────────────────────────────────────────────────────────────────────
    def _predict_eta(self, embedding: np.ndarray) -> int:
        """Return ETA as rounded integer days."""
        raw = self.eta_model.predict(embedding)[0]
        return max(1, int(round(float(raw))))

    # ─────────────────────────────────────────────────────────────────────────
    def _retrieve_similar(self, embedding: np.ndarray, top_k: int = 5) -> list:
        """Return top-K similar complaints from FAISS index."""
        scores, indices = self.faiss_index.search(embedding, top_k + 1)
        results = []
        texts      = self.metadata["texts"]
        officers   = self.metadata["officers"]
        priorities = self.metadata["priorities"]
        etas       = self.metadata["etas"]

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(texts):
                continue
            results.append({
                "text":       texts[idx],
                "officer":    officers[idx],
                "priority":   priorities[idx],
                "eta_days":   etas[idx],
                "similarity": round(float(score) * 100, 1),
            })
            if len(results) >= top_k:
                break
        return results

    # ─────────────────────────────────────────────────────────────────────────
    def _build_xai(self, text: str, officer: str, confidence: float,
                   all_probs: dict, similar: list) -> str:
        """
        Generate a human-readable explanation of why this officer was assigned.
        Uses keyword matching + similar complaint evidence (no external XAI lib).
        """
        text_lower = text.lower()

        # 1. Extract matched keywords for the predicted officer
        matched_kw = [
            kw for kw in OFFICER_KEYWORDS.get(officer, [])
            if kw.lower() in text_lower
        ]

        # 2. Count keyword hits per officer (for secondary evidence)
        officer_hits = {
            off: sum(1 for kw in kws if kw.lower() in text_lower)
            for off, kws in OFFICER_KEYWORDS.items()
        }
        ranked_officers = sorted(officer_hits.items(), key=lambda x: x[1], reverse=True)

        # 3. Similar complaint evidence
        same_officer_similar = [s for s in similar if s["officer"] == officer]

        lines = []
        lines.append(f"🎯 Assigned to: **{officer}** ({confidence:.1f}% confidence)")
        lines.append("")
        lines.append("**Why this officer?**")

        if matched_kw:
            kw_display = ", ".join(f'"{k}"' for k in matched_kw[:6])
            lines.append(f"  • Complaint contains {officer} domain keywords: {kw_display}")
        else:
            lines.append(f"  • Complaint semantically matches {officer} domain patterns")

        # Model confidence breakdown
        lines.append(f"  • ML model confidence: {confidence:.1f}% (RandomForest over 768-dim embedding)")

        # Secondary officer if close
        second = [(o, p) for o, p in all_probs.items() if o != officer]
        if second:
            second_sorted = sorted(second, key=lambda x: x[1], reverse=True)
            o2, p2 = second_sorted[0]
            if p2 > 15:
                lines.append(f"  • Closest alternative: {o2} ({p2:.1f}%) — ruled out by ML confidence")

        # Historical evidence
        if same_officer_similar:
            lines.append(f"  • {len(same_officer_similar)} of top-5 similar historical complaints "
                         f"were also routed to {officer}")
            top_sim = same_officer_similar[0]
            lines.append(f'  • Top match: "{top_sim["text"][:80]}..." '
                         f'({top_sim["similarity"]:.1f}% similar)')
        else:
            lines.append(f"  • Semantic embedding closely aligns with {officer} training examples")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, text: str, top_k: int = 5) -> dict:
        """
        Main inference entry point.

        Args:
            text:  Complaint text (any language supported by the embedding model)
            top_k: Number of similar complaints to retrieve

        Returns:
            dict with keys: officer, officer_confidence, all_officer_probs,
                            priority, priority_confidence, eta_days,
                            similar_complaints, explanation
        """
        text = text.strip()
        if not text:
            raise ValueError("Complaint text cannot be empty.")

        embedding = self._embed(text)

        officer, conf, all_probs = self._predict_officer(embedding)
        priority, pri_conf       = self._predict_priority(embedding)
        eta                      = self._predict_eta(embedding)
        similar                  = self._retrieve_similar(embedding, top_k)
        explanation              = self._build_xai(text, officer, conf, all_probs, similar)

        return {
            "officer":             officer,
            "officer_confidence":  conf,
            "all_officer_probs":   all_probs,
            "priority":            priority,
            "priority_confidence": pri_conf,
            "eta_days":            eta,
            "similar_complaints":  similar,
            "explanation":         explanation,
        }


# ── Module-level singleton accessor ──────────────────────────────────────────
_engine: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """Return the singleton InferenceEngine instance (lazy-loaded)."""
    global _engine
    if _engine is None:
        _engine = InferenceEngine()
    return _engine


def predict(text: str, top_k: int = 5) -> dict:
    """Convenience wrapper: get_engine().predict(text)."""
    return get_engine().predict(text, top_k)
