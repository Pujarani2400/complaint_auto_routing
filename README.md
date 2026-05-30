# Complaint Auto-Routing System

> **AI-powered Municipal Complaint Management** — Fully offline, ML-driven, Multilingual

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Offline](https://img.shields.io/badge/API-Zero%20External-green)](.)
[![License](https://img.shields.io/badge/License-MIT-yellow)](.)

---

## Overview

An end-to-end offline AI/ML system that processes citizen complaints (text, audio, or video) and automatically:

| Output | Model | Description |
|---|---|---|
| **Officer Assignment** | RandomForestClassifier | Routes to 1 of 5 municipal officers |
| **Priority Prediction** | RandomForestClassifier | High / Medium / Low |
| **ETA Prediction** | GradientBoostingRegressor | Resolution time in days |
| **Similar Complaints** | FAISS + Sentence Transformers | Top-5 semantically similar past complaints |
| **XAI Explanation** | Custom keyword + evidence engine | Human-readable reasoning for officer assignment |

---

## Architecture

```
Text / Audio / Video
        │
        ▼
Speech-to-Text  ──  faster-whisper (offline, 99 languages)
        │
        ▼
  Complaint Text
        │
        ▼
Multilingual Embedding  ──  paraphrase-multilingual-mpnet-base-v2 (768-dim)
        │
   ┌────┴──────────────┬──────────────────┐
   ▼                   ▼                  ▼
Officer Routing     Priority Pred.    ETA Prediction
(RandomForest)      (RandomForest)    (GradBoostReg.)
   │                   │                  │
   └────────┬──────────┘                  │
            │                             │
            ▼                             │
   FAISS Similarity Search ◄─────────────┘
   (Top-5 Past Complaints)
            │
            ▼
   XAI Explanation Engine
            │
            ▼
        Results Panel
```

---

## Officers & Domains

| Officer | NYC 311 Complaint Types |
|---|---|
| 💧 **Water Officer** | Water System, Plumbing, Drinking Water, Water Leak |
| ⚡ **Electrical Officer** | Street Light Condition, Electrical, Traffic Signal, Electric Utility |
| 🚧 **Road Officer** | Pothole, Street Condition, Sidewalk Condition, Highway Condition |
| 🗑️ **Sanitation Officer** | Missed Collection, Dirty Conditions, Illegal Dumping, Odor Complaint |
| 🌊 **Drainage Officer** | Sewer, Standing Water, Catch Basin, Flooding, Stormwater Drain |

---

## Dataset

- **Source taxonomy:** NYC 311 Service Request complaint types & descriptors
- **Generation strategy:** Narrative templates × location variation × urgency framing × synonym augmentation
- **Size:** ~1,030 complaints across 5 officer classes
- **Schema:** `complaint_text | officer | priority | eta_days`
- **ETA logic:** Priority × officer domain → deterministic range (not random)

---

## Project Structure

```
complaint_auto_routing/
│
├── data/
│   ├── data_generator.py       # NYC 311 taxonomy-driven dataset builder
│   └── complaints.csv          # Generated dataset (git-ignored if large)
│
├── models/                     # Auto-created by train_pipeline.py
│   ├── officer_model.joblib
│   ├── priority_model.joblib
│   ├── eta_model.joblib
│   ├── label_encoder_officer.joblib
│   ├── label_encoder_priority.joblib
│   ├── faiss_index.bin
│   ├── complaints_metadata.pkl
│   ├── confusion_matrix_officer.png
│   └── confusion_matrix_priority.png
│
├── src/
│   ├── train_pipeline.py       # Full ML training pipeline
│   ├── inference_engine.py     # Singleton inference + XAI
│   ├── media_processor.py      # Audio/Video → text (faster-whisper + ffmpeg)
│   └── evaluation.py           # Standalone evaluation script
│
├── app.py                      # Gradio web app (localhost:7860)
├── cli.py                      # CLI interface
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Prerequisites

**Python 3.9+** required.

**ffmpeg** (for audio/video input):
```bash
# Windows
# Download from https://ffmpeg.org/download.html → extract → add bin/ to PATH

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `paraphrase-multilingual-mpnet-base-v2` (~420MB) and `faster-whisper base` (~150MB) are downloaded **once** from HuggingFace on first run and cached locally. All subsequent runs are 100% offline.

### 3. Generate Dataset

```bash
python data/data_generator.py
```

Generates `data/complaints.csv` with ~1,030 municipal complaints.

### 4. Train Models

```bash
python src/train_pipeline.py
```

Trains all 3 models and builds the FAISS index. Saves to `models/`.
Training time: ~5–10 minutes (CPU, first time including embedding download).

### 5. Run the Web App

```bash
python app.py
```

Open: **http://localhost:7860**

### 6. Run CLI

```bash
python cli.py --text "Water pipe burst near the market"
python cli.py --audio complaint.wav
python cli.py --video complaint.mp4
```

### 7. Evaluate Models

```bash
python src/evaluation.py
```

---

## Evaluation Results

Results are saved to `evaluation_report.json` after training.

| Task | Metric | Target | Notes |
|---|---|---|---|
| Officer Routing | Weighted F1 | > 0.82 | 5-class classification |
| Priority Prediction | Weighted F1 | > 0.80 | 3-class: High/Medium/Low |
| ETA Regression | MAE | < 3 days | GradientBoosting on embeddings |
| Similarity Retrieval | Recall@5 | ≥ 0.75 | FAISS cosine search |

Confusion matrices saved to `models/confusion_matrix_officer.png` and `models/confusion_matrix_priority.png`.

---

## Multilingual Support

- `paraphrase-multilingual-mpnet-base-v2` supports **50+ languages** including Hindi, Bengali, Tamil, Urdu, Arabic, Spanish, French, German, and more.
- `faster-whisper` auto-detects and transcribes **99 languages** locally.
- No translation step needed — multilingual embeddings are cross-lingual by design.

**Test with a Hindi complaint:**
```bash
python cli.py --text "पानी की पाइप बाजार के पास फट गई है और पूरी सड़क में पानी भर गया है"
```

---

## XAI — Explainable AI

Every prediction includes a human-readable explanation:

```
🎯 Assigned to: Water Officer (94.2% confidence)

Why this officer?
  • Complaint contains Water Officer domain keywords: "pipe", "burst", "water supply"
  • ML model confidence: 94.2% (RandomForest over 768-dim embedding)
  • Closest alternative: Drainage Officer (3.1%) — ruled out by ML confidence
  • 4 of top-5 similar historical complaints were also routed to Water Officer
  • Top match: "Water main broke on Elm Street causing..." (96.3% similar)
```

---

## Technical Choices & Trade-offs

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Embedding | `paraphrase-multilingual-mpnet-base-v2` | `bge-m3` | Smaller (~420MB), 50+ languages, strong on complaint-style text |
| Classifier | RandomForest | Fine-tuned transformer | Fast inference, interpretable, no GPU needed |
| ETA | GradientBoosting | SVR / Ridge | Best MAE on tabular+embedding feature sets |
| Similarity | FAISS IndexFlatIP | ChromaDB | Lightest dependency, exact cosine search |
| STT | faster-whisper base | openai-whisper | 4× faster, 40% less RAM, same accuracy |
| UI | Gradio | Streamlit / Flask | Native audio/video upload, fastest setup |

---

## No External APIs

✅ No OpenAI  
✅ No Gemini  
✅ No AWS  
✅ No Google Cloud  
✅ All models run on local CPU  
✅ All data stays on your machine  

---

## License

MIT License — free to use, modify, and distribute.
