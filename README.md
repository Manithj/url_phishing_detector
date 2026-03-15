# URL Phishing Detector

An ensemble phishing URL detection system that combines **machine learning**, **deep learning**, and **generative AI** to classify URLs as legitimate or phishing. Three independent models run in parallel, and their predictions are fused through a weighted scoring orchestrator. A Streamlit web app and Python API are included for interactive and programmatic use.

## Features

- **Multi-model ensemble** — XGBoost (ML), bidirectional LSTM (DL), and Groq LLM (GenAI) work together for robust detection
- **Parallel inference** — all classifiers run concurrently via `ThreadPoolExecutor`
- **Weighted fusion** — final verdict from normalized confidence-weighted scores (GenAI 60%, DL 20%, ML 20%)
- **Automatic failover** — if GenAI is unavailable, remaining models are re-weighted automatically
- **Streamlit web UI** — dark-themed interface with confidence meters, per-model breakdowns, and session history
- **CLI and Python API** — run from the command line or embed in your own applications
- **Evaluation suite** — scripts to benchmark individual models and the full ensemble

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           PhishingOrchestrator           │
                    │         (Weighted Score Fusion)          │
                    └─────────────────┬───────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
     │  ML Classifier │     │  DL Classifier │     │ GenAI Classifier│
     │   (XGBoost)    │     │  (BiLSTM)      │     │  (Groq LLM)     │
     └───────┬────────┘     └───────┬────────┘     └───────┬────────┘
             │                      │                      │
     Numerical + TF-IDF      Character-level           Semantic
     URL features            sequence analysis         URL analysis
```

### How classification works

1. A URL is submitted to the orchestrator.
2. Three classifiers run in parallel:
   - **ML (XGBoost)** — extracts 20+ numerical URL features (length, character counts, suspicious keywords, domain structure) and combines them with TF-IDF text features
   - **DL (BiLSTM)** — tokenizes the URL at the character level and passes it through a bidirectional LSTM network
   - **GenAI (Groq)** — sends the URL to a Llama 3.1 model with a structured cybersecurity prompt for semantic analysis
3. Each model returns a prediction (`Phishing` or `Legitimate`) and a confidence score.
4. The orchestrator computes weighted scores and selects the verdict with the higher total.
5. If GenAI fails (missing API key, timeout, or error), the remaining models are re-normalized and used for the final decision.

Default weights: **GenAI 60%** · **DL 20%** · **ML 20%**

## Quick Start

### Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- [Groq API key](https://console.groq.com/keys) (required for GenAI classifier)

### Installation

```bash
git clone https://github.com/<your-username>/url_phishing_detector.git
cd url_phishing_detector

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell

uv pip install -r requirements.txt
```

### Environment setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Verify model files

Ensure these pre-trained models are present in the `models/` directory:

```
models/
├── phishing_xgboost_model.pkl    # XGBoost classifier
├── tfidf_vectorizer.pkl          # TF-IDF vectorizer for ML features
├── lstm_phishing_best.pt         # LSTM model weights
└── lstm_model_metadata.json      # LSTM configuration and training stats
```

## Usage

### Streamlit web interface (recommended)

```bash
uv run streamlit run app.py
```

Opens at `http://localhost:8501` with:

- URL input and one-click analysis
- Per-model prediction cards (ML, DL, GenAI)
- Confidence meter and session history
- Quick-test buttons for demo URLs

### Command line

```bash
uv run python -m phishing_detector
```

Runs the built-in demo against a set of sample URLs and prints per-model results.

### Python API

```python
from phishing_detector import PhishingOrchestrator

orchestrator = PhishingOrchestrator()

# Single URL
result = orchestrator.classify("suspicious-login.example.com")
print(f"Verdict:   {result.final_verdict}")
print(f"Confidence: {result.confidence:.1%}")
print(f"ML:    {result.ml_result.prediction} ({result.ml_result.confidence:.1%})")
print(f"DL:    {result.dl_result.prediction} ({result.dl_result.confidence:.1%})")
print(f"GenAI: {result.genai_result.prediction} ({result.genai_result.confidence:.1%})")

# Batch classification
urls = ["google.com", "phishing-site.xyz", "github.com"]
results = orchestrator.classify_batch(urls)
```

### Custom configuration

```python
orchestrator = PhishingOrchestrator(
    genai_weight=0.6,
    dl_weight=0.2,
    ml_weight=0.2,
    genai_timeout=30.0,
    ml_model_path="path/to/xgboost.pkl",
    dl_model_path="path/to/lstm.pt",
)
```

## Project structure

```
url_phishing_detector/
├── app.py                          # Streamlit web interface
├── requirements.txt                # Python dependencies
├── models/                         # Pre-trained model artifacts
├── eval/                           # Evaluation scripts and reports
│   ├── Dataset.csv                 # Evaluation dataset
│   ├── evaluate_model.py           # Full ensemble evaluation
│   ├── evaluate_xgboost.py         # XGBoost-only evaluation
│   └── evaluate_lstm.py            # LSTM-only evaluation
└── phishing_detector/              # Main Python package
    ├── __init__.py                 # Package exports
    ├── __main__.py                 # CLI entry point
    ├── core/
    │   ├── data_classes.py         # PredictionResult, OrchestratorResult
    │   └── base_classifier.py      # BaseClassifier ABC
    ├── preprocessing/
    │   ├── feature_extractor.py  # URL feature engineering
    │   └── tokenizer.py            # Character-level tokenizer for LSTM
    ├── models/
    │   └── lstm.py                 # PhishingLSTM (bidirectional LSTM)
    ├── classifiers/
    │   ├── ml_classifier.py        # XGBoost classifier
    │   ├── dl_classifier.py        # PyTorch LSTM classifier
    │   └── genai_classifier.py     # Groq LLM classifier
    └── orchestrator/
        └── phishing_orchestrator.py  # Ensemble orchestrator
```

## Models

| Model | Type | Input | Key details |
|-------|------|-------|-------------|
| **XGBoost-ML** | Gradient boosting | 20+ numerical features + TF-IDF | Fast inference (~3 ms/URL); pattern-based detection |
| **BiLSTM-DL** | Bidirectional LSTM | Character-level URL sequences | 2-layer BiLSTM, 64-dim embeddings, 128 hidden units; trained on 250K+ URLs |
| **Groq-GenAI** | Llama 3.1 8B Instant | Raw URL string | Structured output with confidence and reasoning; semantic/contextual analysis |

### LSTM training stats

From `models/lstm_model_metadata.json`:

| Metric | Value |
|--------|-------|
| Accuracy | 97.1% |
| Precision | 98.3% |
| Recall | 95.8% |
| F1 Score | 97.0% |
| ROC AUC | 99.6% |
| Training samples | 250,275 |
| Test samples | 62,569 |

## Evaluation

The `eval/` directory contains scripts to benchmark models against `Dataset.csv` (600-sample evaluation set).

```bash
# Evaluate full ensemble
uv run python eval/evaluate_model.py

# Evaluate individual models
uv run python eval/evaluate_xgboost.py
uv run python eval/evaluate_lstm.py
```

### Ensemble performance (600-sample eval set)

| Metric | Value |
|--------|-------|
| Accuracy | 92.0% |
| Precision | 76.5% |
| Recall | 69.9% |
| F1 Score | 73.0% |
| Avg. processing time | 1.71 s/URL |

The ensemble outperforms individual models on this evaluation set because the GenAI classifier provides strong semantic reasoning that compensates for the ML and DL models' higher false-positive rates on this particular dataset distribution.

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning (LSTM) |
| `xgboost` | Machine learning classifier |
| `scikit-learn` | ML utilities and TF-IDF |
| `langchain-groq` | Groq LLM integration |
| `pydantic` | Structured GenAI output validation |
| `python-dotenv` | Environment variable management |
| `streamlit` | Web interface |
| `pandas`, `numpy`, `scipy` | Data processing |

## Logging

Classification activity is logged to `phishing_orchestrator.log` in the project root, including per-model predictions, weighted scoring details, and timing information.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'phishing_detector'` | Run commands from the project root directory |
| `GROQ_API_KEY not found` | Create a `.env` file with your Groq API key |
| GenAI classifier fails | System falls back to re-weighted ML + DL scores automatically |
| Model files missing | Ensure all four files exist in `models/` (see [Verify model files](#verify-model-files)) |

## License

This project is licensed under the [MIT License](LICENSE).
