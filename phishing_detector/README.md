# Phishing Detector

A modular ensemble system for phishing URL detection combining ML, DL, and GenAI models with majority voting.

## Package Structure

```
phishing_detector/
├── __init__.py          # Package exports & logging config
├── __main__.py          # Entry point (python -m phishing_detector)
├── core/
│   ├── __init__.py
│   ├── data_classes.py  # PredictionResult, OrchestratorResult
│   └── base_classifier.py  # BaseClassifier ABC
├── preprocessing/
│   ├── __init__.py
│   ├── feature_extractor.py  # URL feature extraction
│   └── tokenizer.py     # CharTokenizer for LSTM
├── models/
│   ├── __init__.py
│   └── lstm.py          # PhishingLSTM nn.Module
├── classifiers/
│   ├── __init__.py
│   ├── ml_classifier.py   # XGBoost classifier
│   ├── dl_classifier.py   # LSTM classifier
│   └── genai_classifier.py  # Groq LLM classifier
└── orchestrator/
    ├── __init__.py
    └── phishing_orchestrator.py  # Main orchestrator
```

## Components

| Module | Description |
|--------|-------------|
| **core** | Data classes (`PredictionResult`, `OrchestratorResult`) and `BaseClassifier` interface |
| **preprocessing** | `FeatureExtractor` for URL features, `CharTokenizer` for LSTM input |
| **models** | `PhishingLSTM` bidirectional LSTM neural network |
| **classifiers** | `MLClassifier` (XGBoost), `DLClassifier` (LSTM), `GenAIClassifier` (Groq LLM) |
| **orchestrator** | `PhishingOrchestrator` - parallel execution with majority voting |

## Usage

### Python API

```python
from phishing_detector import PhishingOrchestrator

# Initialize orchestrator (loads all models)
orchestrator = PhishingOrchestrator()

# Classify a URL
result = orchestrator.classify("suspicious-url.example.com")

print(f"Verdict: {result.final_verdict}")
print(f"Confidence: {result.confidence:.1%}")
print(f"ML: {result.ml_result.prediction}")
print(f"DL: {result.dl_result.prediction}")
print(f"GenAI: {result.genai_result.prediction}")
```

### Command Line

```bash
python -m phishing_detector
```

### Batch Classification

```python
urls = ["google.com", "suspicious-login.xyz", "github.com/repo"]
results = orchestrator.classify_batch(urls)

for url, result in zip(urls, results):
    print(f"{url}: {result.final_verdict}")
```

## How It Works

1. **Parallel Execution**: All three classifiers run simultaneously using `ThreadPoolExecutor`
2. **Majority Voting**: Final verdict based on 2+ agreeing models
3. **Failover**: If GenAI fails, uses weighted ML (60%) + DL (40%) decision
4. **Logging**: Comprehensive audit trail in `phishing_orchestrator.log`

## Requirements

- Python 3.8+
- PyTorch
- XGBoost
- LangChain + Groq (for GenAI)
- See `requirements.txt` for full list

## Model Files

Place these in the `models/` directory:
- `phishing_xgboost_model.pkl` - XGBoost model
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `lstm_phishing_best.pt` - LSTM model weights
- `lstm_model_metadata.json` - LSTM configuration

## Environment Variables

Create a `.env` file with:
```
GROQ_API_KEY=your_api_key_here
```
