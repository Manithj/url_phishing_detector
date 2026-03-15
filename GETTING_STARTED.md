# Getting Started with Phishing Detector

A comprehensive guide to set up and run the Phishing Detector ensemble system using **uv** for dependency management.

## Prerequisites

- **Python 3.8+** installed on your system
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **Groq API Key** for GenAI classifier ([get one here](https://console.groq.com/keys))

---

## Quick Start

### 1. Install uv

If you don't have `uv` installed, run:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installation.

---

### 2. Clone/Navigate to the Project

```bash
cd path/to/final_build
```

---

### 3. Initialize the Project with uv

Create a virtual environment and install dependencies:

```bash
# Create a new virtual environment
uv venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Windows (Command Prompt):
.venv\Scripts\activate.bat

# macOS/Linux:
source .venv/bin/activate
```

---

### 4. Install Dependencies

```bash
uv pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `torch` | Deep Learning (LSTM model) |
| `xgboost` | Machine Learning classifier |
| `scikit-learn` | ML utilities & preprocessing |
| `langchain-groq` | Groq LLM integration |
| `pydantic` | Data validation |
| `python-dotenv` | Environment variable management |
| `pandas`, `numpy`, `scipy` | Data processing |

---

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

> [!IMPORTANT]
> Get your free API key from [Groq Console](https://console.groq.com/keys)

---

### 6. Verify Model Files

Ensure these files exist in the `models/` directory:

```
models/
├── phishing_xgboost_model.pkl    # XGBoost model
├── tfidf_vectorizer.pkl          # TF-IDF vectorizer
├── lstm_phishing_best.pt         # LSTM weights
└── lstm_model_metadata.json      # LSTM config
```

---

## Running the Detector

### Option 1: Streamlit Web Interface (Recommended)

```bash
# Install streamlit if not already installed
uv pip install streamlit

# Run the web app
streamlit run app.py
```

This opens a beautiful web interface at `http://localhost:8501` with:
- 🛡️ Modern dark theme with gradient styling
- 📊 Real-time confidence meters
- 🎯 Individual model predictions (ML, DL, GenAI)
- 📜 Session history tracking
- ⚡ Quick-test buttons for demo URLs

### Option 2: Run as a Module (CLI)

```bash
uv run python -m phishing_detector
```

### Option 3: Direct Execution

```bash
cd final_build
python -m phishing_detector
```

### Expected Output

```
============================================================
PHISHING DETECTION ORCHESTRATOR
============================================================

------------------------------------------------------------
CLASSIFICATION RESULTS
------------------------------------------------------------

URL: www.google.com
  Final Verdict: legitimate
  Confidence: 95.2%
  ML: legitimate (92.1%)
  DL: legitimate (97.3%)
  GenAI: legitimate (96.2%)

URL: paypal.com.login-verify.suspicious-site.xyz/secure/update
  Final Verdict: phishing
  Confidence: 98.7%
  ML: phishing (99.1%)
  DL: phishing (97.8%)
  GenAI: phishing (99.3%)
```

---

## Python API Usage

```python
from phishing_detector import PhishingOrchestrator

# Initialize (loads all models)
orchestrator = PhishingOrchestrator()

# Single URL classification
result = orchestrator.classify("suspicious-login.example.com")
print(f"Verdict: {result.final_verdict}")
print(f"Confidence: {result.confidence:.1%}")

# Batch classification
urls = ["google.com", "phishing-site.xyz", "github.com"]
results = orchestrator.classify_batch(urls)
for url, res in zip(urls, results):
    print(f"{url}: {res.final_verdict}")
```

---

## Project Structure

```
final_build/
├── .env                    # API keys (create this)
├── requirements.txt        # Python dependencies
├── models/                 # Pre-trained model files
│   ├── phishing_xgboost_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── lstm_phishing_best.pt
│   └── lstm_model_metadata.json
└── phishing_detector/      # Main package
    ├── __init__.py
    ├── __main__.py         # Entry point
    ├── core/               # Data classes & base classifier
    ├── preprocessing/      # Feature extraction & tokenization
    ├── models/             # LSTM neural network
    ├── classifiers/        # ML, DL, GenAI classifiers
    └── orchestrator/       # Main orchestrator (majority voting)
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'phishing_detector'`
Make sure you're running from the `final_build` directory:
```bash
cd final_build
uv run python -m phishing_detector
```

### `GROQ_API_KEY not found`
Ensure `.env` file exists in `final_build/` with your API key.

### GenAI Classifier Falls Back to Weighted Decision
If Groq API is unavailable, the system automatically uses:
- **60% weight** → ML (XGBoost)
- **40% weight** → DL (LSTM)

---

## How It Works

1. **Parallel Execution**: ML, DL, and GenAI classifiers run simultaneously
2. **Majority Voting**: Final verdict based on 2+ agreeing models
3. **Failover**: GenAI failure triggers weighted ML+DL decision
4. **Logging**: Audit trail saved to `phishing_orchestrator.log`

---

## Additional Resources

- [uv Documentation](https://docs.astral.sh/uv/)
- [Groq API Docs](https://console.groq.com/docs)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PyTorch LSTM Guide](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
