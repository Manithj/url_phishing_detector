# LSTM Model Evaluation Report

**Model:** LSTM (BiLSTM-DL)

**Sample Size:** 600

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 25.00% |
| Precision | 13.20% |
| Recall | 68.82% |
| F1 Score | 22.15% |

## Confusion Matrix

| | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 86 (TN) | 421 (FP) |
| **Actual Phishing** | 29 (FN) | 64 (TP) |

## Performance Statistics

- Average Inference Time: 0.0041s per URL
- Total Evaluation Time: 3.1s
