# LSTM Model Evaluation Report

**Model:** LSTM (BiLSTM-DL)

**Date:** 2026-01-09 20:07:10

**Sample Size:** 300

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 23.00% |
| Precision | 12.96% |
| Recall | 66.67% |
| F1 Score | 21.69% |

## Confusion Matrix

| | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 37 (TN) | 215 (FP) |
| **Actual Phishing** | 16 (FN) | 32 (TP) |

## Performance Statistics

- Average Inference Time: 0.0052s per URL
- Total Evaluation Time: 1.9s
