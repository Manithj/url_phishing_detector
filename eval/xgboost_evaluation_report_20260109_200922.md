# XGBoost Model Evaluation Report

**Model:** XGBoost (XGBoost-ML)

**Date:** 2026-01-09 20:09:22

**Sample Size:** 300

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 41.67% |
| Precision | 17.44% |
| Recall | 70.83% |
| F1 Score | 27.98% |

## Confusion Matrix

| | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 91 (TN) | 161 (FP) |
| **Actual Phishing** | 14 (FN) | 34 (TP) |

## Performance Statistics

- Average Inference Time: 0.0060s per URL
- Total Evaluation Time: 2.3s
