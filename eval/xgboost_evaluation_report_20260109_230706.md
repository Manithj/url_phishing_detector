# XGBoost Model Evaluation Report

**Model:** XGBoost (XGBoost-ML)


**Sample Size:** 600

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 38.33% |
| Precision | 15.97% |
| Recall | 69.89% |
| F1 Score | 26.00% |

## Confusion Matrix

| | Predicted Legitimate | Predicted Phishing |
|---|---|---|
| **Actual Legitimate** | 165 (TN) | 342 (FP) |
| **Actual Phishing** | 28 (FN) | 65 (TP) |

## Performance Statistics

- Average Inference Time: 0.0027s per URL
- Total Evaluation Time: 2.2s
