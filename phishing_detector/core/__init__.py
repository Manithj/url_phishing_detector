"""
Core Module
===========
Core data structures and base interfaces for the phishing detector.
"""

from .data_classes import PredictionResult, OrchestratorResult
from .base_classifier import BaseClassifier

__all__ = [
    "PredictionResult",
    "OrchestratorResult",
    "BaseClassifier",
]
