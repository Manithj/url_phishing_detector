"""
Base Classifier
================
Abstract base class defining the interface for all classifiers.
"""

from abc import ABC, abstractmethod
from .data_classes import PredictionResult


class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""

    @abstractmethod
    def predict(self, input_data: str) -> PredictionResult:
        """Predict phishing status for input data."""
        pass
