"""
ML Classifier
=============
XGBoost-based ML classifier for phishing detection.
"""

import logging
import pickle
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack

from ..core import BaseClassifier, PredictionResult
from ..preprocessing import FeatureExtractor

logger = logging.getLogger("PhishingOrchestrator")


class MLClassifier(BaseClassifier):
    """XGBoost-based ML classifier for phishing detection."""

    def __init__(self, model_path: str, vectorizer_path: str):
        logger.info("Initializing ML Classifier (XGBoost)...")
        self.feature_extractor = FeatureExtractor()

        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load TF-IDF vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

        logger.info("ML Classifier initialized successfully")

    def predict(self, input_data: str) -> PredictionResult:
        """Run XGBoost prediction on input URL."""
        start_time = time.time()

        try:
            # Extract numerical features
            features = self.feature_extractor.extract_url_features(input_data)
            feature_values = np.array(list(features.values())).reshape(1, -1).astype(np.float32)
            X_numerical = csr_matrix(feature_values)

            # Extract TF-IDF features
            tfidf_features = self.tfidf_vectorizer.transform([input_data])

            # Combine features
            X = hstack([X_numerical, tfidf_features])

            # Predict
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]

            # Label mapping: 0 = bad (phishing), 1 = good (legitimate)
            label = "Legitimate" if prediction == 1 else "Phishing"
            confidence = float(max(probability))

            inference_time = time.time() - start_time

            logger.info(f"ML prediction: {label} (confidence: {confidence:.2%})")

            return PredictionResult(
                model_name="XGBoost-ML",
                prediction=label,
                confidence=confidence,
                inference_time=inference_time
            )

        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"ML prediction error: {e}")
            return PredictionResult(
                model_name="XGBoost-ML",
                prediction="Legitimate",
                confidence=0.5,
                inference_time=inference_time,
                error=str(e)
            )
