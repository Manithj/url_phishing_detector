"""
Phishing Orchestrator
=====================
Main orchestrator that runs ML, DL, and GenAI models in parallel
and combines results using weighted scoring.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np

from ..core import BaseClassifier, PredictionResult, OrchestratorResult
from ..classifiers import MLClassifier, DLClassifier, GenAIClassifier

logger = logging.getLogger("PhishingOrchestrator")

# Get the project root (parent of phishing_detector package)
PACKAGE_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent


class PhishingOrchestrator:
    """
    Main orchestrator that runs ML, DL, and GenAI models in parallel
    and combines results using weighted scoring.
    """

    def __init__(
        self,
        ml_model_path: Optional[str] = None,
        ml_vectorizer_path: Optional[str] = None,
        dl_model_path: Optional[str] = None,
        dl_metadata_path: Optional[str] = None,
        genai_timeout: float = 30.0,
        genai_weight: float = 0.6,
        dl_weight: float = 0.2,
        ml_weight: float = 0.2
    ):
        """
        Initialize the orchestrator with all three classifiers.

        Args:
            ml_model_path: Path to XGBoost model pickle
            ml_vectorizer_path: Path to TF-IDF vectorizer pickle
            dl_model_path: Path to LSTM PyTorch model
            dl_metadata_path: Path to LSTM metadata JSON
            genai_timeout: Timeout for GenAI API calls
            genai_weight: Weight for GenAI model (default: 0.6)
            dl_weight: Weight for DL model (default: 0.2)
            ml_weight: Weight for ML model (default: 0.2)
        """
        logger.info("=" * 60)
        logger.info("INITIALIZING PHISHING ORCHESTRATOR")
        logger.info("=" * 60)

        # Set default paths relative to project root
        models_dir = PROJECT_ROOT / "models"
        
        if ml_model_path is None:
            ml_model_path = str(models_dir / "phishing_xgboost_model.pkl")
        if ml_vectorizer_path is None:
            ml_vectorizer_path = str(models_dir / "tfidf_vectorizer.pkl")
        if dl_model_path is None:
            dl_model_path = str(models_dir / "lstm_phishing_best.pt")
        if dl_metadata_path is None:
            dl_metadata_path = str(models_dir / "lstm_model_metadata.json")

        self.genai_weight = genai_weight
        self.dl_weight = dl_weight
        self.ml_weight = ml_weight
        self.genai_timeout = genai_timeout

        # Initialize classifiers
        try:
            self.ml_classifier = MLClassifier(ml_model_path, ml_vectorizer_path)
        except Exception as e:
            logger.error(f"Failed to load ML classifier: {e}")
            self.ml_classifier = None

        try:
            self.dl_classifier = DLClassifier(dl_model_path, dl_metadata_path)
        except Exception as e:
            logger.error(f"Failed to load DL classifier: {e}")
            self.dl_classifier = None

        try:
            self.genai_classifier = GenAIClassifier(timeout=genai_timeout)
        except Exception as e:
            logger.error(f"Failed to load GenAI classifier: {e}")
            self.genai_classifier = None

        logger.info(f"Weights: GenAI={genai_weight}, DL={dl_weight}, ML={ml_weight}")
        logger.info("Orchestrator initialization complete")

    def _run_classifier(self, classifier: BaseClassifier, input_data: str) -> Optional[PredictionResult]:
        """Run a single classifier with error handling."""
        if classifier is None:
            return None
        try:
            return classifier.predict(input_data)
        except Exception as e:
            logger.error(f"Classifier error: {e}")
            return None

    def _weighted_decision(
        self,
        ml_result: Optional[PredictionResult],
        dl_result: Optional[PredictionResult],
        genai_result: Optional[PredictionResult]
    ) -> Tuple[str, float, Dict, bool]:
        """
        Apply weighted scoring to determine final verdict.
        
        Each model contributes: weight × confidence × (1 for phishing, 0 for legitimate)
        
        Returns: (verdict, confidence, scoring_details, failover_used)
        """
        # Collect valid results and their weights
        results_with_weights = []
        
        if ml_result is not None and ml_result.error is None:
            results_with_weights.append(('ML', ml_result, self.ml_weight))
        if dl_result is not None and dl_result.error is None:
            results_with_weights.append(('DL', dl_result, self.dl_weight))
        if genai_result is not None and genai_result.error is None:
            results_with_weights.append(('GenAI', genai_result, self.genai_weight))

        # Check if GenAI failed (COMMENTED OUT - failsafe disabled)
        genai_failed = genai_result is None or genai_result.error is not None
        failover_used = genai_failed and len(results_with_weights) >= 2

        if not results_with_weights:
            # No valid results - default to Legitimate with low confidence
            return "Legitimate", 0.5, {
                "phishing_score": 0.0,
                "legitimate_score": 0.0,
                "total_valid_models": 0,
                "genai_failed": genai_failed,
                "model_contributions": {}
            }, False

        # Calculate total weight of available models for normalization
        total_weight = sum(w for _, _, w in results_with_weights)

        # Compute weighted scores
        phishing_score = 0.0
        legitimate_score = 0.0
        model_contributions = {}

        for name, result, weight in results_with_weights:
            # Normalize weight based on available models
            normalized_weight = weight / total_weight
            
            if result.prediction == "Phishing":
                contribution = normalized_weight * result.confidence
                phishing_score += contribution
                model_contributions[name] = {
                    "prediction": "Phishing",
                    "confidence": result.confidence,
                    "weight": normalized_weight,
                    "contribution_to_phishing": contribution
                }
            else:
                contribution = normalized_weight * result.confidence
                legitimate_score += contribution
                model_contributions[name] = {
                    "prediction": "Legitimate",
                    "confidence": result.confidence,
                    "weight": normalized_weight,
                    "contribution_to_legitimate": contribution
                }

        # Determine verdict based on weighted scores
        if phishing_score > legitimate_score:
            verdict = "Phishing"
            confidence = phishing_score / (phishing_score + legitimate_score) if (phishing_score + legitimate_score) > 0 else 0.5
        else:
            verdict = "Legitimate"
            confidence = legitimate_score / (phishing_score + legitimate_score) if (phishing_score + legitimate_score) > 0 else 0.5

        scoring_details = {
            "phishing_score": round(phishing_score, 4),
            "legitimate_score": round(legitimate_score, 4),
            "total_valid_models": len(results_with_weights),
            "genai_failed": genai_failed,
            "model_contributions": model_contributions
        }

        # Failover warning (COMMENTED OUT - failsafe disabled)
        if failover_used:
            logger.warning("GenAI failed - using weighted ML/DL failover with normalized weights")

        return verdict, confidence, scoring_details, failover_used

    def classify(self, input_data: str) -> OrchestratorResult:
        """
        Classify input URL/Email using all three models in parallel.

        Args:
            input_data: URL or email content to classify

        Returns:
            OrchestratorResult with final verdict and all details
        """
        start_time = time.time()
        logger.info("-" * 60)
        logger.info(f"Classifying: {input_data[:80]}...")

        ml_result = None
        dl_result = None
        genai_result = None

        # Run classifiers in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}

            if self.ml_classifier:
                futures['ml'] = executor.submit(self._run_classifier, self.ml_classifier, input_data)
            if self.dl_classifier:
                futures['dl'] = executor.submit(self._run_classifier, self.dl_classifier, input_data)
            if self.genai_classifier:
                futures['genai'] = executor.submit(self._run_classifier, self.genai_classifier, input_data)

            # Collect results with timeout handling
            for name, future in futures.items():
                try:
                    timeout = self.genai_timeout if name == 'genai' else 60
                    result = future.result(timeout=timeout)
                    if name == 'ml':
                        ml_result = result
                    elif name == 'dl':
                        dl_result = result
                    elif name == 'genai':
                        genai_result = result
                except FuturesTimeoutError:
                    logger.error(f"{name.upper()} classifier timed out")
                    if name == 'genai':
                        genai_result = PredictionResult(
                            model_name="Groq-GenAI",
                            prediction="Legitimate",
                            confidence=0.5,
                            inference_time=self.genai_timeout,
                            error="Timeout"
                        )
                except Exception as e:
                    logger.error(f"{name.upper()} classifier error: {e}")

        # Apply weighted scoring
        verdict, confidence, scoring_details, failover_used = self._weighted_decision(
            ml_result, dl_result, genai_result
        )

        total_time = time.time() - start_time

        # Log final result
        logger.info(f"FINAL VERDICT: {verdict} (confidence: {confidence:.2%})")
        logger.info(f"Scoring: {scoring_details}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("-" * 60)

        return OrchestratorResult(
            final_verdict=verdict,
            confidence=confidence,
            ml_result=ml_result,
            dl_result=dl_result,
            genai_result=genai_result,
            voting_details=scoring_details,
            total_time=total_time,
            failover_used=failover_used
        )

    def classify_batch(self, inputs: List[str]) -> List[OrchestratorResult]:
        """Classify multiple inputs."""
        results = []
        for input_data in inputs:
            result = self.classify(input_data)
            results.append(result)
        return results
