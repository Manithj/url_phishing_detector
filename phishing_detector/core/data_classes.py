"""
Data Classes
============
Core data structures for phishing detection results.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class PredictionResult:
    """Result from a single model prediction."""
    model_name: str
    prediction: Literal["Phishing", "Legitimate"]
    confidence: float
    inference_time: float
    error: Optional[str] = None


@dataclass
class OrchestratorResult:
    """Final result from the orchestrator."""
    final_verdict: Literal["Phishing", "Legitimate"]
    confidence: float
    ml_result: Optional[PredictionResult]
    dl_result: Optional[PredictionResult]
    genai_result: Optional[PredictionResult]
    voting_details: Dict
    total_time: float
    failover_used: bool = False
