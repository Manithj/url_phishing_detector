"""
Phishing Detector Package
=========================
A modular ensemble system for phishing URL detection combining ML, DL, and GenAI models.
"""

import logging
import sys
from pathlib import Path

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("phishing_orchestrator.log", encoding="utf-8"),
    ],
)

# Package root path
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Main exports
from .core import PredictionResult, OrchestratorResult, BaseClassifier
from .orchestrator import PhishingOrchestrator

__all__ = [
    "PhishingOrchestrator",
    "PredictionResult",
    "OrchestratorResult",
    "BaseClassifier",
    "PACKAGE_ROOT",
    "PROJECT_ROOT",
]

__version__ = "1.0.0"
