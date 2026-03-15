"""
GenAI Classifier
================
Groq LLM-based GenAI classifier for phishing detection.
"""

import logging
import os
import time
from typing import Literal

from dotenv import load_dotenv

from ..core import BaseClassifier, PredictionResult

logger = logging.getLogger("PhishingOrchestrator")


class GenAIClassifier(BaseClassifier):
    """Groq LLM-based GenAI classifier for phishing detection."""

    def __init__(self, model_name: str = "llama-3.1-8b-instant", timeout: float = 30.0):
        logger.info("Initializing GenAI Classifier (Groq)...")
        
        # Try to load .env from multiple locations
        load_dotenv()  # Current directory
        load_dotenv("llm/.env")  # llm subdirectory

        self.timeout = timeout
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            logger.warning("GROQ_API_KEY not found - GenAI classifier will fail")
            self.llm = None
            self.chain = None
            return

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_groq import ChatGroq
            from pydantic import BaseModel, Field

            class URLClassification(BaseModel):
                """Structured output for URL classification."""
                classification: Literal["good", "bad"] = Field(
                    description="Classification: 'good' for legitimate, 'bad' for phishing"
                )
                confidence: float = Field(
                    description="Confidence score between 0.0 and 1.0",
                    ge=0.0, le=1.0
                )
                reason: str = Field(
                    description="Brief explanation for the classification"
                )

            self.llm = ChatGroq(
                api_key=self.api_key,
                model_name=model_name,
                temperature=0.0,
            )

            self.structured_llm = self.llm.with_structured_output(URLClassification)

            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a cybersecurity expert specialized in identifying phishing URLs.

Analyze the given URL and classify it as either:
- 'good': Legitimate, safe URL from reputable sources
- 'bad': Phishing, malicious, or suspicious URL

Consider these factors:
1. Domain reputation and legitimacy
2. URL structure (excessive subdomains, random characters)
3. Suspicious patterns (misspellings, deceptive paths)
4. Security indicators (HTTP vs HTTPS)
5. Known phishing patterns (fake login pages, urgent messages)

Provide a confidence score and brief reasoning."""),
                ("human", "Classify this URL: {url}")
            ])

            self.chain = self.prompt | self.structured_llm
            logger.info(f"GenAI Classifier initialized with model: {model_name}")

        except ImportError as e:
            logger.warning(f"GenAI dependencies not available: {e}")
            self.llm = None
            self.chain = None

    def predict(self, input_data: str) -> PredictionResult:
        """Run GenAI prediction on input URL."""
        start_time = time.time()

        if self.chain is None:
            return PredictionResult(
                model_name="Groq-GenAI",
                prediction="Legitimate",
                confidence=0.5,
                inference_time=time.time() - start_time,
                error="GenAI not configured"
            )

        try:
            result = self.chain.invoke({"url": input_data})

            label = "Phishing" if result.classification == "bad" else "Legitimate"
            confidence = result.confidence

            inference_time = time.time() - start_time

            logger.info(f"GenAI prediction: {label} (confidence: {confidence:.2%}, reason: {result.reason[:50]}...)")

            return PredictionResult(
                model_name="Groq-GenAI",
                prediction=label,
                confidence=confidence,
                inference_time=inference_time
            )

        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"GenAI prediction error: {e}")
            return PredictionResult(
                model_name="Groq-GenAI",
                prediction="Legitimate",
                confidence=0.5,
                inference_time=inference_time,
                error=str(e)
            )
