"""
DL Classifier
=============
PyTorch LSTM-based DL classifier for phishing detection.
"""

import json
import logging
import time

import torch

from ..core import BaseClassifier, PredictionResult
from ..preprocessing import CharTokenizer
from ..models import PhishingLSTM

logger = logging.getLogger("PhishingOrchestrator")


class DLClassifier(BaseClassifier):
    """PyTorch LSTM-based DL classifier for phishing detection."""

    def __init__(self, model_path: str, metadata_path: str):
        logger.info("Initializing DL Classifier (LSTM)...")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Initialize tokenizer
        self.tokenizer = CharTokenizer()
        self.tokenizer.load_vocab(self.metadata['vocab_size'])
        self.max_len = self.metadata['max_sequence_length']

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint to detect actual dimensions
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Detect actual dimensions from checkpoint
        # embedding.weight shape: [vocab_size, embed_dim]
        actual_embed_dim = checkpoint['embedding.weight'].shape[1]
        # lstm.weight_ih_l0 shape: [4*hidden_dim, embed_dim]
        actual_hidden_dim = checkpoint['lstm.weight_ih_l0'].shape[0] // 4
        
        logger.info(f"Detected model dimensions: embed_dim={actual_embed_dim}, hidden_dim={actual_hidden_dim}")

        # Initialize model with actual dimensions from checkpoint
        self.model = PhishingLSTM(
            vocab_size=self.metadata['vocab_size'],
            embed_dim=actual_embed_dim,
            hidden_dim=actual_hidden_dim,
            num_layers=self.metadata['num_layers'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        logger.info(f"DL Classifier initialized on {self.device}")

    def predict(self, input_data: str) -> PredictionResult:
        """Run LSTM prediction on input URL."""
        start_time = time.time()

        try:
            # Tokenize
            encoded = self.tokenizer.encode(input_data, self.max_len)
            input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)

            # Predict
            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).item()

            # Label mapping: probability > 0.5 = bad (phishing)
            label = "Phishing" if probability > 0.5 else "Legitimate"
            confidence = probability if probability > 0.5 else 1 - probability

            inference_time = time.time() - start_time

            logger.info(f"DL prediction: {label} (confidence: {confidence:.2%})")

            return PredictionResult(
                model_name="BiLSTM-DL",
                prediction=label,
                confidence=confidence,
                inference_time=inference_time
            )

        except Exception as e:
            inference_time = time.time() - start_time
            logger.error(f"DL prediction error: {e}")
            return PredictionResult(
                model_name="BiLSTM-DL",
                prediction="Legitimate",
                confidence=0.5,
                inference_time=inference_time,
                error=str(e)
            )
