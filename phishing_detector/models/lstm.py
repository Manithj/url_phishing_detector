"""
Phishing LSTM Model
===================
Bidirectional LSTM for phishing URL classification.
"""

import torch
import torch.nn as nn


class PhishingLSTM(nn.Module):
    """Bidirectional LSTM for phishing URL classification."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super(PhishingLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Trainable character embeddings
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_fwd, hidden_bwd), dim=1)

        out = self.dropout(hidden_concat)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze(1)
