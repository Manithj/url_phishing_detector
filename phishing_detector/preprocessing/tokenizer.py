"""
Character Tokenizer
===================
Character-level tokenizer for URLs (used by LSTM model).
"""

from typing import List


class CharTokenizer:
    """Character-level tokenizer for URLs (used by LSTM model)."""

    def __init__(self):
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_char = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def fit(self, urls: List[str]) -> 'CharTokenizer':
        """Build vocabulary from list of URLs."""
        chars = set()
        for url in urls:
            chars.update(set(url.lower()))

        for char in sorted(chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = self.vocab_size
                self.idx_to_char[self.vocab_size] = char
                self.vocab_size += 1

        return self

    def load_vocab(self, vocab_size: int):
        """Load vocabulary from metadata (builds standard vocab)."""
        # Build a standard character vocabulary
        chars = (
            list(range(256))  # All ASCII characters
        )
        for i, char_code in enumerate(chars):
            char = chr(char_code)
            if char not in self.char_to_idx:
                self.char_to_idx[char] = self.vocab_size
                self.idx_to_char[self.vocab_size] = char
                self.vocab_size += 1
                if self.vocab_size >= vocab_size:
                    break

    def encode(self, url: str, max_len: int = 133) -> List[int]:
        """Convert URL to list of character indices with padding/truncation."""
        url = url.lower()
        indices = [self.char_to_idx.get(c, 1) for c in url]  # 1 = <UNK>

        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [0] * (max_len - len(indices))  # 0 = <PAD>

        return indices
