"""Vocabulary loader for COCO captions."""
import pickle
from pathlib import Path
from typing import Dict


class Vocabulary:
    """Simple wrapper for pre-built vocabulary from vocab.pkl."""

    def __init__(self, vocab_file: str) -> None:
        """Load vocabulary from pickle file.

        Args:
            vocab_file: Path to vocab.pkl created during training.
        """
        vocab_file = Path(vocab_file)
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

        with open(vocab_file, 'rb') as f:
            vocab_obj = pickle.load(f)

        self.word2idx: Dict[str, int] = vocab_obj.word2idx
        self.idx2word: Dict[int, str] = vocab_obj.idx2word

    def __call__(self, word: str) -> int:
        """Get token index for a word (returns <unk> if unknown)."""
        return self.word2idx.get(word, self.word2idx.get('<unk>', 0))

    def __len__(self) -> int:
        """Vocabulary size."""
        return len(self.word2idx)

    def decode(self, tokens: list) -> str:
        """Convert token indices back to words.

        Args:
            tokens: List of token indices.

        Returns:
            Sentence string with special tokens stripped.
        """
        words = []
        for token_id in tokens:
            word = self.idx2word.get(token_id, '<unk>')
            if word not in ['<start>', '<end>', '<unk>']:
                words.append(word)
        return ' '.join(words)
