"""Vocabulary loader for COCO captions."""
import pickle
import sys
import types
from pathlib import Path
from typing import Dict


def _load_legacy_vocab_object(vocab_file: Path):
    """Unpickle a vocab.pkl saved by the original Udacity `vocabulary.Vocabulary` class.

    The original class lived in a top-level module named `vocabulary`, so the
    pickle stream references `vocabulary.Vocabulary` by that module path. That
    module doesn't exist in this package, so we register a throwaway shim
    module under that name just long enough to unpickle, then discard it.
    """
    class _LegacyVocabulary:
        pass

    shim = types.ModuleType('vocabulary')
    shim.Vocabulary = _LegacyVocabulary
    already_present = 'vocabulary' in sys.modules
    previous = sys.modules.get('vocabulary')
    sys.modules['vocabulary'] = shim
    try:
        with open(vocab_file, 'rb') as f:
            return pickle.load(f)
    finally:
        if already_present:
            sys.modules['vocabulary'] = previous
        else:
            del sys.modules['vocabulary']


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

        vocab_obj = _load_legacy_vocab_object(vocab_file)

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
