"""Tests for vocabulary operations."""
import pytest
import pickle
import tempfile
from pathlib import Path
from image_captioning.vocabulary import Vocabulary


def create_test_vocab():
    """Helper: create a minimal test vocabulary."""
    class MockVocab:
        def __init__(self):
            self.word2idx = {
                '<unk>': 0,
                '<start>': 1,
                '<end>': 2,
                'a': 3,
                'cat': 4,
                'dog': 5,
            }
            self.idx2word = {v: k for k, v in self.word2idx.items()}

    return MockVocab()


def test_vocabulary_load_from_file():
    """Test loading vocabulary from pickle file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.pkl"
        mock_vocab = create_test_vocab()

        with open(vocab_path, 'wb') as f:
            pickle.dump(mock_vocab, f)

        vocab = Vocabulary(str(vocab_path))

        assert len(vocab) == 6
        assert '<start>' in vocab.word2idx


def test_vocabulary_word_to_idx():
    """Test word-to-index lookup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.pkl"
        mock_vocab = create_test_vocab()

        with open(vocab_path, 'wb') as f:
            pickle.dump(mock_vocab, f)

        vocab = Vocabulary(str(vocab_path))

        assert vocab('cat') == 4
        assert vocab('dog') == 5
        assert vocab('unknown_word') == 0  # <unk>


def test_vocabulary_decode():
    """Test decoding token indices to words."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.pkl"
        mock_vocab = create_test_vocab()

        with open(vocab_path, 'wb') as f:
            pickle.dump(mock_vocab, f)

        vocab = Vocabulary(str(vocab_path))

        tokens = [3, 4, 2]  # [a, cat, <end>]
        decoded = vocab.decode(tokens)

        assert 'a' in decoded
        assert 'cat' in decoded
        assert '<end>' not in decoded  # Special tokens are stripped


def test_vocabulary_missing_file():
    """Test that loading from missing file raises error."""
    with pytest.raises(FileNotFoundError):
        Vocabulary('/nonexistent/path/vocab.pkl')


def test_vocabulary_len():
    """Test vocabulary size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_path = Path(tmpdir) / "vocab.pkl"
        mock_vocab = create_test_vocab()

        with open(vocab_path, 'wb') as f:
            pickle.dump(mock_vocab, f)

        vocab = Vocabulary(str(vocab_path))

        assert len(vocab) == len(mock_vocab.word2idx)
