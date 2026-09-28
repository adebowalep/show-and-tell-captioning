"""Tests for BLEU score computation."""
import pytest
from image_captioning.metrics import bleu_score, _get_ngrams


def test_get_ngrams():
    """Test n-gram extraction."""
    tokens = ['a', 'cat', 'sat', 'on', 'mat']

    bigrams = _get_ngrams(tokens, 2)
    assert ('a', 'cat') in bigrams
    assert ('cat', 'sat') in bigrams
    assert bigrams[('a', 'cat')] == 1

    trigrams = _get_ngrams(tokens, 3)
    assert ('a', 'cat', 'sat') in trigrams


def test_bleu_exact_match():
    """Test BLEU score for exact match (should be 1.0)."""
    ref = ['a', 'cat', 'sat']
    hyp = ['a', 'cat', 'sat']

    score = bleu_score(ref, hyp)
    assert score == pytest.approx(1.0, abs=0.01)


def test_bleu_partial_match():
    """Test BLEU score for partial overlap."""
    ref = ['a', 'cat', 'sat', 'on', 'mat']
    hyp = ['a', 'cat', 'sat']

    score = bleu_score(ref, hyp)
    assert 0.0 < score < 1.0


def test_bleu_no_match():
    """Test BLEU score for no overlap."""
    ref = ['a', 'cat']
    hyp = ['dog', 'runs']

    score = bleu_score(ref, hyp)
    assert score == pytest.approx(0.0, abs=0.01)


def test_bleu_empty_hypothesis():
    """Test BLEU score for empty hypothesis."""
    ref = ['a', 'cat']
    hyp = []

    score = bleu_score(ref, hyp)
    assert score == 0.0


def test_bleu_brevity_penalty():
    """Test that brevity penalty is applied for short hypotheses."""
    ref = ['a', 'cat', 'sat', 'on', 'mat']
    hyp_short = ['a', 'cat']
    hyp_long = ['a', 'cat', 'sat', 'on', 'mat', 'extra']

    score_short = bleu_score(ref, hyp_short)
    score_long = bleu_score(ref, hyp_long)

    # Short hypothesis should have lower score due to brevity penalty
    assert score_short < score_long


def test_bleu_order_matters():
    """Test that word order matters."""
    ref = ['a', 'cat', 'sat']
    hyp_correct = ['a', 'cat', 'sat']
    hyp_wrong_order = ['sat', 'cat', 'a']

    score_correct = bleu_score(ref, hyp_correct)
    score_wrong = bleu_score(ref, hyp_wrong_order)

    assert score_correct > score_wrong


def test_bleu_repeated_words():
    """Test BLEU score with repeated words."""
    ref = ['a', 'cat', 'sat']
    hyp = ['a', 'a', 'a']  # All same word

    score = bleu_score(ref, hyp)
    assert score < bleu_score(ref, ['a', 'cat', 'sat'])


def test_bleu_different_n_values():
    """Test BLEU score with different maximum n-gram sizes."""
    ref = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    hyp = ['the', 'cat', 'sat', 'on', 'a', 'mat']

    bleu1 = bleu_score(ref, hyp, n=1)
    bleu2 = bleu_score(ref, hyp, n=2)
    bleu4 = bleu_score(ref, hyp, n=4)

    # With lower n, more lenient; with higher n, stricter
    assert bleu1 > bleu2 > bleu4
