"""BLEU score computation for caption evaluation."""
from collections import Counter
from typing import List
import math


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """Extract n-grams as a Counter."""
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams


def bleu_score(reference: List[str], hypothesis: List[str], n: int = 4) -> float:
    """Compute BLEU-n score.

    Args:
        reference: Reference caption as a list of tokens.
        hypothesis: Hypothesis caption as a list of tokens.
        n: Maximum n-gram size (BLEU-n).

    Returns:
        BLEU score in [0, 1].
    """
    if not hypothesis:
        return 0.0

    score = 1.0
    for i in range(1, min(n + 1, len(hypothesis) + 1)):
        ref_ngrams = _get_ngrams(reference, i)
        hyp_ngrams = _get_ngrams(hypothesis, i)

        matches = sum((hyp_ngrams & ref_ngrams).values())
        total = sum(hyp_ngrams.values())

        if total == 0:
            matches = 0
            total = 1

        precision = matches / total if total > 0 else 0
        score *= precision ** (1 / n)

    brevity_penalty = 1.0 if len(hypothesis) >= len(reference) else math.exp(1 - len(reference) / len(hypothesis))

    return score * brevity_penalty
