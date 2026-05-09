import numpy as np
import pandas as pd


class UncertaintyQuantifier:
    """
    Quantify prediction confidence for ASD classification.
    Uses probability margin and entropy to flag uncertain cases for clinical review.
    """

    def __init__(self, model):
        self.model = model

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def entropy(self, X: np.ndarray) -> np.ndarray:
        probs = self.probabilities(X)
        # Shannon entropy per sample
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log2(probs), axis=1)

    def confidence(self, X: np.ndarray) -> np.ndarray:
        """Max class probability — higher = more confident."""
        return self.probabilities(X).max(axis=1)

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Difference between top-2 class probabilities."""
        probs = self.probabilities(X)
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        return sorted_probs[:, 0] - sorted_probs[:, 1]

    def flag_uncertain(self, X: np.ndarray,
                       confidence_threshold: float = 0.70) -> np.ndarray:
        """Return boolean mask: True = uncertain, needs clinical review."""
        return self.confidence(X) < confidence_threshold

    def summary_df(self, X: np.ndarray) -> pd.DataFrame:
        probs = self.probabilities(X)
        return pd.DataFrame({
            'prob_no_asd':  probs[:, 0],
            'prob_asd':     probs[:, 1],
            'confidence':   self.confidence(X),
            'margin':       self.margin(X),
            'entropy':      self.entropy(X),
            'needs_review': self.flag_uncertain(X),
        })
