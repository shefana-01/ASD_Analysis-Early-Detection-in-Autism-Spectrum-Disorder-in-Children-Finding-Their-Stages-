import numpy as np
import pandas as pd


class BehavioralVolatility:
    """
    Measure consistency of behavioral scores over time.
    High volatility may indicate regression-type ASD or unstable symptom profile.
    """

    def __init__(self, timepoints: list, scores: list):
        self.timepoints = np.array(timepoints)
        self.scores = np.array(scores, dtype=float)

    def std(self) -> float:
        return float(np.std(self.scores))

    def coefficient_of_variation(self) -> float:
        mean = np.mean(self.scores)
        return float(self.std() / mean) if mean != 0 else 0.0

    def max_change(self) -> float:
        """Largest single-step change between consecutive timepoints."""
        if len(self.scores) < 2:
            return 0.0
        return float(np.max(np.abs(np.diff(self.scores))))

    def is_high_volatility(self, threshold: float = 0.3) -> bool:
        return self.coefficient_of_variation() > threshold

    def summary(self) -> dict:
        return {
            'std': self.std(),
            'cv': self.coefficient_of_variation(),
            'max_change': self.max_change(),
            'high_volatility': self.is_high_volatility(),
        }

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, time_col: str, score_col: str):
        df = df.sort_values(time_col)
        return cls(df[time_col].tolist(), df[score_col].tolist())
