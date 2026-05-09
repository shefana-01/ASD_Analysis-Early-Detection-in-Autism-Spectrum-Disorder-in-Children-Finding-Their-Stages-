from .pattern_detection import PatternDetector
from .trajectory import DevelopmentalTrajectory
from .volatility import BehavioralVolatility

import numpy as np


class TemporalFeatures:
    """Extract temporal patterns from longitudinal assessments"""

    @staticmethod
    def calculate_regression_slope(age_points, behavior_scores):
        """Detect regression (negative slope) vs progression (positive slope)"""
        from numpy.polynomial import Polynomial
        p = Polynomial.fit(age_points, behavior_scores, 1)
        slope = p.convert().coef[-2]  # Regression coefficient
        return slope  # Negative = regression (Mehek), Positive = progression

    @staticmethod
    def behavioral_volatility(assessment_scores):
        """Measure variability in behavioral expression"""
        return np.std(assessment_scores)

    @staticmethod
    def skill_acquisition_rate(pre_intervention_score, post_intervention_score, months_elapsed):
        """Quantify response to intervention"""
        return (post_intervention_score - pre_intervention_score) / months_elapsed