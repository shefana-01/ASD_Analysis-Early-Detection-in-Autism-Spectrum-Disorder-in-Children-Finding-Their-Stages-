import pandas as pd


class PatternDetector:
    """
    Classify ASD onset type: Regression vs Early-onset.

    Regression (Mehek phenotype): normal early development followed by
    skill loss after 18-24 months.

    Early-onset: ASD signs present from infancy with no regression.
    """

    REGRESSION_INDICATORS = [
        'speech_loss', 'skill_regression', 'late_onset_symptoms'
    ]

    def classify(self, record: dict) -> str:
        """
        record keys expected:
          - diagnosis_age (float, years)
          - had_normal_early_development (bool)
          - skill_loss_reported (bool)
          - age_of_first_concern (float, years)
        """
        had_normal = record.get('had_normal_early_development', False)
        skill_loss = record.get('skill_loss_reported', False)
        first_concern = record.get('age_of_first_concern', 0)

        if had_normal and skill_loss and first_concern >= 1.5:
            return 'Regression'
        return 'Early-onset'

    def classify_dataframe(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(lambda row: self.classify(row.to_dict()), axis=1)

    def regression_risk_score(self, record: dict) -> float:
        """Return 0-1 probability of regression type based on available signals."""
        score = 0.0
        if record.get('had_normal_early_development'):
            score += 0.4
        if record.get('skill_loss_reported'):
            score += 0.4
        if record.get('age_of_first_concern', 0) >= 1.5:
            score += 0.2
        return min(score, 1.0)
