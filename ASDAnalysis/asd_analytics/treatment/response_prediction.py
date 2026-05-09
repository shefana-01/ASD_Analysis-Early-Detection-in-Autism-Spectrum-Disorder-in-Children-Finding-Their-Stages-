import numpy as np


# Response likelihood weights derived from case study outcomes.
# Earlier intervention + lower severity = higher response probability.
_SEVERITY_WEIGHT = {'Mild': 0.85, 'Moderate': 0.65, 'Severe': 0.35}
_MAX_AGE_FOR_FULL_BENEFIT = 3.0   # years — critical window


class ResponsePredictor:
    """
    Predict the likelihood that a child will respond well to intervention.
    Scale: 0.0 (unlikely) to 1.0 (highly likely).
    """

    def predict(self, severity: str, diagnosis_age: float,
                family_support: bool = True) -> float:
        base = _SEVERITY_WEIGHT.get(severity, 0.5)

        # Age penalty: every year past the critical window reduces benefit
        age_factor = max(0.0, 1.0 - (diagnosis_age - _MAX_AGE_FOR_FULL_BENEFIT) * 0.12)
        age_factor = min(age_factor, 1.0)

        support_bonus = 0.05 if family_support else 0.0

        score = base * age_factor + support_bonus
        return round(min(score, 1.0), 3)

    def interpret(self, probability: float) -> str:
        if probability >= 0.75:
            return 'High response likelihood — strong candidate for early intervention'
        elif probability >= 0.50:
            return 'Moderate response likelihood — intervention recommended'
        elif probability >= 0.30:
            return 'Low-moderate — intensive support needed'
        return 'Low response likelihood — requires specialised clinical management'

    def predict_with_interpretation(self, severity: str, diagnosis_age: float,
                                    family_support: bool = True) -> dict:
        prob = self.predict(severity, diagnosis_age, family_support)
        return {
            'probability': prob,
            'percentage': f'{prob * 100:.1f}%',
            'interpretation': self.interpret(prob),
        }
