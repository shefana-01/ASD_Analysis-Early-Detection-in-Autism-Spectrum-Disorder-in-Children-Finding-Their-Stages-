from .outcome_analysis import OutcomeAnalyzer
from .intervention_matching import InterventionMatcher
from .response_prediction import ResponsePredictor


class GeneticRiskFactors:
    """Integrate genetic and family history data"""

    @staticmethod
    def recurrence_risk_score(family_history_pattern, child_aq_score):
        """
        Calculate recurrence risk based on family pattern:
        - Parent with ASD: 40-50% risk for siblings
        - Sibling with ASD: 20-30% risk
        - No family history: 1% baseline risk
        """
        if family_history_pattern == 'parent_affected':
            base_risk = 0.45
        elif family_history_pattern == 'sibling_affected':
            base_risk = 0.25
        else:
            base_risk = 0.01

        # Adjust by child's severity
        severity_modifier = child_aq_score / 10  # Worse = higher risk
        return base_risk * (1 + severity_modifier)

    @staticmethod
    def twin_concordance_adjustment(twin_status, twin_diagnosis):
        """
        MZ twins: 90% concordance
        DZ twins: 30% concordance
        Accounts for genetic loading in family
        """
        if twin_status == 'monozygotic':
            if twin_diagnosis == 'ASD':
                prob = 0.90
            else:
                prob = 0.10
        elif twin_status == 'dizygotic':
            if twin_diagnosis == 'ASD':
                prob = 0.30
            else:
                prob = 0.70
        else:
            prob = None
        return prob


class InterventionResponseModel:
    """Predict intervention effectiveness based on baseline features"""

    @staticmethod
    def behavioral_intervention_candidacy(
            social_score, age_at_intervention, family_support,
            severity_aq_score
    ):
        """
        Behavioral interventions (ABA) work best for:
        - Younger age at intervention (< 24 months optimal)
        - Good social motivation (social_score > 0.5)
        - Family support present
        - Moderate severity (not too severe/rigid)
        """
        candidacy = (
                (age_at_intervention < 24) * 0.3 +  # Age factor
                (social_score > 0.5) * 0.3 +  # Social motivation
                (family_support == 1) * 0.2 +  # Family capacity
                ((severity_aq_score < 7) * 0.2)  # Severity match
        )
        return candidacy

    @staticmethod
    def speech_therapy_response(
            language_profile, hearing_normal,
            oral_motor_intact, motivation_level
    ):
        """
        Speech therapy success factors:
        - Speech-predominant profile
        - Normal hearing
        - Intact oral-motor system
        - Social motivation for communication
        """
        response_likelihood = (
                language_profile * 0.4 +
                (hearing_normal * 1.0) * 0.2 +
                (oral_motor_intact * 1.0) * 0.2 +
                motivation_level * 0.2
        )
        return response_likelihood

    @staticmethod
    def medication_considerations(
            comorbid_adhd_score, comorbid_anxiety_score,
            severe_aggression, sleep_disturbance
    ):
        """
        When to consider medication alongside behavioral therapy:
        - Significant comorbid ADHD/anxiety
        - Severe aggression/self-injury
        - Sleep disturbances affecting intervention capacity
        """
        medication_need = (
                comorbid_adhd_score * 0.3 +
                comorbid_anxiety_score * 0.3 +
                (severe_aggression * 1.0) * 0.2 +
                (sleep_disturbance * 1.0) * 0.2
        )
        return medication_need