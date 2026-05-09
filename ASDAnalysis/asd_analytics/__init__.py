from .clinical import ASDPreprocessor, FeatureEngineer, BehavioralAssessment
from .temporal import PatternDetector, DevelopmentalTrajectory, BehavioralVolatility
from .treatment import OutcomeAnalyzer, InterventionMatcher, ResponsePredictor
from .model_governance import ModelExplainer, FairnessAnalyzer, ModelValidator, UncertaintyQuantifier


class ClinicalPhenotypes:
    """Model ASD phenotypic diversity"""

    # Profile 1: Language-focused ASD
    @staticmethod
    def language_profile(A2, A7, A8):
        """Speech/language-predominant presentation"""
        return (A2 + A7 + A8) / 3

    # Profile 2: Social-focused ASD
    @staticmethod
    def social_profile(A1, A3, A8):
        """Social interaction-predominant presentation"""
        return (A1 + A3 + A8) / 3

    # Profile 3: Sensorimotor-focused ASD
    @staticmethod
    def sensorimotor_profile(A4, A5, A6):
        """Restricted/repetitive & sensory-predominant"""
        return (A4 + A5 + A6) / 3

    # Profile 4: Cognitive-focused ASD
    @staticmethod
    def cognitive_profile(A9, A10):
        """Cognitive developmental delay pattern"""
        return (A9 + A10) / 2


class ComorbidityMarkers:
    """Identify conditions mimicking or co-occurring with ASD"""

    @staticmethod
    def adhd_likelihood(A4, A5, A7):
        """Markers suggesting possible ADHD instead of/alongside ASD"""
        # Hyperactivity (A4), impulsivity (A5), but speech relatively preserved (A7)
        adhd_score = (A4 + A5) / 2 - (A7 / 2)  # High activity, better speech
        return adhd_score

    @staticmethod
    def speech_language_disorder(A2, A7, A8, A1, A3):
        """Pure speech delay without other ASD markers"""
        speech_delay_specificity = A2 + A7  # Poor speech/communication
        social_skills = A1 + A3 + A8  # Better social interaction
        return speech_delay_specificity - social_skills

    @staticmethod
    def intellectual_disability(A10, A9, A3):
        """Global developmental delay"""
        cognitive_delay = (A10 + A9) / 2  # Cognitive + play delays
        return cognitive_delay

    @staticmethod
    def anxiety_features(A1, A6, A8):
        """Anxiety mimicking social withdrawal in ASD"""
        anxiety_markers = A6  # Sensory/anxiety sensitivity
        social_avoidance = A1 + A8  # Eye contact + joint attention avoidance
        return anxiety_markers + social_avoidance