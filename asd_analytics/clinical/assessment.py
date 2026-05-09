"""A1-A10 behavioral marker definitions and scoring guide."""

MARKERS = {
    'A1': {'name': 'Eye Contact',        'domain': 'Social'},
    'A2': {'name': 'Speech Development', 'domain': 'Communication'},
    'A3': {'name': 'Social Interaction', 'domain': 'Social'},
    'A4': {'name': 'Repetitive Behavior','domain': 'Restricted'},
    'A5': {'name': 'Self-Stimulation',   'domain': 'Restricted'},
    'A6': {'name': 'Sensory Issues',     'domain': 'Restricted'},
    'A7': {'name': 'Communication',      'domain': 'Communication'},
    'A8': {'name': 'Joint Attention',    'domain': 'Social'},
    'A9': {'name': 'Pretend Play',       'domain': 'Social'},
    'A10':{'name': 'Cognitive Function', 'domain': 'Cognitive'},
}

SEVERITY_THRESHOLDS = {
    'Mild':     (0.0, 0.33),
    'Moderate': (0.34, 0.66),
    'Severe':   (0.67, 1.0),
}


class BehavioralAssessment:
    """Score and interpret a single patient's A1-A10 responses."""

    def __init__(self, responses: dict):
        """
        responses: dict mapping A1..A10 to 0 (absent) or 1 (present).
        """
        self.responses = responses

    def total_score(self) -> int:
        return sum(self.responses.get(k, 0) for k in MARKERS)

    def domain_scores(self) -> dict:
        scores = {}
        for marker, info in MARKERS.items():
            domain = info['domain']
            scores[domain] = scores.get(domain, 0) + self.responses.get(marker, 0)
        return scores

    def severity(self) -> str:
        ratio = self.total_score() / len(MARKERS)
        for label, (low, high) in SEVERITY_THRESHOLDS.items():
            if low <= ratio <= high:
                return label
        return 'Severe'

    def asd_flag(self) -> bool:
        """Simple threshold: >= 6 positive markers flags ASD risk."""
        return self.total_score() >= 6

    def summary(self) -> dict:
        return {
            'total_score': self.total_score(),
            'max_score': len(MARKERS),
            'domain_scores': self.domain_scores(),
            'severity': self.severity(),
            'asd_flag': self.asd_flag(),
        }
