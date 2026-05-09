"""Match a patient profile to the most appropriate intervention type."""

INTERVENTIONS = {
    'ABA': {
        'description': 'Applied Behavior Analysis — structured skill-building',
        'best_for': ['Severe', 'Moderate'],
        'min_age': 1.5,
        'domains': ['Social', 'Communication', 'Cognitive'],
    },
    'ESDM': {
        'description': 'Early Start Denver Model — play-based naturalistic therapy',
        'best_for': ['Mild', 'Moderate'],
        'min_age': 1.0,
        'domains': ['Social', 'Communication'],
    },
    'PECS': {
        'description': 'Picture Exchange Communication System — for non-verbal children',
        'best_for': ['Severe'],
        'min_age': 2.0,
        'domains': ['Communication'],
    },
    'Social_Skills_Training': {
        'description': 'Group-based social skill development',
        'best_for': ['Mild'],
        'min_age': 4.0,
        'domains': ['Social'],
    },
    'Sensory_Integration': {
        'description': 'Occupational therapy targeting sensory processing',
        'best_for': ['Mild', 'Moderate', 'Severe'],
        'min_age': 1.5,
        'domains': ['Restricted'],
    },
}


class InterventionMatcher:
    """Recommend interventions based on severity, age, and impaired domains."""

    def recommend(self, severity: str, age: float, domains: list) -> list:
        """
        Returns list of recommended intervention names ranked by relevance.
        domains: list of impaired domains e.g. ['Social', 'Communication']
        """
        matches = []
        for name, info in INTERVENTIONS.items():
            if severity not in info['best_for']:
                continue
            if age < info['min_age']:
                continue
            overlap = len(set(domains) & set(info['domains']))
            matches.append((name, overlap, info['description']))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [{'intervention': m[0], 'description': m[2]} for m in matches]

    def full_plan(self, severity: str, age: float, domains: list) -> dict:
        recommendations = self.recommend(severity, age, domains)
        return {
            'severity': severity,
            'age': age,
            'impaired_domains': domains,
            'recommended_interventions': recommendations,
            'priority': recommendations[0]['intervention'] if recommendations else None,
        }
