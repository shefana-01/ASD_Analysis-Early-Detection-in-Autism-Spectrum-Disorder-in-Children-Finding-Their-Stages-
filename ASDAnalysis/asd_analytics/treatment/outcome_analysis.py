import pandas as pd
import matplotlib.pyplot as plt


# Based on real case study findings in the research manuscript.
# 12-month earlier diagnosis → 40-50% improvement in long-term outcomes.
INTERVENTION_OUTCOME_MAP = {
    'Early':  {'independence': 0.70, 'speech_words': 35, 'severity': 'Mild-Moderate'},
    'Middle': {'independence': 0.40, 'speech_words': 15, 'severity': 'Moderate'},
    'Late':   {'independence': 0.10, 'speech_words': 0,  'severity': 'Severe'},
}


class OutcomeAnalyzer:
    """Analyse treatment effectiveness based on intervention timing."""

    def categorize_timing(self, diagnosis_age: float) -> str:
        if diagnosis_age <= 2.0:
            return 'Early'
        elif diagnosis_age <= 3.0:
            return 'Middle'
        return 'Late'

    def predict_outcome(self, diagnosis_age: float) -> dict:
        timing = self.categorize_timing(diagnosis_age)
        outcome = INTERVENTION_OUTCOME_MAP[timing].copy()
        outcome['timing_category'] = timing
        outcome['diagnosis_age'] = diagnosis_age
        return outcome

    def compare_cases(self, cases: list) -> pd.DataFrame:
        """
        cases: list of dicts with keys 'name' and 'diagnosis_age'
        """
        rows = []
        for case in cases:
            outcome = self.predict_outcome(case['diagnosis_age'])
            outcome['name'] = case['name']
            rows.append(outcome)
        return pd.DataFrame(rows).set_index('name')

    def plot_intervention_impact(self, save_path: str = None):
        timings = list(INTERVENTION_OUTCOME_MAP.keys())
        independence = [INTERVENTION_OUTCOME_MAP[t]['independence'] * 100 for t in timings]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(timings, independence, color=['#2ecc71', '#f39c12', '#e74c3c'])
        plt.ylabel('Predicted Independence (%)')
        plt.title('Impact of Intervention Timing on Long-Term Outcomes')
        for bar, val in zip(bars, independence):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.0f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
