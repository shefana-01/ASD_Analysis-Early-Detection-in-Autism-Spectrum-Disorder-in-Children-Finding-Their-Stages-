import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DevelopmentalTrajectory:
    """
    Track and compare developmental trajectories over time.
    Supports multi-case longitudinal comparison (e.g. Mehek vs Tajun vs Tayeeba).
    """

    def __init__(self):
        self.cases = {}

    def add_case(self, name: str, timepoints: list, scores: list):
        """
        timepoints: list of ages in years [1.0, 1.5, 2.0, ...]
        scores: corresponding AQ/severity scores at each timepoint
        """
        self.cases[name] = {'timepoints': timepoints, 'scores': scores}

    def slope(self, name: str) -> float:
        """Linear slope of score over time (positive = worsening)."""
        data = self.cases[name]
        if len(data['timepoints']) < 2:
            return 0.0
        return float(np.polyfit(data['timepoints'], data['scores'], 1)[0])

    def outcome_at_age(self, name: str, age: float) -> float:
        """Interpolate score at a given age."""
        data = self.cases[name]
        return float(np.interp(age, data['timepoints'], data['scores']))

    def compare_all(self) -> pd.DataFrame:
        rows = []
        for name, data in self.cases.items():
            rows.append({
                'Case': name,
                'First_Age': data['timepoints'][0],
                'Last_Age': data['timepoints'][-1],
                'Initial_Score': data['scores'][0],
                'Final_Score': data['scores'][-1],
                'Trend_Slope': self.slope(name),
            })
        return pd.DataFrame(rows)

    def plot(self, save_path: str = None):
        plt.figure(figsize=(10, 6))
        for name, data in self.cases.items():
            plt.plot(data['timepoints'], data['scores'], marker='o', label=name)
        plt.xlabel('Age (years)')
        plt.ylabel('AQ Score / Severity')
        plt.title('Developmental Trajectories')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
