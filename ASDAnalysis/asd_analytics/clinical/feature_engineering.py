import pandas as pd


class FeatureEngineer:
    """Create composite behavioral indices from A1-A10 markers."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Social_Communication_Score'] = (df['A1'] + df['A2'] + df['A3']) / 3
        df['Repetitive_Behavior_Score'] = (df['A4'] + df['A5'] + df['A6']) / 3
        df['Interaction_Score'] = (df['A7'] + df['A8'] + df['A9']) / 3
        df['Cognitive_Score'] = df['A10']
        df['Behavioral_Complexity'] = (
            df['Social_Communication_Score'] + df['Repetitive_Behavior_Score']
        ) / 2
        return df

    @staticmethod
    def feature_names() -> list:
        return [
            'Social_Communication_Score',
            'Repetitive_Behavior_Score',
            'Interaction_Score',
            'Cognitive_Score',
            'Behavioral_Complexity',
        ]

    @staticmethod
    def behavioral_markers() -> list:
        return [f'A{i}' for i in range(1, 11)]
