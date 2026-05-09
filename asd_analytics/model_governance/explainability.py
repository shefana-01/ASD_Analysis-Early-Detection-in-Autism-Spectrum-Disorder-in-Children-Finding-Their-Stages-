import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ModelExplainer:
    """
    Feature importance and explainability for tree-based ASD detection models.
    Provides SHAP-style ranking without requiring the shap package as hard dependency.
    """

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names

    def feature_importance_df(self) -> pd.DataFrame:
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"{type(self.model).__name__} does not expose feature_importances_.")
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances,
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        return df

    def top_features(self, n: int = 10) -> pd.DataFrame:
        return self.feature_importance_df().head(n)

    def plot_importance(self, top_n: int = 15, save_path: str = None):
        df = self.feature_importance_df().head(top_n).sort_values('importance')
        plt.figure(figsize=(10, 6))
        plt.barh(df['feature'], df['importance'], color='steelblue')
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importances — {type(self.model).__name__}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def explain_prediction(self, x_row: np.ndarray) -> pd.DataFrame:
        """Return per-feature contribution estimate for a single prediction."""
        importance = self.model.feature_importances_
        contribution = importance * x_row
        df = pd.DataFrame({
            'feature': self.feature_names,
            'value': x_row,
            'importance': importance,
            'contribution': contribution,
        }).sort_values('contribution', ascending=False)
        return df

    def shap_summary(self, X: np.ndarray) -> pd.DataFrame:
        """
        Approximate SHAP summary: mean absolute contribution per feature over dataset.
        Use the real shap library for production use.
        """
        importance = self.model.feature_importances_
        mean_contributions = importance * np.abs(X).mean(axis=0)
        df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_contribution': mean_contributions,
        }).sort_values('mean_abs_contribution', ascending=False)
        return df
