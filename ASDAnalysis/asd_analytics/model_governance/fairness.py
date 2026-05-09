import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


class FairnessAnalyzer:
    """
    Detect demographic bias in ASD model predictions.
    Compares accuracy, false positive rate, and false negative rate across subgroups.
    """

    def __init__(self, y_true: pd.Series, y_pred: np.ndarray, metadata: pd.DataFrame):
        """
        metadata: DataFrame with columns like 'Gender', 'Ethnicity', 'Age_Group'
        aligned with y_true/y_pred.
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.metadata = metadata.reset_index(drop=True)

    def group_metrics(self, attribute: str) -> pd.DataFrame:
        groups = self.metadata[attribute].unique()
        rows = []
        for group in groups:
            mask = self.metadata[attribute] == group
            yt = self.y_true[mask]
            yp = self.y_pred[mask]
            if len(yt) == 0:
                continue
            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))
            tn = np.sum((yt == 0) & (yp == 0))
            rows.append({
                'group': group,
                'n': len(yt),
                'accuracy': accuracy_score(yt, yp),
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
                'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
                'positive_rate': yp.mean(),
            })
        return pd.DataFrame(rows).set_index('group')

    def disparate_impact(self, attribute: str, privileged_group: str) -> float:
        """
        Ratio of positive prediction rate for unprivileged vs privileged group.
        Value < 0.8 indicates potential bias (80% rule).
        """
        metrics = self.group_metrics(attribute)
        if privileged_group not in metrics.index:
            raise ValueError(f"Group '{privileged_group}' not found.")
        priv_rate = metrics.loc[privileged_group, 'positive_rate']
        if priv_rate == 0:
            return float('inf')
        ratios = metrics['positive_rate'] / priv_rate
        return ratios.drop(index=privileged_group).min()

    def bias_report(self, attributes: list) -> dict:
        report = {}
        for attr in attributes:
            if attr not in self.metadata.columns:
                continue
            report[attr] = self.group_metrics(attr).to_dict()
        return report
