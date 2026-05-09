import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


class ModelValidator:
    """Comprehensive evaluation of ASD detection model performance."""

    def __init__(self, model, X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.X_test = X_test
        self.y_test = np.array(y_test)

    def metrics(self) -> dict:
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            'accuracy':    accuracy_score(self.y_test, y_pred),
            'precision':   precision_score(self.y_test, y_pred, average='weighted'),
            'recall':      recall_score(self.y_test, y_pred, average='weighted'),
            'f1':          f1_score(self.y_test, y_pred, average='weighted'),
            'roc_auc':     roc_auc_score(self.y_test, y_prob),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'ppv':         tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'npv':         tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        }

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5, scoring: str = 'roc_auc') -> dict:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring=scoring)
        return {
            'cv_scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'scoring': scoring,
        }

    def report(self) -> pd.DataFrame:
        m = self.metrics()
        return pd.DataFrame([m], index=[type(self.model).__name__]).T
