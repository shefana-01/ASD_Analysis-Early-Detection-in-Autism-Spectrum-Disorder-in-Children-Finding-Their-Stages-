import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ASDPreprocessor:
    """Clean and encode raw ASD screening data."""

    CATEGORICAL_COLS = ['Gender', 'Ethnicity', 'Family_History', 'Class', 'Severity']

    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._impute(df)
        df = self._encode(df, fit=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        df = df.copy()
        df = self._impute(df)
        df = self._encode(df, fit=False)
        return df

    def scale(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        if fit:
            self._fitted = True
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df.fillna(df.mean(numeric_only=True), inplace=True)
        return df

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in self.CATEGORICAL_COLS:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le:
                    df[col] = le.transform(df[col].astype(str))
        return df

    def get_encoder(self, col: str) -> LabelEncoder:
        return self.encoders.get(col)
