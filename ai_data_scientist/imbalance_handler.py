from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except Exception:  # pragma: no cover
    RandomOverSampler = None
    RandomUnderSampler = None
    SMOTE = None


class ImbalanceHandler:
    """Detect and optionally rebalance classification datasets."""

    def detect(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        value_counts = df[target_column].value_counts(dropna=False)
        if value_counts.empty:
            return {
                "is_imbalanced": False,
                "class_distribution": {},
                "imbalance_ratio": 1.0,
            }

        max_count = int(value_counts.max())
        min_count = int(value_counts.min())
        imbalance_ratio = round(max_count / max(min_count, 1), 2)
        minority_share = round(min_count / max(int(value_counts.sum()), 1), 4)
        n_classes = int(value_counts.shape[0])

        if n_classes <= 2:
            is_imbalanced = bool(imbalance_ratio >= 1.5 or minority_share < 0.4)
        else:
            is_imbalanced = bool(imbalance_ratio >= 1.5)

        return {
            "is_imbalanced": is_imbalanced,
            "class_distribution": value_counts.to_dict(),
            "imbalance_ratio": imbalance_ratio,
            "minority_share": minority_share,
        }

    def rebalance_train_split(self, df: pd.DataFrame, target_column: str, profile: Any) -> Dict[str, Any]:
        if SMOTE is None or RandomOverSampler is None or RandomUnderSampler is None:
            return {
                "available": False,
                "message": "imbalanced-learn is not installed. Install it to enable SMOTE/oversampling/undersampling.",
            }

        X = df.drop(columns=[target_column])
        y = df[target_column]
        numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical_features = [c for c in X.columns if c not in numeric_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if y.nunique() <= max(20, int(len(y) * 0.2)) else None,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", self._make_one_hot_encoder()),
                    ]),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        X_train_processed = preprocessor.fit_transform(X_train)
        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()

        class_counts = Counter(y_train)
        min_class_size = min(class_counts.values())
        if min_class_size >= 6:
            sampler = SMOTE(random_state=42)
            sampler_name = "SMOTE"
        elif min_class_size >= 2:
            sampler = RandomOverSampler(random_state=42)
            sampler_name = "RandomOverSampler"
        else:
            sampler = RandomUnderSampler(random_state=42)
            sampler_name = "RandomUnderSampler"

        X_resampled, y_resampled = sampler.fit_resample(X_train_processed, y_train)

        return {
            "available": True,
            "sampler": sampler_name,
            "before": dict(Counter(y_train)),
            "after": dict(Counter(y_resampled)),
            "train_shape_before": tuple(np.asarray(X_train_processed).shape),
            "train_shape_after": tuple(np.asarray(X_resampled).shape),
            "test_shape": tuple(np.asarray(X_test).shape),
        }

    @staticmethod
    def _make_one_hot_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:  # scikit-learn < 1.2
            return OneHotEncoder(handle_unknown="ignore", sparse=False)
