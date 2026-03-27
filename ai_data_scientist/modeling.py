from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid")


@dataclass
class ModelRunResult:
    model_name: str
    metrics: Dict[str, float]
    estimator: Any


class BaselineModelTrainer:
    """Train baseline models and persist evaluation artifacts."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        problem_type: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        if not target_column or target_column not in df.columns:
            return {"available": False, "message": "No valid target column for supervised training."}
        if problem_type not in {"classification", "regression"}:
            return {"available": False, "message": f"Problem type '{problem_type}' does not support supervised baselines."}

        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        stratify = y if problem_type == "classification" and y.nunique(dropna=True) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=stratify,
        )

        if problem_type == "classification":
            models = {
                "LogisticRegression": LogisticRegression(max_iter=2000),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=self.random_state),
            }
        else:
            models = {
                "LinearRegression": LinearRegression(),
                "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=self.random_state),
            }

        runs: List[ModelRunResult] = []
        best_name = ""
        best_score = -np.inf
        best_estimator = None

        for name, estimator in models.items():
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            if problem_type == "classification":
                metrics = {
                    "accuracy": float(accuracy_score(y_test, preds)),
                    "f1_weighted": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
                }
                score = metrics["f1_weighted"]
            else:
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                metrics = {
                    "rmse": rmse,
                    "mae": float(mean_absolute_error(y_test, preds)),
                    "r2": float(r2_score(y_test, preds)),
                }
                score = metrics["r2"]

            runs.append(ModelRunResult(model_name=name, metrics=metrics, estimator=pipe))
            if score > best_score:
                best_score = score
                best_name = name
                best_estimator = pipe

        plots: List[str] = []
        if problem_type == "classification" and best_estimator is not None:
            y_pred = best_estimator.predict(X_test)
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix - {best_name}")
            cm_path = output / "confusion_matrix.png"
            fig.tight_layout()
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            plots.append(str(cm_path))
        elif problem_type == "regression" and best_estimator is not None:
            y_pred = best_estimator.predict(X_test)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, y_pred, alpha=0.7)
            lo = min(float(np.min(y_test)), float(np.min(y_pred)))
            hi = max(float(np.max(y_test)), float(np.max(y_pred)))
            ax.plot([lo, hi], [lo, hi], "r--")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs Predicted - {best_name}")
            reg_path = output / "actual_vs_predicted.png"
            fig.tight_layout()
            fig.savefig(reg_path, dpi=150)
            plt.close(fig)
            plots.append(str(reg_path))

        return {
            "available": True,
            "problem_type": problem_type,
            "target_column": target_column,
            "models": [{"model": run.model_name, "metrics": run.metrics} for run in runs],
            "best_model": best_name,
            "best_estimator": best_estimator,
            "plots": plots,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        }
