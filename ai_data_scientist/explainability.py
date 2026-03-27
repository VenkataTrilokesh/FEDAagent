from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ExplainabilityEngine:
    """Generate SHAP and LIME explanations for trained baseline model."""

    def run(self, model_report: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        if not model_report.get("available"):
            return {"available": False, "message": "Model report unavailable."}

        estimator = model_report.get("best_estimator")
        X_test: pd.DataFrame = model_report.get("X_test")
        if estimator is None or X_test is None or X_test.empty:
            return {"available": False, "message": "No trained model or test data for explainability."}

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        shap_report = self._run_shap(estimator, X_test, output)
        lime_report = self._run_lime(estimator, X_test, output)

        return {
            "available": True,
            "shap": shap_report,
            "lime": lime_report,
        }

    def _run_shap(self, estimator: Any, X_test: pd.DataFrame, output: Path) -> Dict[str, Any]:
        try:
            import shap
        except Exception:
            return {"available": False, "message": "shap is not installed."}

        sample = X_test.head(min(120, len(X_test))).copy()
        try:
            predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
            explainer = shap.Explainer(predict_fn, sample)
            shap_values = explainer(sample)

            plt.figure(figsize=(8, 6))
            shap.plots.beeswarm(shap_values, max_display=12, show=False)
            shap_path = output / "shap_summary.png"
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150)
            plt.close()
            return {"available": True, "plot": str(shap_path)}
        except Exception as exc:
            return {"available": False, "message": f"SHAP failed: {exc}"}

    def _run_lime(self, estimator: Any, X_test: pd.DataFrame, output: Path) -> Dict[str, Any]:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except Exception:
            return {"available": False, "message": "lime is not installed."}

        if X_test.empty:
            return {"available": False, "message": "No samples for LIME."}

        try:
            sample = X_test.head(min(300, len(X_test))).copy()
            explainer = LimeTabularExplainer(
                training_data=sample.to_numpy(),
                feature_names=sample.columns.tolist(),
                mode="classification" if hasattr(estimator, "predict_proba") else "regression",
                discretize_continuous=True,
            )
            row = sample.iloc[0].to_numpy()
            predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
            explanation = explainer.explain_instance(row, predict_fn, num_features=min(10, sample.shape[1]))

            labels, vals = zip(*explanation.as_list()) if explanation.as_list() else ([], [])
            fig, ax = plt.subplots(figsize=(9, 6))
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, vals, color="#7b68ee")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_title("LIME Local Explanation (first sample)")
            ax.axvline(x=0, color="black", linewidth=1)
            fig.tight_layout()
            lime_path = output / "lime_explanation.png"
            fig.savefig(lime_path, dpi=150)
            plt.close(fig)
            return {"available": True, "plot": str(lime_path)}
        except Exception as exc:
            return {"available": False, "message": f"LIME failed: {exc}"}
