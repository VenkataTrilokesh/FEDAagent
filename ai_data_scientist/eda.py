from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .imbalance_handler import ImbalanceHandler


class EDAEngine:
    """Compute descriptive statistics and text insights for notebook generation."""

    def __init__(self) -> None:
        self.imbalance_handler = ImbalanceHandler()

    def summarize(self, df: pd.DataFrame, profile: Any) -> Dict[str, Any]:
        target = profile.target_column
        numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != target]
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c != target and not pd.api.types.is_datetime64_any_dtype(df[c])]
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        summary = {
            "dataset_overview": self._dataset_overview(df),
            "numeric_summary": self._numeric_summary(df, numeric_cols),
            "categorical_summary": self._categorical_summary(df, categorical_cols),
            "datetime_summary": self._datetime_summary(df, datetime_cols),
            "correlation_summary": self._correlation_summary(df, numeric_cols, target),
            "target_summary": self._target_summary(df, target),
            "insights": self._generate_insights(df, profile, numeric_cols, categorical_cols),
            "recommended_models": self._suggest_models(df, profile),
        }

        if target and profile.problem_type == "classification":
            summary["imbalance"] = self.imbalance_handler.detect(df, target)
        return summary

    @staticmethod
    def _dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_cells": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(float(df.memory_usage(deep=True).sum() / (1024 ** 2)), 2),
        }

    @staticmethod
    def _numeric_summary(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
        if not numeric_cols:
            return {"columns": [], "skewed_features": [], "top_variance_features": []}

        stats_df = df[numeric_cols].describe().T.round(3)
        skewness = df[numeric_cols].skew(numeric_only=True).sort_values(ascending=False)
        variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)

        return {
            "columns": numeric_cols,
            "describe": stats_df.to_dict(orient="index"),
            "skewed_features": skewness[skewness.abs() > 1].round(3).to_dict(),
            "top_variance_features": variances.head(10).round(3).to_dict(),
        }

    @staticmethod
    def _categorical_summary(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, Any]:
        details = {}
        for col in categorical_cols[:20]:
            vc = df[col].astype(str).value_counts(dropna=False).head(10)
            details[col] = {
                "nunique": int(df[col].nunique(dropna=True)),
                "top_values": vc.to_dict(),
            }
        return {"columns": categorical_cols, "details": details}

    @staticmethod
    def _datetime_summary(df: pd.DataFrame, datetime_cols: List[str]) -> Dict[str, Any]:
        details = {}
        for col in datetime_cols:
            details[col] = {
                "min": str(df[col].min()),
                "max": str(df[col].max()),
            }
        return {"columns": datetime_cols, "details": details}

    @staticmethod
    def _correlation_summary(df: pd.DataFrame, numeric_cols: List[str], target: str | None) -> Dict[str, Any]:
        if len(numeric_cols) < 2:
            return {"top_pairs": {}, "target_correlations": {}}

        corr = df[numeric_cols].corr(numeric_only=True)
        corr_pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )
        top_pairs = {f"{a} vs {b}": round(v, 3) for (a, b), v in corr_pairs.head(10).items()}

        target_corr = {}
        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            usable_cols = [c for c in numeric_cols if c != target]
            if usable_cols:
                corr_to_target = df[usable_cols + [target]].corr(numeric_only=True)[target].drop(target)
                corr_to_target = corr_to_target.sort_values(key=lambda s: s.abs(), ascending=False)
                target_corr = corr_to_target.head(10).round(3).to_dict()

        return {"top_pairs": top_pairs, "target_correlations": target_corr}

    @staticmethod
    def _target_summary(df: pd.DataFrame, target: str | None) -> Dict[str, Any]:
        if not target or target not in df.columns:
            return {"available": False}

        series = df[target]
        summary = {
            "available": True,
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "nunique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            summary["describe"] = series.describe().round(3).to_dict()
        else:
            summary["distribution"] = series.astype(str).value_counts().head(15).to_dict()
        return summary

    def _generate_insights(self, df: pd.DataFrame, profile: Any, numeric_cols: List[str], categorical_cols: List[str]) -> List[str]:
        insights: List[str] = []

        missing = df.isna().mean().sort_values(ascending=False)
        heavy_missing = missing[missing > 0.2]
        if not heavy_missing.empty:
            cols = ", ".join([f"{c} ({v:.0%})" for c, v in heavy_missing.head(5).items()])
            insights.append(f"High missingness remains concentrated in: {cols}. Consider targeted domain-aware imputation or feature removal if these columns are not business critical.")
        else:
            insights.append("After cleaning, the dataset has no materially problematic missing-value concentrations.")

        if numeric_cols:
            skewness = df[numeric_cols].skew(numeric_only=True).sort_values(key=lambda s: s.abs(), ascending=False)
            skewed = skewness[skewness.abs() > 1].head(5)
            if not skewed.empty:
                cols = ", ".join([f"{c} ({v:.2f})" for c, v in skewed.items()])
                insights.append(f"Several numeric variables are strongly skewed: {cols}. Log or Box-Cox style transformations may improve downstream model stability.")

            corr_info = self._correlation_summary(df, numeric_cols, profile.target_column)
            if corr_info["top_pairs"]:
                strongest_pair, strongest_value = next(iter(corr_info["top_pairs"].items()))
                insights.append(f"The strongest numeric relationship is {strongest_pair} with correlation {strongest_value}. This is a strong candidate for multicollinearity checks or interaction features.")

        high_cardinality = [c for c in categorical_cols if df[c].nunique(dropna=True) > 20][:5]
        if high_cardinality:
            insights.append(
                "High-cardinality categorical fields detected: "
                + ", ".join(high_cardinality)
                + ". Frequency encoding, target encoding, or grouping rare labels will help keep models efficient."
            )

        if profile.target_column and profile.problem_type == "classification":
            imbalance = self.imbalance_handler.detect(df, profile.target_column)
            if imbalance.get("is_imbalanced"):
                insights.append(
                    f"Target imbalance detected with ratio {imbalance['imbalance_ratio']}. Use SMOTE or class-aware training strategies to prevent bias toward majority classes."
                )
            else:
                insights.append("Target classes are reasonably balanced, so standard stratified training should work well.")
        elif profile.target_column and profile.problem_type == "regression":
            insights.append("A continuous target is available, so regression modeling, feature scaling, and residual analysis should be part of the modeling phase.")
        else:
            insights.append("No reliable target was inferred, so the workflow should emphasize descriptive analytics, anomaly detection, and segmentation before predictive modeling.")

        return insights

    @staticmethod
    def _suggest_models(df: pd.DataFrame, profile: Any) -> Dict[str, Any]:
        rows, cols = df.shape
        problem_type = profile.problem_type

        if problem_type == "classification":
            models = ["LogisticRegression", "RandomForestClassifier", "HistGradientBoostingClassifier"]
        elif problem_type == "regression":
            models = ["LinearRegression", "RandomForestRegressor", "HistGradientBoostingRegressor"]
        else:
            models = ["KMeans", "DBSCAN", "IsolationForest"]

        complexity = "large" if rows > 100000 or cols > 100 else "medium" if rows > 10000 or cols > 30 else "small"

        return {
            "problem_type": problem_type,
            "complexity_band": complexity,
            "starter_models": models,
            "notes": "Prefer interpretable baselines first, then add tree-based models for non-linear relationships.",
        }
