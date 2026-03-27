from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CleaningResult:
    cleaned_df: pd.DataFrame
    report: Dict[str, Any]


class DataCleaner:
    """Rule-based cleaner for generic tabular datasets."""

    def __init__(self, outlier_cap_quantiles: tuple[float, float] = (0.01, 0.99)) -> None:
        self.outlier_cap_quantiles = outlier_cap_quantiles

    def clean(self, df: pd.DataFrame, profile: Any) -> CleaningResult:
        working_df = df.copy()
        original_shape = working_df.shape
        working_df.columns = [str(c).strip().replace(" ", "_").replace("/", "_").replace("-", "_") for c in working_df.columns]

        duplicate_rows = int(working_df.duplicated().sum())
        if duplicate_rows:
            working_df = working_df.drop_duplicates().reset_index(drop=True)

        datetime_converted = []
        for col in profile.datetime_columns:
            if col in working_df.columns:
                converted = pd.to_datetime(working_df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    working_df[col] = converted
                    datetime_converted.append(col)

        missing_strategy: Dict[str, str] = {}
        numeric_imputed = []
        categorical_imputed = []

        numeric_cols = [c for c in profile.numeric_columns if c in working_df.columns and c != profile.target_column]
        categorical_cols = [c for c in profile.categorical_columns if c in working_df.columns and c != profile.target_column]
        text_cols = [c for c in profile.text_columns if c in working_df.columns and c != profile.target_column]

        numeric_missing_ratios = working_df[numeric_cols].isna().mean() if numeric_cols else pd.Series(dtype=float)
        use_knn = bool(
            len(numeric_cols) >= 2
            and len(numeric_cols) <= 12
            and not numeric_missing_ratios.empty
            and numeric_missing_ratios.max() <= 0.25
            and numeric_missing_ratios.mean() >= 0.03
        )

        if use_knn:
            filled = working_df[numeric_cols].copy()
            for col in numeric_cols:
                filled[col] = filled[col].fillna(filled[col].median())
            from sklearn.impute import KNNImputer

            imputer = KNNImputer(n_neighbors=5, weights="distance")
            working_df[numeric_cols] = imputer.fit_transform(filled)
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    missing_strategy[col] = "KNNImputer"
                    numeric_imputed.append(col)
        else:
            for col in numeric_cols:
                if working_df[col].isna().sum() == 0:
                    continue
                skewness = working_df[col].dropna().skew()
                strategy = "median" if abs(skewness) > 1 else "mean"
                fill_value = working_df[col].median() if strategy == "median" else working_df[col].mean()
                working_df[col] = working_df[col].fillna(fill_value)
                missing_strategy[col] = strategy
                numeric_imputed.append(col)

        for col in categorical_cols + text_cols:
            if working_df[col].isna().sum() == 0:
                continue
            mode = working_df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            working_df[col] = working_df[col].fillna(fill_value)
            missing_strategy[col] = "most_frequent"
            categorical_imputed.append(col)

        if profile.target_column and profile.target_column in working_df.columns:
            target = profile.target_column
            if working_df[target].isna().sum() > 0:
                working_df = working_df.loc[working_df[target].notna()].reset_index(drop=True)
                missing_strategy[target] = "dropped_missing_target_rows"

        outlier_report = self._handle_outliers(working_df, numeric_cols)
        feature_engineering_report = self._engineer_features(working_df, profile)

        report = {
            "original_shape": original_shape,
            "final_shape": working_df.shape,
            "rows_removed": int(original_shape[0] - working_df.shape[0]),
            "duplicate_rows_removed": duplicate_rows,
            "datetime_converted": datetime_converted,
            "missing_strategies": missing_strategy,
            "numeric_imputed_columns": numeric_imputed,
            "categorical_imputed_columns": categorical_imputed,
            "outlier_report": outlier_report,
            "feature_engineering": feature_engineering_report,
        }
        return CleaningResult(cleaned_df=working_df, report=report)

    def _handle_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        report: Dict[str, Dict[str, Any]] = {}
        lower_q, upper_q = self.outlier_cap_quantiles

        for col in numeric_cols:
            series = df[col]
            if series.isna().all() or series.nunique(dropna=True) < 5:
                continue

            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_mask = (series < lower_bound) | (series > upper_bound)

            z = np.abs(stats.zscore(series, nan_policy="omit"))
            if isinstance(z, np.ndarray):
                z_mask = z > 3
                z_outliers = int(np.nansum(z_mask))
            else:
                z_outliers = 0

            outlier_count = int(iqr_mask.sum())
            if outlier_count == 0:
                continue

            cap_low = series.quantile(lower_q)
            cap_high = series.quantile(upper_q)
            df[col] = series.clip(lower=cap_low, upper=cap_high)
            report[col] = {
                "iqr_outliers": outlier_count,
                "zscore_outliers": z_outliers,
                "capped_to": [float(cap_low), float(cap_high)],
            }
        return report

    def _engineer_features(self, df: pd.DataFrame, profile: Any) -> Dict[str, Any]:
        engineered: Dict[str, Any] = {"datetime": [], "text": []}

        for col in profile.datetime_columns:
            if col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            engineered["datetime"].append(col)

        for col in profile.text_columns:
            if col not in df.columns:
                continue
            text_series = df[col].astype(str)
            df[f"{col}_char_len"] = text_series.str.len()
            df[f"{col}_word_count"] = text_series.str.split().str.len()
            engineered["text"].append(col)

        return engineered

    @staticmethod
    def build_preprocessing_recommendations(df: pd.DataFrame, profile: Any) -> Dict[str, Any]:
        numeric_cols = [c for c in profile.numeric_columns if c in df.columns and c != profile.target_column]
        categorical_cols = [c for c in profile.categorical_columns if c in df.columns and c != profile.target_column]

        encoding = {}
        for col in categorical_cols:
            cardinality = df[col].nunique(dropna=True)
            if cardinality <= 10:
                encoding[col] = "one_hot"
            elif cardinality <= 50:
                encoding[col] = "frequency_or_target_encoding"
            else:
                encoding[col] = "high_cardinality_reduce_then_target_encoding"

        scaling = {
            "recommended": bool(numeric_cols),
            "default_scaler": "StandardScaler",
            "alternative_scaler": "MinMaxScaler for distance-based models or bounded features",
        }

        return {
            "encoding_recommendations": encoding,
            "scaling_recommendations": scaling,
        }
