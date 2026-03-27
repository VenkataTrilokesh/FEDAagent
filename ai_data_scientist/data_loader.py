from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

TARGET_HINTS = {
    "target",
    "label",
    "class",
    "outcome",
    "response",
    "y",
    "saleprice",
    "price",
    "charges",
    "survived",
    "diagnosis",
    "species",
    "quality",
    "status",
}


@dataclass
class DatasetProfile:
    file_path: str
    dataset_name: str
    rows: int
    columns: int
    memory_mb: float
    duplicate_rows: int
    missing_cells: int
    missing_percent: float
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    boolean_columns: List[str]
    target_column: Optional[str]
    target_confidence: float
    problem_type: str
    feature_summary: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetLoader:
    """Load tabular data and infer a profiling schema."""

    def load(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        raise ValueError("Only CSV and Excel files are supported.")

    def profile(self, df: pd.DataFrame, file_path: str, target_column: Optional[str] = None) -> DatasetProfile:
        working_df = df.copy()
        working_df.columns = [self._clean_column_name(c) for c in working_df.columns]

        feature_summary = self._build_feature_summary(working_df)
        inferred_target, confidence, problem_type = self._infer_target(working_df, target_column)

        numeric_columns = [c for c, meta in feature_summary.items() if meta["feature_type"] == "numeric"]
        categorical_columns = [c for c, meta in feature_summary.items() if meta["feature_type"] == "categorical"]
        datetime_columns = [c for c, meta in feature_summary.items() if meta["feature_type"] == "datetime"]
        text_columns = [c for c, meta in feature_summary.items() if meta["feature_type"] == "text"]
        boolean_columns = [c for c, meta in feature_summary.items() if meta["feature_type"] == "boolean"]

        rows, cols = working_df.shape
        missing_cells = int(working_df.isna().sum().sum())
        total_cells = max(rows * cols, 1)

        return DatasetProfile(
            file_path=file_path,
            dataset_name=Path(file_path).stem,
            rows=rows,
            columns=cols,
            memory_mb=round(float(working_df.memory_usage(deep=True).sum() / (1024 ** 2)), 2),
            duplicate_rows=int(working_df.duplicated().sum()),
            missing_cells=missing_cells,
            missing_percent=round((missing_cells / total_cells) * 100, 2),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            datetime_columns=datetime_columns,
            text_columns=text_columns,
            boolean_columns=boolean_columns,
            target_column=inferred_target,
            target_confidence=confidence,
            problem_type=problem_type,
            feature_summary=feature_summary,
        )

    @staticmethod
    def _clean_column_name(name: Any) -> str:
        return str(name).strip().replace(" ", "_").replace("/", "_").replace("-", "_")

    def _build_feature_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for col in df.columns:
            series = df[col]
            inferred_type = self._infer_feature_type(series)
            summary[col] = {
                "dtype": str(series.dtype),
                "feature_type": inferred_type,
                "missing_count": int(series.isna().sum()),
                "missing_percent": round(float(series.isna().mean() * 100), 2),
                "unique_count": int(series.nunique(dropna=True)),
                "unique_ratio": round(float(series.nunique(dropna=True) / max(len(series), 1)), 4),
                "sample_values": [str(v) for v in series.dropna().astype(str).head(3).tolist()],
            }
        return summary

    def _infer_feature_type(self, series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return "categorical"

        if is_bool_dtype(series):
            return "boolean"
        if is_datetime64_any_dtype(series):
            return "datetime"
        if is_numeric_dtype(series):
            return "numeric"

        parsed = self._try_parse_datetime(non_null)
        if parsed is not None and parsed.notna().mean() >= 0.8:
            return "datetime"

        if is_string_dtype(series) or is_object_dtype(series):
            avg_len = non_null.astype(str).str.len().mean()
            nunique = non_null.nunique()
            ratio = nunique / max(len(non_null), 1)
            if avg_len > 40 or ratio > 0.5:
                return "text"
            return "categorical"

        return "categorical"

    @staticmethod
    def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
        sample = series.astype(str).head(50)
        date_like_ratio = sample.str.contains(r"[-/:]|\d{4}", regex=True, na=False).mean()
        if date_like_ratio < 0.6:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return pd.to_datetime(series, errors="coerce")
        except Exception:
            return None

    def _infer_target(self, df: pd.DataFrame, user_target: Optional[str]) -> tuple[Optional[str], float, str]:
        if user_target:
            if user_target not in df.columns:
                raise ValueError(f"Provided target column '{user_target}' does not exist in the dataset.")
            return user_target, 1.0, self._infer_problem_type(df[user_target])

        candidates: List[tuple[str, float, str]] = []
        for idx, col in enumerate(df.columns):
            series = df[col]
            name_score = 0.65 if col.lower() in TARGET_HINTS else 0.0
            last_column_bonus = 0.20 if idx == len(df.columns) - 1 else 0.0
            distinct = series.nunique(dropna=True)
            unique_ratio = distinct / max(len(series), 1)

            if is_numeric_dtype(series):
                if distinct <= 20:
                    score = 0.55 + name_score + last_column_bonus
                    problem_type = "classification"
                elif unique_ratio >= 0.95:
                    score = 0.05 + name_score
                    problem_type = "regression"
                else:
                    score = 0.45 + name_score + last_column_bonus
                    problem_type = "regression"
            else:
                score = 0.50 + name_score + last_column_bonus
                problem_type = "classification" if distinct <= max(20, int(len(df) * 0.2)) else "unsupervised"

            if unique_ratio > 0.98:
                score -= 0.35
            if distinct == 1:
                score = 0.0

            candidates.append((col, max(min(score, 0.99), 0.0), problem_type))

        if not candidates:
            return None, 0.0, "unsupervised"

        candidates.sort(key=lambda item: item[1], reverse=True)
        best_col, best_score, best_problem_type = candidates[0]
        if best_score < 0.45:
            return None, round(best_score, 2), "unsupervised"
        return best_col, round(best_score, 2), best_problem_type

    @staticmethod
    def _infer_problem_type(target: pd.Series) -> str:
        distinct = target.nunique(dropna=True)
        unique_ratio = distinct / max(len(target), 1)
        if is_numeric_dtype(target):
            if distinct <= 20:
                return "classification"
            if unique_ratio > 0.95:
                return "regression"
            return "regression"
        return "classification" if distinct <= max(20, int(len(target) * 0.2)) else "unsupervised"
