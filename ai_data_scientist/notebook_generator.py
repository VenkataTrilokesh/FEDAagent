from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import nbformat as nbf


class NotebookGenerator:
    """Build a structured Jupyter notebook for a dataset analysis workflow."""

    def generate(
        self,
        output_path: str,
        dataset_path: str,
        profile: Any,
        cleaning_report: Dict[str, Any],
        eda_summary: Dict[str, Any],
        imbalance_report: Optional[Dict[str, Any]] = None,
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        nb = nbf.v4.new_notebook()
        nb["cells"] = [
            self._md(self._title_block(profile)),
            self._md(self._problem_understanding(profile)),
            self._code(self._imports_cell()),
            self._code(self._config_cell(dataset_path, profile.target_column)),
            self._md("## 1. Data Loading"),
            self._code(self._load_and_profile_cell()),
            self._md(self._overview_markdown(profile, eda_summary)),
            self._md("## 2. Data Cleaning & Preprocessing"),
            self._code(self._cleaning_cell()),
            self._md(self._cleaning_markdown(cleaning_report)),
            self._md("## 3. Feature Engineering"),
            self._code(self._feature_engineering_review_cell()),
            self._md(self._feature_engineering_markdown(cleaning_report)),
            self._md("## 4. Exploratory Data Analysis (EDA)"),
            self._code(self._eda_visuals_cell()),
            self._md(self._eda_insights_markdown(eda_summary)),
            self._md("## 5. Target Analysis & Imbalance Handling"),
            self._code(self._imbalance_cell()),
            self._md(self._imbalance_markdown(imbalance_report, profile)),
            self._md("## 6. Model Suggestions & Next Steps"),
            self._code(self._model_suggestion_cell()),
            self._md(self._conclusion_markdown(eda_summary, profile)),
        ]

        nbf.write(nb, output)
        return str(output)

    @staticmethod
    def export_html(notebook_path: str) -> Optional[str]:
        try:
            from nbconvert import HTMLExporter
        except Exception:
            return None

        html_exporter = HTMLExporter()
        body, _ = html_exporter.from_filename(notebook_path)
        html_path = str(Path(notebook_path).with_suffix(".html"))
        Path(html_path).write_text(body, encoding="utf-8")
        return html_path

    @staticmethod
    def _md(text: str):
        return nbf.v4.new_markdown_cell(text)

    @staticmethod
    def _code(text: str):
        return nbf.v4.new_code_cell(text)

    @staticmethod
    def _bullet_list(items: Iterable[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    def _title_block(self, profile: Any) -> str:
        return f"# Automated AI Data Scientist Report\n\n**Dataset:** `{profile.dataset_name}`  \n**Rows:** {profile.rows:,}  \n**Columns:** {profile.columns}  \n**Inferred target:** `{profile.target_column}`  \n**Problem type:** `{profile.problem_type}`\n\nThis notebook was auto-generated to mirror a professional EDA + feature engineering workflow with business-facing commentary."

    def _problem_understanding(self, profile: Any) -> str:
        target_text = (
            f"The system inferred **`{profile.target_column}`** as the likely target with confidence **{profile.target_confidence:.2f}**."
            if profile.target_column
            else "No reliable target was inferred, so the notebook focuses on descriptive analytics and feature discovery."
        )
        return (
            "## Problem Understanding\n\n"
            "This notebook follows an end-to-end workflow:\n"
            "1. Load the dataset\n"
            "2. Audit quality issues\n"
            "3. Clean and preprocess the data\n"
            "4. Engineer useful features\n"
            "5. Run deep EDA\n"
            "6. Summarize business insights and modeling recommendations\n\n"
            f"{target_text}"
        )

    def _overview_markdown(self, profile: Any, eda_summary: Dict[str, Any]) -> str:
        overview = eda_summary["dataset_overview"]
        model_info = eda_summary["recommended_models"]
        return (
            "### Initial Dataset Snapshot\n\n"
            f"- Memory footprint: **{overview['memory_mb']} MB**\n"
            f"- Duplicate rows: **{overview['duplicates']}**\n"
            f"- Missing cells: **{overview['missing_cells']}**\n"
            f"- Numeric features: **{len(profile.numeric_columns)}**\n"
            f"- Categorical features: **{len(profile.categorical_columns)}**\n"
            f"- Datetime features: **{len(profile.datetime_columns)}**\n"
            f"- Text features: **{len(profile.text_columns)}**\n\n"
            f"**Recommended starter models:** {', '.join(model_info['starter_models'])}."
        )

    def _cleaning_markdown(self, cleaning_report: Dict[str, Any]) -> str:
        strategies = cleaning_report.get("missing_strategies", {})
        if strategies:
            strategy_text = self._bullet_list([f"`{col}` → `{strategy}`" for col, strategy in strategies.items()])
        else:
            strategy_text = "- No imputation was necessary."

        outliers = cleaning_report.get("outlier_report", {})
        outlier_text = (
            self._bullet_list([
                f"`{col}` had {meta['iqr_outliers']} IQR outliers and was capped to {meta['capped_to']}"
                for col, meta in list(outliers.items())[:8]
            ])
            if outliers
            else "- No major outlier capping was required."
        )

        return (
            "### Cleaning Summary\n\n"
            f"- Original shape: **{cleaning_report['original_shape']}**\n"
            f"- Final shape: **{cleaning_report['final_shape']}**\n"
            f"- Rows removed: **{cleaning_report['rows_removed']}**\n"
            f"- Duplicate rows removed: **{cleaning_report['duplicate_rows_removed']}**\n\n"
            "**Missing-value handling**\n"
            f"{strategy_text}\n\n"
            "**Outlier handling**\n"
            f"{outlier_text}"
        )

    def _feature_engineering_markdown(self, cleaning_report: Dict[str, Any]) -> str:
        fe = cleaning_report.get("feature_engineering", {})
        dt_features = fe.get("datetime", [])
        text_features = fe.get("text", [])
        lines = []
        if dt_features:
            lines.append("Datetime decomposition created year, month, day, and weekday features for: " + ", ".join(f"`{c}`" for c in dt_features))
        if text_features:
            lines.append("Text-length features created character and word-count features for: " + ", ".join(f"`{c}`" for c in text_features))
        if not lines:
            lines.append("No additional datetime or text features were required for this dataset.")
        return "### Feature Engineering Notes\n\n" + self._bullet_list(lines)

    def _eda_insights_markdown(self, eda_summary: Dict[str, Any]) -> str:
        insights = eda_summary.get("insights", [])
        correlation = eda_summary.get("correlation_summary", {}).get("top_pairs", {})
        corr_block = (
            self._bullet_list([f"{pair}: {value}" for pair, value in correlation.items()][:5])
            if correlation
            else "- Not enough numeric features for pairwise correlation analysis."
        )
        return (
            "### Key Insights\n\n"
            f"{self._bullet_list(insights)}\n\n"
            "**Top correlation pairs**\n"
            f"{corr_block}"
        )

    def _imbalance_markdown(self, imbalance_report: Optional[Dict[str, Any]], profile: Any) -> str:
        if not profile.target_column:
            return "### Imbalance Review\n\nNo target column was available, so class-imbalance handling was skipped."
        if profile.problem_type != "classification":
            return "### Imbalance Review\n\nThe inferred task is not classification, so imbalance handling is not required."
        if not imbalance_report:
            return "### Imbalance Review\n\nClassification target detected, but no resampling summary was generated."
        if not imbalance_report.get("available", True):
            return f"### Imbalance Review\n\n{imbalance_report.get('message', 'Resampling library unavailable.')}"
        return (
            "### Imbalance Review\n\n"
            f"- Sampler used: **{imbalance_report.get('sampler', 'N/A')}**\n"
            f"- Before balancing: **{json.dumps(imbalance_report.get('before', {}))}**\n"
            f"- After balancing: **{json.dumps(imbalance_report.get('after', {}))}**\n"
            f"- Train shape before: **{imbalance_report.get('train_shape_before', 'N/A')}**\n"
            f"- Train shape after: **{imbalance_report.get('train_shape_after', 'N/A')}**"
        )

    def _conclusion_markdown(self, eda_summary: Dict[str, Any], profile: Any) -> str:
        models = eda_summary["recommended_models"]["starter_models"]
        return (
            "## 7. Conclusion\n\n"
            "This auto-generated report identified the main structure of the dataset, cleaned the core quality issues,"
            " surfaced useful features, and summarized the most actionable EDA findings.\n\n"
            f"For modeling, start with: **{', '.join(models)}**. "
            "Track data leakage, validate transformations in cross-validation, and document business assumptions before production deployment.\n\n"
            f"If the inferred target **`{profile.target_column}`** is incorrect, rerun the pipeline with an explicit `--target` argument."
        )

    @staticmethod
    def _imports_cell() -> str:
        return '''from pathlib import Path
import sys

import pandas as pd
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 100)

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ai_data_scientist.data_loader import DatasetLoader
from ai_data_scientist.preprocessing import DataCleaner
from ai_data_scientist.eda import EDAEngine
from ai_data_scientist.visualization import Visualizer
from ai_data_scientist.imbalance_handler import ImbalanceHandler'''

    @staticmethod
    def _config_cell(dataset_path: str, target_column: Optional[str]) -> str:
        return f'''DATASET_PATH = r"{dataset_path}"
TARGET_OVERRIDE = {repr(target_column)}
loader = DatasetLoader()
cleaner = DataCleaner()
eda_engine = EDAEngine()
visualizer = Visualizer()
imbalance_handler = ImbalanceHandler()'''

    @staticmethod
    def _load_and_profile_cell() -> str:
        return '''df = loader.load(DATASET_PATH)
profile = loader.profile(df, DATASET_PATH, target_column=TARGET_OVERRIDE)

print("Dataset shape:", df.shape)
print("Target column:", profile.target_column)
print("Problem type:", profile.problem_type)
print("Numeric columns:", profile.numeric_columns[:10])
print("Categorical columns:", profile.categorical_columns[:10])
print("Datetime columns:", profile.datetime_columns[:10])
print("Text columns:", profile.text_columns[:10])

df.head()'''

    @staticmethod
    def _cleaning_cell() -> str:
        return '''cleaning_result = cleaner.clean(df, profile)
clean_df = cleaning_result.cleaned_df
cleaning_report = cleaning_result.report

print(cleaning_report)
clean_df.head()'''

    @staticmethod
    def _feature_engineering_review_cell() -> str:
        return '''recommendations = cleaner.build_preprocessing_recommendations(clean_df, profile)
print("Preprocessing recommendations:")
print(recommendations)

clean_df.head()'''

    @staticmethod
    def _eda_visuals_cell() -> str:
        return '''eda_summary = eda_engine.summarize(clean_df, profile)
print("Dataset overview:")
print(eda_summary["dataset_overview"])

visualizer.plot_missing_values(clean_df)
visualizer.plot_numeric_distributions(clean_df, clean_df.select_dtypes(include="number").columns[:6])
visualizer.plot_boxplots(clean_df, clean_df.select_dtypes(include="number").columns[:6])
visualizer.plot_categorical_counts(clean_df, [c for c in clean_df.columns if clean_df[c].dtype == "object"][:6])
visualizer.plot_correlation_heatmap(clean_df)
visualizer.plot_pairplot(clean_df, clean_df.select_dtypes(include="number").columns[:5], target_col=profile.target_column)
visualizer.plot_target_relationships(clean_df, profile.target_column)
visualizer.plot_interactive_summary(clean_df, profile.target_column)'''

    @staticmethod
    def _imbalance_cell() -> str:
        return '''if profile.target_column and profile.problem_type == "classification":
    detection = imbalance_handler.detect(clean_df, profile.target_column)
    print("Imbalance detection:", detection)
    rebalance_report = imbalance_handler.rebalance_train_split(clean_df, profile.target_column, profile)
    print("Rebalancing report:", rebalance_report)
else:
    print("Imbalance handling skipped because no classification target was detected.")'''

    @staticmethod
    def _model_suggestion_cell() -> str:
        return '''print("Recommended starter models:")
print(eda_summary["recommended_models"])

if profile.target_column:
    print("Top target summary:")
    print(eda_summary["target_summary"])
else:
    print("No target summary available.")'''