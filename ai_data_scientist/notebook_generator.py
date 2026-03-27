from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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
            self._code(self._imports_cell()),
            self._code(self._config_cell(dataset_path, profile.target_column)),
            self._md("## 1) Load and Profile Data"),
            self._code(self._load_and_profile_cell()),
            self._md(self._overview_markdown(profile, eda_summary)),
            self._md("## 2) Clean + Feature Engineering"),
            self._code(self._cleaning_cell()),
            self._md(self._cleaning_markdown(cleaning_report)),
            self._md("## 3) EDA Visuals"),
            self._code(self._eda_visuals_cell()),
            self._md("## 4) Baseline Model Training + Metrics"),
            self._code(self._training_cell()),
            self._md("## 5) Explainability (SHAP + LIME)"),
            self._code(self._explainability_cell()),
            self._md("## 6) Save Report Assets"),
            self._code(self._save_assets_cell()),
            self._md(self._imbalance_markdown(imbalance_report, profile)),
        ]

        nbf.write(nb, output)
        return str(output)

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
        return f"# Automated Full Pipeline Notebook\n\n**Dataset:** `{profile.dataset_name}`  \n**Rows:** {profile.rows:,}  \n**Columns:** {profile.columns}  \n**Target:** `{profile.target_column}`  \n**Problem type:** `{profile.problem_type}`\n"

    @staticmethod
    def _imports_cell() -> str:
        return '''from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ai_data_scientist.data_loader import DatasetLoader
from ai_data_scientist.preprocessing import DataCleaner
from ai_data_scientist.eda import EDAEngine
from ai_data_scientist.reporting import ReportBuilder
from ai_data_scientist.modeling import BaselineModelTrainer
from ai_data_scientist.explainability import ExplainabilityEngine

pd.set_option("display.max_columns", 200)
'''

    @staticmethod
    def _config_cell(dataset_path: str, target_column: Optional[str]) -> str:
        return f'''DATASET_PATH = r"{dataset_path}"
TARGET_OVERRIDE = {repr(target_column)}
loader = DatasetLoader()
cleaner = DataCleaner()
eda_engine = EDAEngine()
report_builder = ReportBuilder()
trainer = BaselineModelTrainer()
explainer = ExplainabilityEngine()
'''

    @staticmethod
    def _load_and_profile_cell() -> str:
        return '''df = loader.load(DATASET_PATH)
profile = loader.profile(df, DATASET_PATH, target_column=TARGET_OVERRIDE)
print("Shape:", df.shape)
print("Target:", profile.target_column)
print("Problem type:", profile.problem_type)
df.head()'''

    @staticmethod
    def _cleaning_cell() -> str:
        return '''cleaning_result = cleaner.clean(df, profile)
clean_df = cleaning_result.cleaned_df
cleaning_report = cleaning_result.report
print(cleaning_report)
clean_df.head()'''

    @staticmethod
    def _eda_visuals_cell() -> str:
        return '''eda_summary = eda_engine.summarize(clean_df, profile)
plots = report_builder.generate_visualizations(clean_df, profile.target_column, "./notebook_plots")
print("Generated plots:", plots)
eda_summary["insights"]'''

    @staticmethod
    def _training_cell() -> str:
        return '''model_report = trainer.train_and_evaluate(clean_df, profile.target_column, profile.problem_type, "./notebook_plots")
model_report.get("models", [])'''

    @staticmethod
    def _explainability_cell() -> str:
        return '''explainability_report = explainer.run(model_report, "./notebook_plots")
explainability_report'''

    @staticmethod
    def _save_assets_cell() -> str:
        return '''markdown_report = report_builder.build_markdown_report(
    output_path="./notebook_plots/notebook_report.md",
    profile=profile,
    eda_summary=eda_summary,
    cleaning_report=cleaning_report,
    model_report=model_report,
    explainability_report=explainability_report,
    plot_paths=plots + model_report.get("plots", []),
)
report_builder.export_html(markdown_report)
report_builder.export_pdf(markdown_report)
markdown_report'''

    def _overview_markdown(self, profile: Any, eda_summary: Dict[str, Any]) -> str:
        overview = eda_summary["dataset_overview"]
        model_info = eda_summary["recommended_models"]
        return (
            "### Dataset Snapshot\n\n"
            f"- Memory footprint: **{overview['memory_mb']} MB**\n"
            f"- Duplicate rows: **{overview['duplicates']}**\n"
            f"- Missing cells: **{overview['missing_cells']}**\n"
            f"- Numeric features: **{len(profile.numeric_columns)}**\n"
            f"- Categorical features: **{len(profile.categorical_columns)}**\n\n"
            f"Starter models: {', '.join(model_info['starter_models'])}."
        )

    def _cleaning_markdown(self, cleaning_report: Dict[str, Any]) -> str:
        return (
            "### Cleaning Summary\n\n"
            f"- Original shape: **{cleaning_report['original_shape']}**\n"
            f"- Final shape: **{cleaning_report['final_shape']}**\n"
            f"- Duplicate rows removed: **{cleaning_report['duplicate_rows_removed']}**\n"
            f"- Missing strategies: **{json.dumps(cleaning_report.get('missing_strategies', {}))}**"
        )

    def _imbalance_markdown(self, imbalance_report: Optional[Dict[str, Any]], profile: Any) -> str:
        if not profile.target_column:
            return "### Imbalance Review\n\nNo target detected."
        if profile.problem_type != "classification":
            return "### Imbalance Review\n\nNot a classification task."
        if not imbalance_report:
            return "### Imbalance Review\n\nNo resampling summary generated."
        return (
            "### Imbalance Review\n\n"
            f"- Sampler used: **{imbalance_report.get('sampler', 'N/A')}**\n"
            f"- Before balancing: **{json.dumps(imbalance_report.get('before', {}))}**\n"
            f"- After balancing: **{json.dumps(imbalance_report.get('after', {}))}**"
        )
