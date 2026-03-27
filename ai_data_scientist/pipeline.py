from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .data_loader import DatasetLoader
from .eda import EDAEngine
from .explainability import ExplainabilityEngine
from .imbalance_handler import ImbalanceHandler
from .modeling import BaselineModelTrainer
from .notebook_generator import NotebookGenerator
from .preprocessing import DataCleaner
from .reporting import ReportBuilder


class FullPipeline:
    """End-to-end train + explain + reporting pipeline."""

    def __init__(self) -> None:
        self.loader = DatasetLoader()
        self.cleaner = DataCleaner()
        self.eda = EDAEngine()
        self.imbalance = ImbalanceHandler()
        self.notebook = NotebookGenerator()
        self.modeling = BaselineModelTrainer()
        self.explainability = ExplainabilityEngine()
        self.reporting = ReportBuilder()

    def run(
        self,
        file_path: str,
        target: Optional[str] = None,
        output_root: str = "artifacts",
        export_html: bool = True,
        export_pdf: bool = True,
    ) -> Dict[str, Any]:
        source = Path(file_path)
        dataset_stem = source.stem
        out_dir = Path(output_root) / f"{dataset_stem}_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = self.loader.load(file_path)
        profile = self.loader.profile(df, file_path, target_column=target)
        cleaning = self.cleaner.clean(df, profile)
        clean_df = cleaning.cleaned_df
        cleaning_report = cleaning.report
        eda_summary = self.eda.summarize(clean_df, profile)

        imbalance_report = None
        if profile.target_column and profile.problem_type == "classification":
            imbalance_report = self.imbalance.rebalance_train_split(clean_df, profile.target_column, profile)

        model_report = self.modeling.train_and_evaluate(clean_df, profile.target_column, profile.problem_type, str(out_dir))
        explainability_report = self.explainability.run(model_report, str(out_dir))

        generated_plots = self.reporting.generate_visualizations(clean_df, profile.target_column, str(out_dir))
        generated_plots.extend(model_report.get("plots", []))
        for key in ("shap", "lime"):
            p = explainability_report.get(key, {}).get("plot")
            if p:
                generated_plots.append(p)

        notebook_path = out_dir / f"{dataset_stem}_auto_report.ipynb"
        self.notebook.generate(
            output_path=str(notebook_path),
            dataset_path=str(source.resolve()),
            profile=profile,
            cleaning_report=cleaning_report,
            eda_summary=eda_summary,
            imbalance_report=imbalance_report,
        )

        report_md = out_dir / f"{dataset_stem}_report.md"
        report_md_path = self.reporting.build_markdown_report(
            output_path=str(report_md),
            profile=profile,
            eda_summary=eda_summary,
            cleaning_report=cleaning_report,
            model_report=model_report,
            explainability_report=explainability_report,
            plot_paths=generated_plots,
        )

        html_path = self.reporting.export_html(report_md_path) if export_html else None
        pdf_path = self.reporting.export_pdf(report_md_path) if export_pdf else None

        profile_json = out_dir / f"{dataset_stem}_profile.json"
        summary_json = out_dir / f"{dataset_stem}_eda_summary.json"
        model_json = out_dir / f"{dataset_stem}_model_metrics.json"

        profile_json.write_text(json.dumps(profile.to_dict(), indent=2, default=str), encoding="utf-8")
        summary_json.write_text(json.dumps(eda_summary, indent=2, default=str), encoding="utf-8")
        model_copy = {k: v for k, v in model_report.items() if k not in {"best_estimator", "X_train", "X_test", "y_train", "y_test"}}
        model_json.write_text(json.dumps(model_copy, indent=2, default=str), encoding="utf-8")

        return {
            "dataset": file_path,
            "output_dir": str(out_dir),
            "profile_json": str(profile_json),
            "eda_summary_json": str(summary_json),
            "model_metrics_json": str(model_json),
            "notebook": str(notebook_path),
            "report_markdown": report_md_path,
            "report_html": html_path,
            "report_pdf": pdf_path,
            "plots": generated_plots,
        }
