from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_data_scientist.data_loader import DatasetLoader
from ai_data_scientist.eda import EDAEngine
from ai_data_scientist.imbalance_handler import ImbalanceHandler
from ai_data_scientist.notebook_generator import NotebookGenerator
from ai_data_scientist.preprocessing import DataCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated AI Data Scientist for CSV/XLSX datasets")
    parser.add_argument("--file", required=True, help="Path to CSV or XLSX dataset")
    parser.add_argument("--target", default=None, help="Optional explicit target column")
    parser.add_argument("--output-dir", default="artifacts", help="Directory to store generated notebook and reports")
    parser.add_argument("--title", default=None, help="Optional custom report title (reserved for future use)")
    parser.add_argument("--export-html", action="store_true", help="Export generated notebook to HTML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader()
    cleaner = DataCleaner()
    eda_engine = EDAEngine()
    imbalance_handler = ImbalanceHandler()
    notebook_generator = NotebookGenerator()

    df = loader.load(args.file)
    profile = loader.profile(df, args.file, target_column=args.target)
    cleaning_result = cleaner.clean(df, profile)
    clean_df = cleaning_result.cleaned_df
    cleaning_report = cleaning_result.report
    eda_summary = eda_engine.summarize(clean_df, profile)

    imbalance_report = None
    if profile.target_column and profile.problem_type == "classification":
        imbalance_report = imbalance_handler.rebalance_train_split(clean_df, profile.target_column, profile)

    notebook_path = output_dir / f"{profile.dataset_name}_auto_report.ipynb"
    notebook_generator.generate(
        output_path=str(notebook_path),
        dataset_path=str(Path(args.file).resolve()),
        profile=profile,
        cleaning_report=cleaning_report,
        eda_summary=eda_summary,
        imbalance_report=imbalance_report,
    )

    profile_json = output_dir / f"{profile.dataset_name}_profile.json"
    with open(profile_json, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2, default=str)

    summary_json = output_dir / f"{profile.dataset_name}_eda_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(eda_summary, f, indent=2, default=str)

    html_path = None
    if args.export_html:
        html_path = notebook_generator.export_html(str(notebook_path))

    print("\n=== AI Data Scientist Pipeline Completed ===")
    print(f"Dataset: {args.file}")
    print(f"Target column: {profile.target_column}")
    print(f"Problem type: {profile.problem_type}")
    print(f"Notebook: {notebook_path}")
    print(f"Profile JSON: {profile_json}")
    print(f"EDA Summary JSON: {summary_json}")
    if html_path:
        print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
