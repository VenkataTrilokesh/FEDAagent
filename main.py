from __future__ import annotations

import argparse
from pathlib import Path

from ai_data_scientist.pipeline import FullPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full AI data science pipeline (EDA + models + explainability + reports)")
    parser.add_argument("--file", required=True, help="Path to CSV or XLSX dataset")
    parser.add_argument("--target", default=None, help="Optional explicit target column")
    parser.add_argument("--output-dir", default="artifacts", help="Root directory that will contain <dataset>_output")
    parser.add_argument("--no-html", action="store_true", help="Disable HTML export")
    parser.add_argument("--no-pdf", action="store_true", help="Disable PDF export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = FullPipeline()
    result = pipeline.run(
        file_path=args.file,
        target=args.target,
        output_root=args.output_dir,
        export_html=not args.no_html,
        export_pdf=not args.no_pdf,
    )

    print("\n=== Full Pipeline Completed ===")
    for key, value in result.items():
        if key == "plots":
            print(f"{key}: {len(value)} files")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
