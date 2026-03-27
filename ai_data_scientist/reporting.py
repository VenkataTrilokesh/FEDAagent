from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ReportBuilder:
    """Create visual assets and rich markdown/html/pdf reports."""

    def generate_visualizations(
        self,
        df: pd.DataFrame,
        target_column: Optional[str],
        output_dir: str,
    ) -> List[str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]

        missing = df.isna().mean().sort_values(ascending=False)
        missing = missing[missing > 0]
        if not missing.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            missing.head(20).sort_values().mul(100).plot(kind="barh", ax=ax, color="#5B8FF9")
            ax.set_title("Missing Value (%)")
            ax.set_xlabel("Percent")
            fig.tight_layout()
            p = out / "missing_values.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(str(p))

        if len(numeric_cols) >= 1:
            sample_cols = numeric_cols[: min(4, len(numeric_cols))]
            fig, axes = plt.subplots(len(sample_cols), 1, figsize=(9, 3 * len(sample_cols)))
            axes = np.atleast_1d(axes)
            for ax, col in zip(axes, sample_cols):
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution - {col}")
            fig.tight_layout()
            p = out / "numeric_distributions.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(str(p))

        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Correlation Heatmap")
            fig.tight_layout()
            p = out / "correlation_heatmap.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(str(p))

        if target_column and target_column in df.columns:
            if target_column in numeric_cols and len(numeric_cols) >= 2:
                x_col = [c for c in numeric_cols if c != target_column][0]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=df, x=x_col, y=target_column, ax=ax)
                ax.set_title(f"{x_col} vs {target_column}")
                fig.tight_layout()
                p = out / "target_scatter.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                paths.append(str(p))
            else:
                top = df[target_column].astype(str).value_counts().head(20)
                fig, ax = plt.subplots(figsize=(9, 6))
                sns.barplot(x=top.values, y=top.index, ax=ax)
                ax.set_title(f"Target Distribution - {target_column}")
                ax.set_xlabel("Count")
                fig.tight_layout()
                p = out / "target_distribution.png"
                fig.savefig(p, dpi=150)
                plt.close(fig)
                paths.append(str(p))

        if categorical_cols:
            col = categorical_cols[0]
            top = df[col].astype(str).value_counts().head(15)
            fig, ax = plt.subplots(figsize=(9, 6))
            sns.barplot(x=top.values, y=top.index, ax=ax)
            ax.set_title(f"Top Categories - {col}")
            ax.set_xlabel("Count")
            fig.tight_layout()
            p = out / "categorical_top_values.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths.append(str(p))

        return paths

    def build_markdown_report(
        self,
        output_path: str,
        profile: Any,
        eda_summary: Dict[str, Any],
        cleaning_report: Dict[str, Any],
        model_report: Dict[str, Any],
        explainability_report: Dict[str, Any],
        plot_paths: List[str],
    ) -> str:
        report_file = Path(output_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        model_lines = []
        if model_report.get("available"):
            for item in model_report.get("models", []):
                metrics = ", ".join([f"{k}: {v:.4f}" for k, v in item.get("metrics", {}).items()])
                model_lines.append(f"- **{item['model']}** → {metrics}")
            model_lines.append(f"- **Best model:** {model_report.get('best_model', 'N/A')}")
        else:
            model_lines.append(f"- {model_report.get('message', 'Modeling unavailable.')}")

        explain_lines = []
        shap_meta = explainability_report.get("shap", {})
        lime_meta = explainability_report.get("lime", {})
        explain_lines.append(f"- SHAP: {'available' if shap_meta.get('available') else shap_meta.get('message', 'unavailable')}")
        explain_lines.append(f"- LIME: {'available' if lime_meta.get('available') else lime_meta.get('message', 'unavailable')}")

        figures_md = "\n".join([f"![{Path(p).stem}]({Path(p).name})" for p in plot_paths])

        md = f"""# AutoML + EDA Report: {profile.dataset_name}

## Dataset Overview
- Rows: **{profile.rows}**
- Columns: **{profile.columns}**
- Target: **{profile.target_column}**
- Problem type: **{profile.problem_type}**
- Missing %: **{profile.missing_percent}**

## Cleaning Summary
- Original shape: **{cleaning_report.get('original_shape')}**
- Final shape: **{cleaning_report.get('final_shape')}**
- Duplicate rows removed: **{cleaning_report.get('duplicate_rows_removed')}**

## Baseline Modeling Metrics
{chr(10).join(model_lines)}

## Explainability
{chr(10).join(explain_lines)}

## Key EDA Insights
{chr(10).join([f'- {i}' for i in eda_summary.get('insights', [])])}

## Visualizations
{figures_md}
"""

        report_file.write_text(md, encoding="utf-8")
        return str(report_file)

    def export_html(self, markdown_path: str) -> str:
        path = Path(markdown_path)
        html_path = path.with_suffix(".html")
        body = path.read_text(encoding="utf-8").splitlines()
        html_parts = ["<html><body>"]
        for line in body:
            if line.startswith("# "):
                html_parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("- "):
                html_parts.append(f"<p>{line}</p>")
            elif line.startswith("![") and "](" in line:
                img = line.split("](")[1][:-1]
                html_parts.append(f"<img src='{img}' style='max-width:900px; width:100%; margin-bottom:20px;' />")
            else:
                html_parts.append(f"<p>{line}</p>")
        html_parts.append("</body></html>")
        html_path.write_text("\n".join(html_parts), encoding="utf-8")
        return str(html_path)

    def export_pdf(self, markdown_path: str) -> Optional[str]:
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except Exception:
            return None

        md_path = Path(markdown_path)
        pdf_path = md_path.with_suffix(".pdf")
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter
        y = height - 40

        for line in md_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("![") and "](" in line:
                img = line.split("](")[1][:-1]
                img_path = md_path.parent / img
                if img_path.exists():
                    if y < 260:
                        c.showPage()
                        y = height - 40
                    c.drawImage(str(img_path), 40, y - 220, width=520, height=200, preserveAspectRatio=True, mask='auto')
                    y -= 235
                continue

            text = line.strip() or " "
            if y < 50:
                c.showPage()
                y = height - 40
            c.drawString(40, y, text[:110])
            y -= 14

        c.save()
        return str(pdf_path)
