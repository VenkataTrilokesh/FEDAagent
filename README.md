# FEDAagent - Full AI Data Science Pipeline

This project now includes:

- Full training pipeline with baseline models + metrics
- SHAP explainability module
- LIME support
- Markdown + HTML + PDF export
- Streamlit web app UI
- FastAPI upload endpoint for report/notebook generation
- Visualization plots embedded inside generated reports

## Run from CLI

```bash
pip install -r requirements.txt
python main.py --file sample_data/iris.csv --target species --output-dir artifacts
```

For `iris.csv`, output is created under:

```text
artifacts/iris_output/
```

Containing:

- notebook (`iris_auto_report.ipynb`)
- markdown/html/pdf report
- metrics JSON/profile JSON/EDA summary JSON
- plots (`*.png`) including model + explainability visuals when available

## Streamlit UI

```bash
streamlit run streamlit_app.py
```

## FastAPI endpoint

```bash
uvicorn fastapi_app:app --reload
```

POST `/generate` with multipart form-data:

- `file`: dataset file
- `target` (optional): target column name

## Notes

- SHAP/LIME plots are generated when dependencies and compatible model behavior are available.
- PDF export is generated with embedded plot images.
