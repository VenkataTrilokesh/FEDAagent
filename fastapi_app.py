from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from ai_data_scientist.pipeline import FullPipeline

app = FastAPI(title="AI Data Scientist API")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/generate")
async def generate_report(file: UploadFile = File(...), target: str | None = Form(default=None)) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / file.filename
        input_path.write_bytes(await file.read())

        pipeline = FullPipeline()
        result = pipeline.run(
            file_path=str(input_path),
            target=target or None,
            output_root="artifacts",
            export_html=True,
            export_pdf=True,
        )
        return result
