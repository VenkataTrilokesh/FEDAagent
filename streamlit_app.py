from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ai_data_scientist.pipeline import FullPipeline

st.set_page_config(page_title="AI Data Scientist", layout="wide")
st.title("AI Data Scientist - Full Pipeline UI")

uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
target = st.text_input("Optional target column", value="")
run_btn = st.button("Run full pipeline")

if run_btn and uploaded is not None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir) / uploaded.name
        temp_path.write_bytes(uploaded.getvalue())

        pipeline = FullPipeline()
        result = pipeline.run(
            file_path=str(temp_path),
            target=target.strip() or None,
            output_root="artifacts",
            export_html=True,
            export_pdf=True,
        )

        st.success("Pipeline completed.")
        st.write(result)

        out_dir = Path(result["output_dir"])
        image_files = sorted(out_dir.glob("*.png"))
        if image_files:
            st.subheader("Generated Plots")
            for img in image_files:
                st.image(str(img), caption=img.name)

        html = result.get("report_html")
        if html and Path(html).exists():
            st.subheader("HTML Report")
            st.markdown(f"[Open HTML report]({html})")
else:
    st.info("Upload a file and click 'Run full pipeline'.")
