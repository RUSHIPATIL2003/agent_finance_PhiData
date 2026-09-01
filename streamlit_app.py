"""Streamlit Community Cloud Root Entrypoint."""

import runpy
from pathlib import Path

app_path = Path(__file__).parent / "frontend" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
