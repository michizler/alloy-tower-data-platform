"""
Launcher for the AlloyTower AVM Streamlit app.

Run from the project root:
    streamlit run run_app.py

This works around the import-path issue where Streamlit's working directory
is the directory of the script being run, which prevents `from app.lib import ...`
from resolving when Home.py is the entry point.
"""

import sys
from pathlib import Path

# Make the project root importable so `app.lib` and `api.predictor` resolve
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now run the Home page's code in this script's namespace
exec(open(PROJECT_ROOT / "app" / "Home.py", encoding="utf-8").read())
