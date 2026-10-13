"""
streamlit_app.py — Fachada de compatibilidade com a aplicação Streamlit modularizada em app/.
Permite execução contínua via `streamlit run src/streamlit_app.py` ou `streamlit run app/main.py`.
Reexporta funções de diagnóstico para retrocompatibilidade com a suíte de testes.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"

for p in (str(SRC_DIR), str(APP_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from app.main import main
from app.services import (
    check_stagnation_anomaly,
    generate_tactical_narrative,
)

# Aliases de compatibilidade para suítes de testes existentes
_check_stagnation_anomaly = check_stagnation_anomaly
_generate_tactical_narrative = generate_tactical_narrative

if __name__ == "__main__":
    # Quando executado via `streamlit run src/streamlit_app.py` ou python src/streamlit_app.py
    main()
