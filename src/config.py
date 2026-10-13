"""
config.py — Centraliza caminhos, constantes do campo de Rocket League e metadados.
Permite execução a partir de qualquer diretório de trabalho (CWD).
"""

import os
from pathlib import Path

# Diretórios base
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASET_CSV_PATH = PROCESSED_DIR / "features_dataset.csv"
MODEL_PATH = PROCESSED_DIR / "model_detector.joblib"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
BIN_DIR = PROJECT_ROOT / "bin"
RRROCKET_PATH = BIN_DIR / "rrrocket.exe"

# Constantes do campo oficial de Rocket League (Unreal Units: 10240 x 8192)
FIELD_HALF_LENGTH = 5120  # y: -5120 a +5120
FIELD_HALF_WIDTH = 4096   # x: -4096 a +4096
BALL_CHASE_DIST = 1500    # Distância para proximidade de bola (~15m)

# Dimensões da trave
GOAL_WIDTH = 1800
GOAL_DEPTH = 400

# Metadados do dataset
META_COLS = {"label", "replay_name", "team", "team_label"}

# Paleta e visualização
TEAM_COLORS = {0: "#3498db", 1: "#e67e22"}
TEAM_FILL_COLORS = {0: "rgba(52, 152, 219, 0.15)", 1: "rgba(230, 126, 34, 0.15)"}
TEAM_NAMES = {0: "Azul", 1: "Laranja"}
BALL_COLOR = "#ffffff"
DARK_BG = "#0e1117"
DARK_PAPER = "#1e1e1e"

CLUSTER_PALETTE = [
    "#e94560", "#2ecc71", "#3498db", "#e67e22",
    "#9b59b6", "#1abc9c", "#f39c12", "#e74c3c",
]
