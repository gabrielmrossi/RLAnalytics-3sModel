"""
scripts/build_dataset.py — CLI para extração e geração do dataset de features a partir dos replays.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import build_dataset


def main():
    print("[*] Iniciando extração de features de replays...")
    df = build_dataset()
    if not df.empty:
        print(f"[+] Dataset gerado com sucesso! Shape: {df.shape}")
    else:
        print("[-] Nenhum dado pôde ser extraído.")


if __name__ == "__main__":
    main()
