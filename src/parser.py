"""
parser.py — Converte arquivos .replay em JSON usando rrrocket.exe.

Suporta replays organizados em subpastas (higher_rank_replays / lower_rank_replays)
para classificação automática por nível de rank.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any


RRROCKET_PATH = os.path.join("bin", "rrrocket.exe")
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

# Mapeamento de nome de pasta para label
_FOLDER_LABEL_MAP = {
    "higher_rank_replays": "higher",
    "lower_rank_replays": "lower",
    "higher": "higher",
    "lower": "lower",
}


def parse_single_replay(replay_path: str) -> dict[str, Any] | None:
    """Converte 1 arquivo .replay em dict Python via rrrocket.exe.

    Args:
        replay_path: Caminho para o arquivo .replay.

    Returns:
        Dict com dados do replay, ou None em caso de erro.
    """
    if not os.path.exists(replay_path):
        print(f"[parser] Arquivo não encontrado: {replay_path}")
        return None

    try:
        result = subprocess.run(
            [RRROCKET_PATH, "-n", replay_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            print(f"[parser] Erro no rrrocket para {replay_path}: {result.stderr}")
            return None

        return json.loads(result.stdout)

    except Exception as e:
        print(f"[parser] Exceção ao processar {replay_path}: {e}")
        return None


def ensure_processed(
    replay_path: str, processed_dir: str = PROCESSED_DIR
) -> tuple[dict[str, Any], str] | None:
    """Se o JSON já existe em processed/, carrega. Senão, parseia e salva.

    Args:
        replay_path: Caminho para o arquivo .replay.
        processed_dir: Diretório onde salvar/buscar JSONs processados.

    Returns:
        Tupla (data_dict, json_path) ou None em caso de erro.
    """
    stem = Path(replay_path).stem
    target = Path(processed_dir) / f"{stem}.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    # Se já existe, carrega do cache
    if target.exists():
        print(f"[parser] Cache encontrado: {target}")
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f), str(target)

    # Senão, parseia e salva
    print(f"[parser] Processando: {replay_path}")
    data = parse_single_replay(replay_path)
    if data is None:
        return None

    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[parser] JSON salvo: {target}")
    return data, str(target)


def discover_replays(raw_dir: str = RAW_DIR) -> dict[str, list[str]]:
    """Descobre todos os .replay organizados por categoria.

    Procura em subpastas cujo nome está em _FOLDER_LABEL_MAP.
    Replays soltos na raiz de raw_dir recebem label "unlabeled".

    Returns:
        Dict mapeando label -> lista de caminhos absolutos dos replays.
        Ex: {"higher": ["data/raw/higher_rank_replays/abc.replay", ...],
             "lower": [...], "unlabeled": [...]}
    """
    root = Path(raw_dir)
    if not root.exists():
        return {}

    result: dict[str, list[str]] = {}

    # Replays em subpastas categorizadas
    for subfolder in root.iterdir():
        if subfolder.is_dir():
            label = _FOLDER_LABEL_MAP.get(subfolder.name)
            if label:
                replays = sorted(str(p) for p in subfolder.glob("*.replay"))
                if replays:
                    result.setdefault(label, []).extend(replays)

    # Replays soltos na raiz (sem categoria)
    root_replays = sorted(str(p) for p in root.glob("*.replay"))
    if root_replays:
        result["unlabeled"] = root_replays

    return result


def process_all_replays(
    raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR
) -> dict[str, list[tuple[dict[str, Any], str]]]:
    """Descobre e processa todos os replays.

    Returns:
        Dict mapeando label -> lista de (data_dict, json_path).
    """
    discovered = discover_replays(raw_dir)

    if not discovered:
        print("[parser] Nenhum replay encontrado.")
        return {}

    total = sum(len(v) for v in discovered.values())
    print(f"[parser] {total} replays encontrados: {', '.join(f'{k}={len(v)}' for k, v in discovered.items())}")

    results: dict[str, list[tuple[dict[str, Any], str]]] = {}

    for label, paths in discovered.items():
        results[label] = []
        for path in paths:
            processed = ensure_processed(path, processed_dir)
            if processed:
                results[label].append(processed)
            else:
                print(f"[parser] FALHA ao processar: {path}")

    success = sum(len(v) for v in results.values())
    print(f"[parser] Concluído: {success}/{total} replays processados com sucesso.")
    return results


if __name__ == "__main__":
    discovered = discover_replays()
    print("Replays descobertos:")
    for label, paths in discovered.items():
        print(f"  [{label}] {len(paths)} replays:")
        for p in paths:
            print(f"    - {Path(p).name}")

    print("\nProcessando todos...")
    results = process_all_replays()
    for label, items in results.items():
        print(f"\n[{label}] {len(items)} processados:")
        for data, path in items:
            num_frames = data.get("properties", {}).get("NumFrames", "?")
            map_name = data.get("properties", {}).get("MapName", "?")
            print(f"  - {Path(path).name}: {num_frames} frames, mapa={map_name}")