"""
ballchasing_downloader.py — Baixa replays do ballchasing.com por faixa de rank.

Uso:
    python src/ballchasing_downloader.py --token SUA_API_KEY --count 50

O script busca replays de ranked 3v3 em duas faixas:
    - Higher: Grand Champion+
    - Lower: Bronze 1 a Silver 3

Os arquivos .replay sao salvos em:
    data/raw/higher_rank_replays/
    data/raw/lower_rank_replays/
"""

import argparse
import os
import sys
import time

import requests

API_BASE = "https://ballchasing.com/api"

RANK_GROUPS = {
    "higher": {
        "folder": "data/raw/higher_rank_replays",
        "min_rank": "grand-champion",
        "max_rank": None,
    },
    "lower": {
        "folder": "data/raw/lower_rank_replays",
        "min_rank": "bronze-1",
        "max_rank": "silver-3",
    },
}


def list_replays(token: str, min_rank: str, max_rank: str | None, count: int) -> list[dict]:
    """Busca replays filtrados por rank na API do ballchasing."""
    params = {
        "playlist": "ranked-standard",
        "min-rank": min_rank,
        "count": min(count, 200),
        "sort-by": "replay-date",
        "sort-dir": "desc",
    }
    if max_rank:
        params["max-rank"] = max_rank

    headers = {"Authorization": token}

    all_replays = []
    url = f"{API_BASE}/replays"

    while len(all_replays) < count:
        print(f"  Buscando replays... ({len(all_replays)}/{count})")
        resp = requests.get(url, headers=headers, params=params)

        if resp.status_code == 429:
            print("  Rate limit atingido, aguardando 5s...")
            time.sleep(5)
            continue

        if resp.status_code != 200:
            print(f"  Erro na API: {resp.status_code} - {resp.text[:200]}")
            break

        data = resp.json()
        replays = data.get("list", [])

        if not replays:
            print("  Nenhum replay encontrado com esses filtros.")
            break

        all_replays.extend(replays)

        next_url = data.get("next")
        if not next_url or len(all_replays) >= count:
            break

        url = next_url
        params = {}
        time.sleep(0.6)

    return all_replays[:count]


def download_replay(token: str, replay_id: str, save_path: str) -> bool:
    """Baixa um arquivo .replay pelo ID."""
    headers = {"Authorization": token}
    url = f"{API_BASE}/replays/{replay_id}/file"

    resp = requests.get(url, headers=headers)

    if resp.status_code == 429:
        print("    Rate limit no download, aguardando 5s...")
        time.sleep(5)
        resp = requests.get(url, headers=headers)

    if resp.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True

    print(f"    Falha ao baixar {replay_id}: {resp.status_code}")
    return False


def run(token: str, count_per_group: int):
    """Executa o download completo."""
    print(f"Verificando token...")
    resp = requests.get(f"{API_BASE}/", headers={"Authorization": token})
    if resp.status_code != 200:
        print(f"Token invalido ou API inacessivel (status {resp.status_code})")
        sys.exit(1)

    steam_id = resp.json().get("steam_id", "?")
    print(f"Token OK. Steam ID: {steam_id}\n")

    for group_name, config in RANK_GROUPS.items():
        folder = config["folder"]
        os.makedirs(folder, exist_ok=True)

        existing = [f for f in os.listdir(folder) if f.endswith(".replay")]
        print(f"=== {group_name.upper()} (rank {config['min_rank']} a {config.get('max_rank', 'max')}) ===")
        print(f"  Pasta: {folder} ({len(existing)} replays existentes)")

        replays = list_replays(token, config["min_rank"], config.get("max_rank"), count_per_group)
        print(f"  Encontrados: {len(replays)} replays\n")

        downloaded = 0
        skipped = 0

        for i, replay in enumerate(replays):
            replay_id = replay["id"]
            filename = f"{replay_id}.replay"
            save_path = os.path.join(folder, filename)

            if os.path.exists(save_path):
                skipped += 1
                continue

            print(f"  [{i+1}/{len(replays)}] Baixando {replay_id}...")
            success = download_replay(token, replay_id, save_path)

            if success:
                downloaded += 1

            time.sleep(1.2)

        print(f"\n  Concluido: {downloaded} baixados, {skipped} ja existiam\n")

    print("Download completo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa replays do ballchasing.com")
    parser.add_argument("--token", required=True, help="API key do ballchasing.com")
    parser.add_argument("--count", type=int, default=50, help="Replays por grupo (default: 50)")
    args = parser.parse_args()

    run(args.token, args.count)
