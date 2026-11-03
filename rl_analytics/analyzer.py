"""
analyzer.py — Motor de análise de replays de Rocket League.

Recebe o dict JSON gerado pelo rrrocket.exe e extrai dados estruturados:
- Mapeamento de actors (carros, bola, boost, jogadores, times)
- Posições (x, y, z) e velocidades de cada jogador e da bola (~1 amostra/segundo)
- Timeline de boost por jogador
- Metadados (placar, mapa, duração, estatísticas dos jogadores)
- Eventos (gols, demolições)
"""

import math
from typing import Any

import pandas as pd


class ReplayAnalyzer:
    """Analisa o JSON de um replay e extrai dados estruturados.

    Uso:
        data = json.load(open("data/processed/replay.json"))
        analyzer = ReplayAnalyzer(data)
        result = analyzer.analyze()
        positions_df = result["positions"]
        boost_df = result["boost"]
    """

    # object_id para RigidBody (confirmado na investigação)
    _RIGIDBODY_NAMES = ("TAGame.RBActor_TA:ReplicatedRBState",)

    def __init__(self, data: dict[str, Any]):
        self.data = data
        self.objects: list[str] = data.get("objects", [])
        self.frames: list[dict] = data.get("network_frames", {}).get("frames", [])
        self._obj_to_id: dict[str, int] = {name: idx for idx, name in enumerate(self.objects)}

        # Registros preenchidos por build_actor_registry()
        self._car_actors: dict[int, dict] = {}     # car_actor_id -> {pri_id, name, team}
        self._pri_info: dict[int, dict] = {}       # pri_actor_id -> {name, team}
        self._ball_actor_ids: set[int] = set()      # actor_ids da bola
        self._component_to_car: dict[int, int] = {} # ANY CarComponent actor_id -> car_actor_id
        self._rigidbody_object_ids: set[int] = set()  # object_ids que indicam RigidBody

    def _get_object_id(self, name: str) -> int | None:
        """Busca O(1) do índice de um objeto pelo nome exato."""
        return self._obj_to_id.get(name)

    def _find_object_ids_containing(self, substring: str) -> set[int]:
        """Busca O(N) única para objetos que contenham a substring."""
        return {idx for name, idx in self._obj_to_id.items() if substring in name}

    def _update_car_to_pri(
        self,
        actor: dict,
        car_to_pri: dict[int, int],
        obj_player_ri_id: int | None,
    ) -> bool:
        """Atualiza car_to_pri se o actor for uma atualização de PlayerReplicationInfo."""
        if obj_player_ri_id is not None and actor.get("object_id") == obj_player_ri_id:
            aid = actor.get("actor_id")
            attr = actor.get("attribute", {})
            if isinstance(attr, dict):
                act = attr.get("ActiveActor", {})
                target = act.get("actor", -1)
                if target >= 0 and target in self._pri_info:
                    car_to_pri[aid] = target
                elif target < 0:
                    car_to_pri.pop(aid, None)
                return True
        return False

    # ──────────────────────────────────────────────
    #  Etapa 1: Mapear Actors
    # ──────────────────────────────────────────────

    def build_actor_registry(self) -> None:
        """Percorre os frames e constrói mapeamentos completos de actors.

        Mapeia:
            - Carros (Archetypes.Car.Car_Default) -> actor_id
            - Bola (Archetypes.Ball.Ball_Default) -> actor_id
            - PRIs (PlayerReplicationInfo) -> nome, time
            - Boost components (CarComponent_Boost) -> car_actor_id
            - Trata re-criação de carros em demolições/respawn
        """
        # Passo 1: Identificar o object_id para RigidBody
        self._rigidbody_object_ids = {
            self._obj_to_id[name] for name in self._RIGIDBODY_NAMES if name in self._obj_to_id
        }

        # Identificar object_ids importantes (lookup O(1))
        obj_player_name_id = self._get_object_id("Engine.PlayerReplicationInfo:PlayerName")
        obj_vehicle_link_id = self._get_object_id("TAGame.CarComponent_TA:Vehicle")
        obj_player_ri_id = self._get_object_id("Engine.Pawn:PlayerReplicationInfo")

        # Passo 2: Percorrer frames para construir registros
        # PRIs: actor_id -> {name, team}
        pri_info: dict[int, dict] = {}
        # Car -> PRI links
        car_to_pri: dict[int, int] = {}

        # Todos os actor_ids do Team0 e Team1
        team_actor_ids: dict[int, int] = {}  # actor_id -> team_num

        for frame in self.frames:
            # new_actors: identificar carros, bola, boost components, teams
            for actor in frame.get("new_actors", []):
                aid = actor.get("actor_id")
                oid = actor.get("object_id")
                if oid is None or oid >= len(self.objects):
                    continue

                obj_name = self.objects[oid]

                if "Car.Car_Default" in obj_name:
                    self._car_actors.setdefault(aid, {})
                elif "Ball.Ball_Default" in obj_name:
                    self._ball_actor_ids.add(aid)
                elif "Teams.Team0" in obj_name:
                    team_actor_ids[aid] = 0
                elif "Teams.Team1" in obj_name:
                    team_actor_ids[aid] = 1

            # updated_actors: extrair nomes, times, links
            for actor in frame.get("updated_actors", []):
                aid = actor.get("actor_id")
                oid = actor.get("object_id")
                attr = actor.get("attribute", {})

                if not isinstance(attr, dict) or oid is None:
                    continue

                # Player name (String attribute em PlayerReplicationInfo:PlayerName)
                if oid == obj_player_name_id and "String" in attr:
                    pri_info.setdefault(aid, {})["name"] = attr["String"]

                # Team link (ActiveActor apontando para Team0/Team1)
                if "ActiveActor" in attr:
                    active = attr["ActiveActor"]
                    if isinstance(active, dict) and "actor" in active:
                        target = active["actor"]

                        # PRI -> Team
                        if target in team_actor_ids:
                            pri_info.setdefault(aid, {})["team"] = team_actor_ids[target]

                        # Car -> PRI (Pawn:PlayerReplicationInfo)
                        if oid == obj_player_ri_id and aid in self._car_actors:
                            if target >= 0:  # -1 = desconectou
                                car_to_pri[aid] = target

                        # Any CarComponent -> Car (CarComponent_TA:Vehicle)
                        if oid == obj_vehicle_link_id:
                            self._component_to_car[aid] = target

        # Passo 3: Salvar pri_info e resolver cadeia Car -> PRI -> nome/time
        self._pri_info = pri_info
        for car_id, pri_id in car_to_pri.items():
            info = pri_info.get(pri_id, {})
            name = info.get("name", f"Player_{car_id}")
            team = info.get("team", -1)

            # Filtrar servidor (PRIs com nome contendo "SAM" que não estão linkados a carros)
            # O filtro é: se aparece como PRI de um carro, é jogador legítimo
            self._car_actors[car_id] = {
                "pri_id": pri_id,
                "name": name,
                "team": team,
            }

        # Limpar carros que não têm PRI (fantasmas)
        self._car_actors = {
            k: v for k, v in self._car_actors.items() if "pri_id" in v
        }

    def get_players(self) -> list[dict]:
        """Retorna lista de jogadores com nome, time e car_actor_id."""
        seen_names = set()
        players = []
        for pri_id, info in self._pri_info.items():
            name = info.get("name", "")
            team = info.get("team", -1)
            if not name or team not in (0, 1) or name in seen_names:
                continue
            seen_names.add(name)
            car_id = next((cid for cid, cinfo in self._car_actors.items() if cinfo.get("pri_id") == pri_id), pri_id)
            players.append({
                "name": name,
                "team": team,
                "car_actor_id": car_id,
            })
        return sorted(players, key=lambda p: (p["team"], p["name"]))

    # ──────────────────────────────────────────────
    #  Etapa 2: Extrair Posições
    # ──────────────────────────────────────────────

    def extract_positions(self, sample_interval_frames: int = 30) -> pd.DataFrame:
        """Extrai posição (x,y,z) e velocidade de cada carro e bola.

        Rastreia a ligação dinâmica entre Carros e PRI frame-a-frame, respeitando
        a criação e destruição de atores (tratando deleted_actors e evitando
        reaproveitamento indevido de IDs de atores da Unreal Engine).

        Returns:
            DataFrame com colunas:
                frame, time, entity, team, x, y, z, vx, vy, vz
            Onde entity = nome do jogador ou "ball".
        """
        obj_player_ri_id = self._get_object_id("Engine.Pawn:PlayerReplicationInfo")

        # Rastreamento dinâmico frame-a-frame
        car_to_pri: dict[int, int] = {}
        active_ball_actors: set[int] = set()

        # Estado corrente: entity_name -> {team, x, y, z, vx, vy, vz}
        current_state: dict[str, dict] = {}
        rows: list[dict] = []

        for fi, frame in enumerate(self.frames):
            time_val = frame.get("time", 0.0)

            # 1. Identificar novos atores da bola
            for actor in frame.get("new_actors", []):
                aid = actor.get("actor_id")
                oid = actor.get("object_id")
                if oid is not None and oid < len(self.objects):
                    if "Ball.Ball_Default" in self.objects[oid]:
                        active_ball_actors.add(aid)

            # 2. Remover atores destruídos (deleted_actors)
            for da in frame.get("deleted_actors", []):
                car_to_pri.pop(da, None)
                active_ball_actors.discard(da)

            # 3. Atualizar links Car -> PRI
            for actor in frame.get("updated_actors", []):
                self._update_car_to_pri(actor, car_to_pri, obj_player_ri_id)

            # 4. Atualizar física (RigidBody)
            for actor in frame.get("updated_actors", []):
                aid = actor.get("actor_id")
                oid = actor.get("object_id")

                if oid not in self._rigidbody_object_ids:
                    continue

                entity_name = None
                team = -1

                if aid in car_to_pri:
                    pri_id = car_to_pri[aid]
                    pinfo = self._pri_info.get(pri_id, {})
                    entity_name = pinfo.get("name")
                    team = pinfo.get("team", -1)
                elif aid in active_ball_actors or aid in self._ball_actor_ids:
                    entity_name = "ball"
                    team = -1

                if not entity_name:
                    continue

                attr = actor.get("attribute", {})
                rb = attr.get("RigidBody", {})
                loc = rb.get("location", {})
                vel = rb.get("linear_velocity") or {}

                current_state[entity_name] = {
                    "team": team,
                    "x": loc.get("x", 0.0),
                    "y": loc.get("y", 0.0),
                    "z": loc.get("z", 0.0),
                    "vx": vel.get("x", 0.0),
                    "vy": vel.get("y", 0.0),
                    "vz": vel.get("z", 0.0),
                }

            # 5. Emitir snapshot no intervalo de amostragem
            if fi % sample_interval_frames == 0 and current_state:
                active_player_names = {
                    self._pri_info[pid]["name"]
                    for pid in car_to_pri.values()
                    if pid in self._pri_info
                }
                for entity_name, state in current_state.items():
                    if entity_name == "ball":
                        if not active_ball_actors and not self._ball_actor_ids:
                            continue
                    else:
                        if entity_name not in active_player_names:
                            continue

                    rows.append({
                        "frame": fi,
                        "time": time_val,
                        "entity": entity_name,
                        "team": state["team"],
                        "x": state["x"],
                        "y": state["y"],
                        "z": state["z"],
                        "vx": state["vx"],
                        "vy": state["vy"],
                        "vz": state["vz"],
                    })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Ordenar por frame e entidade
        df = df.sort_values(["frame", "entity"]).reset_index(drop=True)
        return df

    # ──────────────────────────────────────────────
    #  Etapa 3: Extrair Boost
    # ──────────────────────────────────────────────

    def extract_boost_timeline(self, sample_interval_frames: int = 30) -> pd.DataFrame:
        """Extrai nível de boost de cada jogador ao longo do tempo.

        Rastreia os componentes de boost e carros de forma dinâmica frame-a-frame.

        Returns:
            DataFrame com colunas:
                frame, time, player_name, team, boost_pct (0-100)
        """
        obj_boost_id = self._get_object_id("TAGame.CarComponent_Boost_TA:ReplicatedBoost")
        obj_vehicle_link_id = self._get_object_id("TAGame.CarComponent_TA:Vehicle")
        obj_player_ri_id = self._get_object_id("Engine.Pawn:PlayerReplicationInfo")

        if obj_boost_id is None:
            return pd.DataFrame()

        component_to_car: dict[int, int] = {}
        car_to_pri: dict[int, int] = {}

        current_boost: dict[str, float] = {}
        player_teams: dict[str, int] = {}
        rows: list[dict] = []

        for fi, frame in enumerate(self.frames):
            time_val = frame.get("time", 0.0)

            # Limpar atores deletados
            for da in frame.get("deleted_actors", []):
                component_to_car.pop(da, None)
                car_to_pri.pop(da, None)

            # Atualizar componentes e carros
            for actor in frame.get("updated_actors", []):
                aid = actor.get("actor_id")
                oid = actor.get("object_id")
                attr = actor.get("attribute", {})
                if not isinstance(attr, dict):
                    continue

                if oid == obj_vehicle_link_id:
                    component_to_car[aid] = attr.get("ActiveActor", {}).get("actor", -1)

                elif self._update_car_to_pri(actor, car_to_pri, obj_player_ri_id):
                    pass

                elif oid == obj_boost_id:
                    car_id = component_to_car.get(aid)
                    if car_id is None:
                        car_id = self._component_to_car.get(aid)
                    if car_id is not None:
                        pri_id = car_to_pri.get(car_id)
                        if pri_id is None:
                            car_info = self._car_actors.get(car_id, {})
                            pri_id = car_info.get("pri_id")
                        if pri_id is not None and pri_id in self._pri_info:
                            pdata = self._pri_info[pri_id]
                            player_name = pdata.get("name")
                            team = pdata.get("team", -1)
                            if player_name:
                                boost_data = attr.get("ReplicatedBoost", {})
                                boost_amount = boost_data.get("boost_amount", 0)
                                boost_pct = round((boost_amount / 255.0) * 100.0, 1)
                                current_boost[player_name] = boost_pct
                                player_teams[player_name] = team

            # Emitir snapshot apenas nos frames de amostragem
            if fi % sample_interval_frames == 0 and current_boost:
                active_players = {
                    self._pri_info[pid]["name"]
                    for pid in car_to_pri.values()
                    if pid in self._pri_info
                }
                for player_name, boost_pct in current_boost.items():
                    if not active_players or player_name in active_players:
                        rows.append({
                            "frame": fi,
                            "time": time_val,
                            "player_name": player_name,
                            "team": player_teams.get(player_name, -1),
                            "boost_pct": boost_pct,
                        })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        return df.sort_values(["frame", "player_name"]).reset_index(drop=True)

    # ──────────────────────────────────────────────
    #  Etapa 4: Metadados e Eventos
    # ──────────────────────────────────────────────

    def extract_metadata(self) -> dict[str, Any]:
        """Extrai metadados do replay a partir de 'properties'."""
        props = self.data.get("properties", {})

        player_stats = []
        for ps in props.get("PlayerStats", []):
            player_stats.append({
                "name": ps.get("Name", "?"),
                "team": ps.get("Team", -1),
                "score": ps.get("Score", 0),
                "goals": ps.get("Goals", 0),
                "assists": ps.get("Assists", 0),
                "saves": ps.get("Saves", 0),
                "shots": ps.get("Shots", 0),
            })

        return {
            "map_name": props.get("MapName", "Desconhecido"),
            "team_size": props.get("TeamSize", 0),
            "team_0_score": props.get("Team0Score", 0),
            "team_1_score": props.get("Team1Score", 0),
            "num_frames": props.get("NumFrames", len(self.frames)),
            "total_seconds": props.get("TotalSecondsPlayed", 0.0),
            "date": props.get("Date", ""),
            "replay_name": props.get("ReplayName", ""),
            "player_stats": player_stats,
        }

    def extract_goals(self) -> list[dict[str, Any]]:
        """Extrai lista de gols do replay."""
        props = self.data.get("properties", {})
        goals = []
        for g in props.get("Goals", []):
            goals.append({
                "frame": g.get("frame", 0),
                "player_name": g.get("PlayerName", "?"),
                "player_team": g.get("PlayerTeam", -1),
            })
        return goals

    def extract_demolitions(self) -> list[dict[str, Any]]:
        """Extrai demolições dos network_frames.

        Procura por atualizações com ReplicatedDemolish/ReplicatedDemolishExtended.
        """
        demolish_oids = self._find_object_ids_containing("ReplicatedDemolish")

        if not demolish_oids:
            return []

        demos: list[dict] = []
        for fi, frame in enumerate(self.frames):
            for actor in frame.get("updated_actors", []):
                oid = actor.get("object_id")
                if oid not in demolish_oids:
                    continue
                attr = actor.get("attribute", {})
                # A estrutura de demolish varia, extraímos o que pudermos
                demos.append({
                    "frame": fi,
                    "time": frame.get("time", 0.0),
                    "raw": attr,
                })

        return demos

    # ──────────────────────────────────────────────
    #  Método Completo
    # ──────────────────────────────────────────────

    def analyze(self) -> dict[str, Any]:
        """Roda toda a pipeline de análise e retorna resultados estruturados.

        Returns:
            Dict com chaves:
                - metadata: dict com informações gerais do replay
                - players: lista de jogadores com nome, time, car_id
                - positions: DataFrame com posições frame-a-frame
                - boost: DataFrame com boost timeline
                - goals: lista de gols
                - demolitions: lista de demolições
        """
        self.build_actor_registry()

        return {
            "metadata": self.extract_metadata(),
            "players": self.get_players(),
            "positions": self.extract_positions(),
            "boost": self.extract_boost_timeline(),
            "goals": self.extract_goals(),
            "demolitions": self.extract_demolitions(),
        }


# ──────────────────────────────────────────────
#  Funções auxiliares de geometria
# ──────────────────────────────────────────────

def compute_triangle_area(p1: tuple, p2: tuple, p3: tuple) -> float:
    """Calcula a área de um triângulo dados 3 pontos (x, y).

    Usa a fórmula do shoelace (produto vetorial).
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


def compute_distance(p1: tuple, p2: tuple) -> float:
    """Distância euclidiana entre 2 pontos (x, y)."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


if __name__ == "__main__":
    import json

    # Teste rápido com o replay de teste
    test_path = "data/processed/teste.json"

    print(f"Carregando {test_path}...")
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    analyzer = ReplayAnalyzer(data)
    result = analyzer.analyze()

    meta = result["metadata"]
    print(f"\n=== Metadados ===")
    print(f"  Mapa: {meta['map_name']}")
    print(f"  Placar: {meta['team_0_score']} x {meta['team_1_score']}")
    print(f"  Frames: {meta['num_frames']}")
    print(f"  Duracao: {meta['total_seconds']:.0f}s")

    print(f"\n=== Jogadores ===")
    for p in result["players"]:
        team_label = "Azul" if p["team"] == 0 else "Laranja" if p["team"] == 1 else "?"
        print(f"  {p['name']:20s} | {team_label}")

    pos = result["positions"]
    print(f"\n=== Posicoes ===")
    print(f"  Total de linhas: {len(pos)}")
    if not pos.empty:
        print(f"  Entidades unicas: {pos['entity'].unique().tolist()}")
        print(f"  Frames amostrados: {pos['frame'].nunique()}")
        print(f"\n  Amostra (primeiras 10 linhas):")
        print(pos.head(10).to_string(index=False))

    boost = result["boost"]
    print(f"\n=== Boost ===")
    print(f"  Total de linhas: {len(boost)}")
    if not boost.empty:
        print(f"  Jogadores: {boost['player_name'].unique().tolist()}")

    goals = result["goals"]
    print(f"\n=== Gols ({len(goals)}) ===")
    for g in goals:
        print(f"  Frame {g['frame']}: {g['player_name']} (Time {g['player_team']})")

    print(f"\n=== Demolicoes ({len(result['demolitions'])}) ===")
