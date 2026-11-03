"""
tests/test_api_contract.py — Testes de contrato REST v1, ciclo de vida e carga (Fase 3.6).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parent.parent


class TestAnalyticsAPIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp_root = Path(tempfile.mkdtemp(prefix="rl_api_test_"))
        cls.data_dir = cls.tmp_root / "data"
        cls.parquet_dir = cls.data_dir / "parquet"
        cls.tmp_dir = cls.tmp_root / "tmp"
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        cls.parquet_dir.mkdir(parents=True, exist_ok=True)
        cls.tmp_dir.mkdir(parents=True, exist_ok=True)

        os.environ["ANALYTICS_DATABASE_URL"] = "sqlite://"
        os.environ["ANALYTICS_QUEUE_MODE"] = "sync"
        os.environ["DATA_DIR"] = str(cls.data_dir)
        os.environ["PARQUET_DIR"] = str(cls.parquet_dir)
        os.environ["TMP_DIR"] = str(cls.tmp_dir)
        os.environ["AUTO_SEED_FROM_CSV"] = "true"
        os.environ["AUTO_SEED_FIXTURE_REPLAYS"] = "true"

        from rl_analytics_api import database
        from rl_analytics_api.main import app

        database.configure_database("sqlite://")
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)
        shutil.rmtree(cls.tmp_root, ignore_errors=True)

    def test_01_health_and_openapi(self):
        r_live = self.client.get("/api/analytics/v1/health/live")
        self.assertEqual(r_live.status_code, 200)
        self.assertEqual(r_live.json()["status"], "ok")

        r_ready = self.client.get("/api/analytics/v1/health/ready")
        self.assertEqual(r_ready.status_code, 200)
        body = r_ready.json()
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["database"], "ok")

        r_oas = self.client.get("/api/analytics/openapi.json")
        self.assertEqual(r_oas.status_code, 200)
        oas = r_oas.json()
        self.assertIn("/api/analytics/v1/replays", oas["paths"])
        self.assertIn("/api/analytics/v1/replays/{replay_id}/diagnosis", oas["paths"])
        self.assertIn("/api/analytics/v1/replays/{replay_id}/heatmap", oas["paths"])
        self.assertIn("/api/analytics/v1/replays/{replay_id}/radar", oas["paths"])
        self.assertIn("/api/analytics/v1/replays/{replay_id}/report", oas["paths"])

    def test_02_model_endpoint(self):
        r = self.client.get("/api/analytics/v1/model")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["feature_names"]), 10)
        self.assertTrue(data["model"]["version"].startswith("rf-"))
        self.assertIn("ball_chase_ratio", data["feature_importances"])

    def test_03_catalog_pagination_and_filters(self):
        r_all = self.client.get("/api/analytics/v1/replays?limit=10")
        self.assertEqual(r_all.status_code, 200)
        body = r_all.json()
        self.assertGreaterEqual(body["total"], 100)
        self.assertEqual(len(body["items"]), 10)
        self.assertIsNotNone(body["next_cursor"])
        # Os replays das fixtures têm telemetria espacial e vêm primeiro
        self.assertTrue(body["items"][0]["has_spatial_telemetry"])

        r_high = self.client.get("/api/analytics/v1/replays?label=higher&limit=5")
        self.assertEqual(r_high.status_code, 200)
        for item in r_high.json()["items"]:
            self.assertEqual(item["catalog_label"], "higher")

    def test_04_diagnosis_radar_heatmap_and_etag(self):
        # Usar o snapshot 03515d59-b3fb-4c81-a9bd-6e3621928694 já populado no startup
        stem = "03515d59-b3fb-4c81-a9bd-6e3621928694"

        r_diag = self.client.get(f"/api/analytics/v1/replays/{stem}/diagnosis")
        self.assertEqual(r_diag.status_code, 200)
        diag = r_diag.json()
        self.assertTrue(diag["is_training_sample"])
        self.assertEqual(len(diag["teams"]), 2)
        for t in diag["teams"]:
            self.assertGreaterEqual(t["score"], 0.0)
            self.assertLessEqual(t["score"], 100.0)
            self.assertNotIn("<strong>", t["summary"])
            self.assertEqual(len(t["findings"]), 10)
            self.assertIn("higher", t["prediction"]["proba"])

        # Radar (array plano direction-aware)
        r_radar = self.client.get(f"/api/analytics/v1/replays/{stem}/radar")
        self.assertEqual(r_radar.status_code, 200)
        radar = r_radar.json()
        self.assertEqual(len(radar["series"]), 4)
        self.assertEqual(len(radar["data"]), 10)
        first_pt = radar["data"][0]
        for key in ("feature", "axis", "direction", "team_0", "team_1", "ref_higher", "ref_lower"):
            self.assertIn(key, first_pt)

        # Heatmap padrão (cells)
        r_hm = self.client.get(f"/api/analytics/v1/replays/{stem}/heatmap")
        self.assertEqual(r_hm.status_code, 200)
        hm = r_hm.json()
        self.assertEqual(hm["grid"]["nx"], 24)
        self.assertEqual(hm["grid"]["ny"], 32)
        self.assertGreater(len(hm["cells"]), 100)

        # Heatmap customizado via Parquet sob demanda (grid + ball)
        r_hm_grid = self.client.get(
            f"/api/analytics/v1/replays/{stem}/heatmap?bins=16x20&entity=ball&format=grid"
        )
        self.assertEqual(r_hm_grid.status_code, 200)
        hm_grid = r_hm_grid.json()
        self.assertEqual(len(hm_grid["teams"]["0"]["counts"]), 16 * 20)

        # Report agregado + ETag 304
        r_rep = self.client.get(f"/api/analytics/v1/replays/{stem}/report")
        self.assertEqual(r_rep.status_code, 200)
        etag = r_rep.headers.get("etag")
        self.assertIsNotNone(etag)

        r_304 = self.client.get(
            f"/api/analytics/v1/replays/{stem}/report",
            headers={"If-None-Match": etag},
        )
        self.assertEqual(r_304.status_code, 304)

    def test_05_upload_validation_and_deduplication(self):
        # 1. Extensão inválida -> 422 problem+json
        r_bad_ext = self.client.post(
            "/api/analytics/v1/replays",
            files={"file": ("bad.txt", b"hello world", "text/plain")},
        )
        self.assertEqual(r_bad_ext.status_code, 422)
        self.assertIn("application/problem+json", r_bad_ext.headers["content-type"])
        self.assertEqual(r_bad_ext.json()["code"], "INVALID_REPLAY")

        # 2. Header binário inválido -> 422 problem+json
        r_bad_hdr = self.client.post(
            "/api/analytics/v1/replays",
            files={"file": ("corrupt.replay", b"NOT_A_ROCKET_LEAGUE_REPLAY_AT_ALL", "application/octet-stream")},
        )
        self.assertEqual(r_bad_hdr.status_code, 422)
        self.assertEqual(r_bad_hdr.json()["code"], "INVALID_REPLAY")

        # 3. Upload válido de um replay 3v3 real
        sample_replay = ROOT_DIR / "tests" / "fixtures" / "replays" / "040a5cfa-c8c9-4d93-8f10-15dd7a290f8e.replay"
        candidate_3v3 = ROOT_DIR / "data" / "raw" / "higher_rank_replays" / "052e8744-ac15-456e-b32f-ae5bba452904.replay"
        if candidate_3v3.exists():
            sample_replay = candidate_3v3

        payload_bytes = sample_replay.read_bytes()
        r_up1 = self.client.post(
            "/api/analytics/v1/replays",
            files={"file": (sample_replay.name, payload_bytes, "application/octet-stream")},
            headers={"Prefer": "wait=5"},
        )
        self.assertIn(r_up1.status_code, (200, 202), f"Resposta inesperada: {r_up1.text}")
        up1_body = r_up1.json()
        replay_id = up1_body["replay_id"]
        self.assertEqual(up1_body["job"]["status"], "done")

        # Verificar status do job
        if not up1_body["deduplicated"]:
            job_id = up1_body["job"]["id"]
            r_job = self.client.get(f"/api/analytics/v1/jobs/{job_id}")
            self.assertEqual(r_job.status_code, 200)
            self.assertEqual(r_job.json()["status"], "done")

        # Verificar que TMP_DIR está 100% limpo de JSONs brutos
        leftover = list(self.tmp_dir.glob("*.json"))
        self.assertEqual(len(leftover), 0, f"JSONs brutos vazaram: {leftover}")

        # 4. Segundo upload do mesmo arquivo -> deduplicated=True (200 OK imediato)
        r_up2 = self.client.post(
            "/api/analytics/v1/replays",
            files={"file": (sample_replay.name, payload_bytes, "application/octet-stream")},
        )
        self.assertEqual(r_up2.status_code, 200)
        self.assertTrue(r_up2.json()["deduplicated"])
        self.assertEqual(r_up2.json()["replay_id"], replay_id)

    def test_06_concurrent_reads_latency(self):
        stem = "03515d59-b3fb-4c81-a9bd-6e3621928694"

        def _fetch_report(_i: int) -> tuple[int, float]:
            t0 = time.perf_counter()
            resp = self.client.get(f"/api/analytics/v1/replays/{stem}/report")
            return resp.status_code, (time.perf_counter() - t0) * 1000.0

        with ThreadPoolExecutor(max_workers=10) as pool:
            results = list(pool.map(_fetch_report, range(20)))

        statuses = [s for s, _ in results]
        latencies = sorted(ms for _, ms in results)
        p95 = latencies[int(len(latencies) * 0.95) - 1]

        self.assertTrue(all(s == 200 for s in statuses), f"Status inesperados: {statuses}")
        self.assertLess(p95, 100.0, f"Latência p95 de leitura ({p95:.1f} ms) acima de 100 ms")


if __name__ == "__main__":
    unittest.main()
