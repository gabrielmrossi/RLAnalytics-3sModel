"""
tests/test_baselines.py — Testes de caracterização da Fase 1.
Garante que refatorações não alteram nenhum resultado numérico nem lógico.
"""

import hashlib
import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from analyzer import ReplayAnalyzer
from knn_model import (
    ModelBenchmarking,
    ModelDrivenErrorDetector,
    PositioningErrorDetector,
    RankClassifierKNN,
    RankClassifierRandomForest,
    RankClassifierTree,
    TacticalClusterKMeans,
    get_X_y,
    load_dataset,
)
from streamlit_app import _check_stagnation_anomaly, _generate_tactical_narrative


def compute_df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_file = ROOT_DIR / "tests" / "fixtures" / "baselines.json"
        if not cls.baseline_file.exists():
            raise FileNotFoundError(
                f"Arquivo de baseline não encontrado: {cls.baseline_file}. "
                "Execute tests/generate_baselines.py primeiro."
            )
        with open(cls.baseline_file, "r", encoding="utf-8") as f:
            cls.baselines = json.load(f)

        cls.csv_path = str(ROOT_DIR / "data" / "processed" / "features_dataset.csv")
        cls.df = load_dataset(cls.csv_path)
        cls.X, cls.y, cls.feature_names = get_X_y(cls.df)
        cls.k = min(3, len(cls.X) - 1)

    def test_01_dataset_integrity(self):
        b_data = self.baselines["dataset"]
        self.assertEqual(list(self.df.shape), b_data["shape"])
        self.assertEqual(list(self.df.columns), b_data["columns"])
        self.assertEqual(self.feature_names, b_data["feature_names"])
        self.assertEqual(compute_df_hash(self.df), b_data["df_hash"])

    def test_02_knn_metrics(self):
        b_knn = self.baselines["knn"]
        knn = RankClassifierKNN(n_neighbors=self.k)
        knn_loo = knn.evaluate_loo(self.X, self.y)
        self.assertAlmostEqual(knn_loo["accuracy"], b_knn["accuracy"], places=6)
        self.assertEqual(knn_loo["confusion_matrix"], b_knn["confusion_matrix"])

        ablation = knn.feature_importance_loo(self.X, self.y, self.feature_names)
        for feat, val in b_knn["feature_importance_ablation"].items():
            self.assertAlmostEqual(ablation[feat], val, places=4)

    def test_03_decision_tree_metrics(self):
        b_dt = self.baselines["decision_tree"]
        dt = RankClassifierTree(max_depth=3)
        dt_loo = dt.evaluate_loo(self.X, self.y)
        self.assertAlmostEqual(dt_loo["accuracy"], b_dt["accuracy"], places=6)
        self.assertEqual(dt_loo["confusion_matrix"], b_dt["confusion_matrix"])

        dt.train(self.X, self.y, self.feature_names)
        dt_imp = dt.get_feature_importances()
        for feat, val in b_dt["feature_importances"].items():
            self.assertAlmostEqual(dt_imp[feat], val, places=4)

    def test_04_random_forest_metrics(self):
        b_rf = self.baselines["random_forest"]
        rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
        rf_loo = rf.evaluate_loo(self.X, self.y)
        self.assertAlmostEqual(rf_loo["accuracy"], b_rf["accuracy"], places=6)
        self.assertEqual(rf_loo["confusion_matrix"], b_rf["confusion_matrix"])

        rf.train(self.X, self.y, self.feature_names)
        rf_imp = rf.get_feature_importances()
        for feat, val in b_rf["feature_importances"].items():
            self.assertAlmostEqual(rf_imp[feat], val, places=4)

    def test_05_benchmarking_metrics(self):
        b_bench = self.baselines["benchmarking"]
        bench = ModelBenchmarking(cv=5, random_state=42)
        bench_df = bench.evaluate(self.X, self.y, knn_k=self.k)
        curr_records = bench_df.to_dict(orient="records")
        self.assertEqual(curr_records, b_bench["summary_table"])

        for model_name, cm_list in b_bench["confusion_matrices"].items():
            curr_cm = bench.confusion_matrices_[model_name].tolist()
            np.testing.assert_allclose(curr_cm, cm_list, atol=1e-5)

    def test_06_kmeans_clustering(self):
        b_km = self.baselines["kmeans"]
        clusterer = TacticalClusterKMeans(n_clusters=3, random_state=42)
        labels = clusterer.fit_predict(self.X)
        self.assertEqual(labels.tolist(), b_km["labels"])
        if b_km["silhouette"] is not None:
            self.assertAlmostEqual(clusterer.silhouette_, b_km["silhouette"], places=5)

    def test_07_positioning_error_detector(self):
        b_ped = self.baselines["positioning_detector"]
        ped = PositioningErrorDetector()
        ped.build_reference(self.df)

        for col, mean_val in b_ped["reference_mean"].items():
            self.assertAlmostEqual(ped.reference_mean[col], mean_val, places=5)
            self.assertAlmostEqual(ped.reference_std[col], b_ped["reference_std"][col], places=5)

        counts = []
        for _, row in self.df.iterrows():
            fdict = {c: row[c] for c in self.feature_names}
            counts.append(len(ped.detect_errors(fdict)))
        self.assertEqual(counts, b_ped["errors_count"])

    def test_08_model_driven_detector_and_narratives(self):
        b_mded = self.baselines["model_driven_detector"]
        b_narr = self.baselines["tactical_narratives"]
        model_path = ROOT_DIR / "data" / "processed" / "model_detector.joblib"

        if model_path.exists():
            mded = ModelDrivenErrorDetector.load(str(model_path))
        else:
            mded = ModelDrivenErrorDetector(n_estimators=100, random_state=42)
            mded.build(self.df, self.X, self.y, self.feature_names)

        preds = mded.predict(self.X).tolist()
        self.assertEqual(preds, b_mded["predictions"])

        scores = []
        counts = []
        for _, row in self.df.iterrows():
            fdict = {c: row[c] for c in self.feature_names}
            errs = mded.detect_errors(fdict)
            counts.append(len(errs))

            narrative, score = _generate_tactical_narrative(fdict, mded)
            scores.append(score)

        self.assertEqual(counts, b_mded["errors_count"])
        self.assertEqual(scores, b_narr["scores"])

    def test_09_analyzer_snapshots(self):
        b_snapshots = self.baselines.get("analyzer_snapshots", {})
        for rep, b_snap in b_snapshots.items():
            json_file = ROOT_DIR / "data" / "processed" / f"{rep}.json"
            if not json_file.exists():
                continue
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            analyzer = ReplayAnalyzer(raw_data)
            res = analyzer.analyze()
            pos_df = res["positions"]
            boost_df = res["boost"]

            self.assertEqual(list(pos_df.shape), b_snap["positions_shape"])
            if b_snap["positions_hash"]:
                self.assertEqual(compute_df_hash(pos_df), b_snap["positions_hash"])

            self.assertEqual(list(boost_df.shape), b_snap["boost_shape"])
            if b_snap["boost_hash"]:
                self.assertEqual(compute_df_hash(boost_df), b_snap["boost_hash"])

            self.assertEqual(len(res["goals"]), b_snap["goals_count"])
            self.assertEqual(len(res["demolitions"]), b_snap["demolitions_count"])


if __name__ == "__main__":
    unittest.main()
