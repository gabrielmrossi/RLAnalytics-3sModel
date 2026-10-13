"""
tests/generate_baselines.py — Gera os baselines numéricos e estruturais
do projeto no estado atual (antes de qualquer refatoração da Fase 1).
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Adicionar src ao path para importar módulos existentes
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


def round_floats(obj, decimals=6):
    """Arredonda floats recursivamente para consistência no JSON."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, np.floating):
        return round(float(obj), decimals)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return round_floats(obj.tolist(), decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [round_floats(x, decimals) for x in obj]
    return obj


def compute_df_hash(df: pd.DataFrame) -> str:
    """Calcula hash determinístico do DataFrame."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def generate():
    fixtures_dir = ROOT_DIR / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT_DIR / "data" / "processed" / "features_dataset.csv"
    model_path = ROOT_DIR / "data" / "processed" / "model_detector.joblib"

    print("1. Carregando dataset...")
    df = load_dataset(str(csv_path))
    X, y, feature_names = get_X_y(df)
    k = min(3, len(X) - 1)

    baselines = {
        "dataset": {
            "shape": list(df.shape),
            "label_counts": df["label"].value_counts().to_dict(),
            "columns": list(df.columns),
            "feature_names": feature_names,
            "feature_means": df[feature_names].mean().to_dict(),
            "feature_stds": df[feature_names].std().to_dict(),
            "df_hash": compute_df_hash(df),
        }
    }

    print("2. Avaliando KNN...")
    knn = RankClassifierKNN(n_neighbors=k)
    knn_loo = knn.evaluate_loo(X, y)
    knn_ablation = knn.feature_importance_loo(X, y, feature_names)
    baselines["knn"] = {
        "accuracy": knn_loo["accuracy"],
        "confusion_matrix": knn_loo["confusion_matrix"],
        "predictions": knn_loo["predictions"],
        "feature_importance_ablation": knn_ablation,
    }

    print("3. Avaliando Decision Tree...")
    dt = RankClassifierTree(max_depth=3)
    dt_loo = dt.evaluate_loo(X, y)
    dt.train(X, y, feature_names)
    baselines["decision_tree"] = {
        "accuracy": dt_loo["accuracy"],
        "confusion_matrix": dt_loo["confusion_matrix"],
        "feature_importances": dt.get_feature_importances(),
        "rules": dt.get_rules(),
    }

    print("4. Avaliando Random Forest...")
    rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
    rf_loo = rf.evaluate_loo(X, y)
    rf.train(X, y, feature_names)
    baselines["random_forest"] = {
        "accuracy": rf_loo["accuracy"],
        "confusion_matrix": rf_loo["confusion_matrix"],
        "feature_importances": rf.get_feature_importances(),
    }

    print("5. Avaliando ModelBenchmarking (cv=5)...")
    bench = ModelBenchmarking(cv=5, random_state=42)
    bench_df = bench.evaluate(X, y, knn_k=k)
    baselines["benchmarking"] = {
        "summary_table": bench_df.to_dict(orient="records"),
        "confusion_matrices": {
            m: cm.tolist() for m, cm in bench.confusion_matrices_.items()
        },
    }

    print("6. Avaliando TacticalClusterKMeans...")
    clusterer = TacticalClusterKMeans(n_clusters=3, random_state=42)
    # Elbow inertias
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    elbow_inertias = []
    for k_val in range(2, 9):
        km = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
        km.fit(X_scaled)
        elbow_inertias.append(float(km.inertia_))

    labels = clusterer.fit_predict(X)
    cluster_summary = clusterer.cluster_summary(df, labels, feature_names)
    baselines["kmeans"] = {
        "elbow_inertias_k2_to_k8": elbow_inertias,
        "labels": labels.tolist(),
        "silhouette": clusterer.silhouette_,
        "cluster_summary": cluster_summary.to_dict(),
    }

    print("7. Avaliando Detectores e Narrativas...")
    # PositioningErrorDetector (baseline legado)
    ped = PositioningErrorDetector()
    ped.build_reference(df)
    ped_errors_per_row = []
    for _, row in df.iterrows():
        fdict = {col: row[col] for col in feature_names}
        errs = ped.detect_errors(fdict)
        ped_errors_per_row.append(errs)

    # ModelDrivenErrorDetector
    if os.path.exists(model_path):
        mded = ModelDrivenErrorDetector.load(str(model_path))
    else:
        mded = ModelDrivenErrorDetector(n_estimators=100, random_state=42)
        mded.build(df, X, y, feature_names)

    mded_preds = mded.predict(X).tolist()
    mded_errors_per_row = []
    narratives_per_row = []

    for _, row in df.iterrows():
        fdict = {col: row[col] for col in feature_names}
        errs = mded.detect_errors(fdict)
        mded_errors_per_row.append(errs)

        narrative, score = _generate_tactical_narrative(fdict, mded)
        narratives_per_row.append({"narrative": narrative, "score": score})

    baselines["positioning_detector"] = {
        "reference_mean": ped.reference_mean,
        "reference_std": ped.reference_std,
        "errors_count": [len(e) for e in ped_errors_per_row],
        "sample_errors": ped_errors_per_row[:10],
    }

    baselines["model_driven_detector"] = {
        "predictions": mded_preds,
        "reference_mean": mded.reference_mean,
        "reference_std": mded.reference_std,
        "feature_importances": mded.feature_importances,
        "errors_count": [len(e) for e in mded_errors_per_row],
        "sample_errors": mded_errors_per_row[:10],
    }

    baselines["tactical_narratives"] = {
        "scores": [n["score"] for n in narratives_per_row],
        "sample_narratives": narratives_per_row[:10],
    }

    print("8. Snapshot do ReplayAnalyzer em 2 replays...")
    sample_replays = [
        "03515d59-b3fb-4c81-a9bd-6e3621928694",
        "040a5cfa-c8c9-4d93-8f10-15dd7a290f8e",
    ]
    baselines["analyzer_snapshots"] = {}

    for rep in sample_replays:
        json_file = ROOT_DIR / "data" / "processed" / f"{rep}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            analyzer = ReplayAnalyzer(raw_data)
            res = analyzer.analyze()
            pos_df = res["positions"]
            boost_df = res["boost"]

            stagnation_0 = _check_stagnation_anomaly(pos_df, 0)
            stagnation_1 = _check_stagnation_anomaly(pos_df, 1)

            baselines["analyzer_snapshots"][rep] = {
                "metadata": res["metadata"],
                "player_names": [p["name"] for p in res["players"]],
                "positions_shape": list(pos_df.shape),
                "positions_hash": compute_df_hash(pos_df) if not pos_df.empty else None,
                "boost_shape": list(boost_df.shape),
                "boost_hash": compute_df_hash(boost_df) if not boost_df.empty else None,
                "goals_count": len(res["goals"]),
                "demolitions_count": len(res["demolitions"]),
                "stagnation_alerts_team0": stagnation_0,
                "stagnation_alerts_team1": stagnation_1,
            }

    # Salvar em tests/fixtures/baselines.json
    out_path = fixtures_dir / "baselines.json"
    cleaned_baselines = round_floats(baselines)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_baselines, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Baselines gerados com sucesso e salvos em: {out_path}")
    print(f"Tamanho do baseline JSON: {os.path.getsize(out_path):,} bytes")


if __name__ == "__main__":
    generate()
