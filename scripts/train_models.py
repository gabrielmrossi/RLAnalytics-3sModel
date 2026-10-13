"""
scripts/train_models.py — CLI para treinamento, validação e persistência do modelo detector.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATASET_CSV_PATH, MODEL_PATH
from knn_model import (
    ModelDrivenErrorDetector,
    RankClassifierKNN,
    RankClassifierRandomForest,
    RankClassifierTree,
    get_X_y,
    load_dataset,
)


def main():
    csv_path = Path(DATASET_CSV_PATH)
    if not csv_path.exists():
        print(f"[!] Erro: Dataset não encontrado em {csv_path}. Execute 'rl-build-dataset' primeiro.")
        sys.exit(1)

    print(f"[*] Carregando dataset de {csv_path}...")
    df = load_dataset(str(csv_path))
    X, y, feature_names = get_X_y(df)
    k = min(3, len(X) - 1)

    print(f"[*] Total de amostras: {len(X)} | Features: {len(feature_names)}")

    # 1. Avalia KNN
    knn = RankClassifierKNN(n_neighbors=k)
    knn_res = knn.evaluate_loo(X, y)
    print(f"  [KNN] Acurácia (LOO): {knn_res['accuracy']:.1%}")

    # 2. Avalia Decision Tree
    tree = RankClassifierTree(max_depth=3)
    tree_res = tree.evaluate_loo(X, y)
    print(f"  [Decision Tree] Acurácia (LOO): {tree_res['accuracy']:.1%}")

    # 3. Avalia e treina Random Forest
    rf = RankClassifierRandomForest(n_estimators=100, random_state=42)
    rf_res = rf.evaluate_loo(X, y)
    print(f"  [Random Forest] Acurácia (LOO): {rf_res['accuracy']:.1%}")

    # 4. Constrói e persiste o ModelDrivenErrorDetector
    print(f"[*] Treinando ModelDrivenErrorDetector e salvando em {MODEL_PATH}...")
    detector = ModelDrivenErrorDetector(n_estimators=100, random_state=42)
    detector.build(df, X, y, feature_names)
    detector.save(str(MODEL_PATH))
    print("[+] Modelos treinados e detector persistido com sucesso!")


if __name__ == "__main__":
    main()
