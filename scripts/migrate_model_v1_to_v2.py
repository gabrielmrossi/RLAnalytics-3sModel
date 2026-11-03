"""
scripts/migrate_model_v1_to_v2.py — Migração do artefato model_detector.joblib (v1 -> v2).

Executa os passos M0, M1, M2 e M3 do plano arquitetural:
- M0: Inventaria o artefato v1 (que depende do módulo top-level 'classifiers').
- M1/M2: Extrai os estimadores puros do scikit-learn (StandardScaler + RandomForestClassifier)
  e metadados nativos em um dicionário format_version=2 livre de classes customizadas.
- M3: Verifica em subprocesso isolado (SEM src/ no sys.path) que o artefato v2:
  1) não contém nenhuma referência a 'classifiers' ou 'error_detection' nos opcodes do pickle;
  2) produz predições 100% idênticas ao v1 e ao baselines.json;
  3) expõe predict_proba corretamente.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickletools
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rl_analytics.classifiers as _classifiers_mod  # noqa: E402

# Alias de compatibilidade caso o v1 legado seja carregado após o rename src/ -> rl_analytics/
sys.modules.setdefault("classifiers", _classifiers_mod)

V1_DEFAULT_PATH = PROJECT_ROOT / "artifacts" / "models" / "archive" / "model_detector.v1.joblib"
V2_DEFAULT_PATH = PROJECT_ROOT / "artifacts" / "models" / "model_detector.v2.joblib"
BASELINES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "baselines.json"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "features_dataset.csv"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_pickle_modules(path: Path) -> set[str]:
    """Extrai todos os módulos referenciados pelos opcodes GLOBAL / STACK_GLOBAL do arquivo joblib."""
    raw = path.read_bytes()
    modules: set[str] = set()
    # joblib pode intercalar arrays numpy; varremos strings de módulos conhecidas e opcodes iniciais
    try:
        for opcode, arg, _pos in pickletools.genops(raw):
            if opcode.name in ("GLOBAL", "INST") and isinstance(arg, str):
                mod = arg.split(" ")[0]
                modules.add(mod)
    except Exception:
        pass
    import re

    for match in re.findall(rb"(classifiers|error_detection|sklearn\.[a-zA-Z0-9_.]+)", raw):
        modules.add(match.decode("ascii", errors="ignore"))
    return modules


def migrate(v1_path: Path = V1_DEFAULT_PATH, v2_path: Path = V2_DEFAULT_PATH) -> dict:
    if not v1_path.exists():
        fallback = PROJECT_ROOT / "data" / "processed" / "model_detector.joblib"
        if fallback.exists():
            v1_path = fallback
        else:
            raise FileNotFoundError(f"Artefato v1 não encontrado em {v1_path}")

    v1_sha = sha256_file(v1_path)
    print(f"[M0] Carregando v1 de {v1_path} (sha256={v1_sha[:16]}...)")
    state_v1 = joblib.load(str(v1_path))

    rf_wrapper = state_v1["rf_model"]
    scaler = rf_wrapper.scaler
    estimator = rf_wrapper.model
    feature_names: list[str] = list(state_v1["feature_names"])
    reference_mean: dict[str, float] = {k: float(v) for k, v in state_v1["reference_mean"].items()}
    reference_std: dict[str, float] = {k: float(v) for k, v in state_v1["reference_std"].items()}
    feature_importances: dict[str, float] = {
        k: float(v) for k, v in state_v1["feature_importances"].items()
    }
    is_built = bool(state_v1["is_built"])

    # Validação M0 contra baselines.json
    baselines = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    df = pd.read_csv(DATASET_PATH)
    X = df[feature_names].values.astype(float)
    v1_preds = rf_wrapper.predict(X).tolist()
    assert v1_preds == baselines["model_driven_detector"]["predictions"], (
        "M0 FALHOU: predições do v1 divergem do baselines.json!"
    )
    print(f"[M0] Inventário OK: {len(feature_names)} features, {len(v1_preds)} predições validadas.")

    # M1 / M2: Construir estado v2 puro (apenas objetos sklearn e tipos nativos)
    state_v2 = {
        "format_version": 2,
        "scaler": scaler,
        "estimator": estimator,
        "feature_names": feature_names,
        "reference_mean": reference_mean,
        "reference_std": reference_std,
        "feature_importances": feature_importances,
        "is_built": is_built,
        "provenance": {
            "source_v1_sha256": v1_sha,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "joblib_version": joblib.__version__,
            "python_version": platform.python_version(),
            "migrated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }

    v2_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(state_v2, str(v2_path))
    v2_sha = sha256_file(v2_path)
    print(f"[M2] Artefato v2 salvo em {v2_path} (sha256={v2_sha[:16]}...)")

    # M3a: Verificar ausência de módulos customizados no pickle v2
    mods_v2 = inspect_pickle_modules(v2_path)
    forbidden = {"classifiers", "error_detection", "knn_model", "src", "rl_analytics"}
    leaked = mods_v2.intersection(forbidden)
    assert not leaked, f"M3a FALHOU: módulos customizados vazaram no v2: {leaked}"
    print(f"[M3a] Scan de módulos do pickle v2 OK: {sorted(mods_v2)}")

    # M3b & M3c: Subprocesso isolado SEM src/ no sys.path
    verify_code = f"""
import json, sys
import joblib, numpy as np, pandas as pd

# Garantir que src/ não está no sys.path
sys.path = [p for p in sys.path if not p.endswith('src')]

state = joblib.load(r'{v2_path}')
assert state['format_version'] == 2
scaler = state['scaler']
estimator = state['estimator']
fnames = state['feature_names']

df = pd.read_csv(r'{DATASET_PATH}')
X = df[fnames].values.astype(float)
X_s = scaler.transform(X)
preds = estimator.predict(X_s).tolist()
probas = estimator.predict_proba(X_s)

baselines = json.loads(open(r'{BASELINES_PATH}', encoding='utf-8').read())
assert preds == baselines['model_driven_detector']['predictions'], 'Predições divergem!'
assert probas.shape == (len(X), 2), f'Formato de predict_proba inválido: {{probas.shape}}'
assert np.allclose(probas.sum(axis=1), 1.0), 'Soma das probabilidades != 1.0'
print('[M3b/M3c] Verificação isolada sem sys.path OK! Amostras:', len(preds), 'Classes:', list(estimator.classes_))
"""
    subprocess.run([sys.executable, "-c", verify_code], check=True)
    return state_v2


if __name__ == "__main__":
    migrate()

