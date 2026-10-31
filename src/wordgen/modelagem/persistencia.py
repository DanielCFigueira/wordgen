"""Módulo de persistência de modelos, pipelines e metadados."""

import json
from pathlib import Path

import joblib


def salvar_artefato_modelo(
    pipeline,
    nome_artefato: str,
    metadados: dict,
    models_dir: Path | str = "models/modelagem",
) -> Path:
    """Persiste o pipeline completo do Scikit-learn e os metadados JSON correspondentes."""
    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib_path = out_dir / f"{nome_artefato}.joblib"
    json_path = out_dir / f"{nome_artefato}_metadata.json"

    # Salva o pipeline Scikit-learn completo
    joblib.dump(pipeline, joblib_path)

    # Converte tipos numpy/pandas para tipos nativos do python para serialização JSON
    metadados_serializaveis = {}
    for k, v in metadados.items():
        if hasattr(v, "tolist"):
            metadados_serializaveis[k] = v.tolist()
        elif hasattr(v, "item"):
            metadados_serializaveis[k] = v.item()
        elif isinstance(v, dict):
            metadados_serializaveis[k] = {
                sub_k: sub_v.item() if hasattr(sub_v, "item") else sub_v
                for sub_k, sub_v in v.items()
            }
        else:
            metadados_serializaveis[k] = v

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadados_serializaveis, f, indent=2, ensure_ascii=False)

    return joblib_path


def carregar_artefato_modelo(
    nome_artefato: str,
    models_dir: Path | str = "models/modelagem",
):
    """Carrega o pipeline Scikit-learn e os metadados salvos."""
    base_dir = Path(models_dir)
    joblib_path = base_dir / f"{nome_artefato}.joblib"
    json_path = base_dir / f"{nome_artefato}_metadata.json"

    if not joblib_path.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {joblib_path}")

    pipeline = joblib.load(joblib_path)
    metadados = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            metadados = json.load(f)

    return pipeline, metadados
