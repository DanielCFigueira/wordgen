"""Carregamento dos datasets finais e artefatos agregados do WordGen."""

import json
import os
from pathlib import Path

import pandas as pd

from wordgen.dashboard.analise import exigir_colunas, normalizar_booleano

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("WORDGEN_DATA_DIR", ROOT / "data")).expanduser()
MODEL_DIR = Path(os.getenv("WORDGEN_MODEL_DIR", ROOT / "models" / "modelagem")).expanduser()
REPORTS_DIR = Path(os.getenv("WORDGEN_REPORTS_DIR", ROOT / "reports")).expanduser()
FINAL_DIR = DATA_DIR / "final"
ANALYTICS_DIR = DATA_DIR / "analytics"
METRICS_FILE = REPORTS_DIR / "modelagem" / "metricas" / "metricas_comparativas_gerais.csv"

_COLUNAS_AVALIACAO = {
    "aluno_id",
    "escola",
    "serie",
    "turma",
    "fase",
    "periodo",
    "completo",
    "pct_completude",
}
_COLUNAS_PAREADAS = {
    "aluno_id",
    "fase",
    "escola",
    "serie",
    "turma",
    "total_pre",
    "total_pos",
    "completo_pre",
    "completo_pos",
}
_COLUNAS_DIMENSAO = {
    "aluno_id",
    "sexo",
    "faixa_etaria",
    "cor_raca",
    "zona",
    "necessidade_especial",
}
_COLUNAS_METRICAS = {"Cenário", "Modelo", "MAE", "RMSE", "R²"}
_ARQUIVOS_ANALITICOS = {
    "kpis": "dashboard_kpis.csv",
    "cubo_filtros": "dashboard_cubo_filtros.csv",
    "effect_size": "metricas_effect_size.csv",
    "longitudinais": "metricas_longitudinais.csv",
    "painel_longitudinal": "painel_longitudinal.csv",
    "perfil_contextual": "perfil_socioeconomico_desempenho.csv",
}


def _ler_csv_obrigatorio(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset necessário não encontrado: {path.name}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _ler_csv_analitico(nome: str) -> pd.DataFrame:
    """Lê artefato analítico já calculado, sem refazer análises no dashboard."""
    path = ANALYTICS_DIR / _ARQUIVOS_ANALITICOS[nome]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _validar_avaliacoes(dados: pd.DataFrame, nome: str, coluna_total: str, maximo: int) -> None:
    exigir_colunas(dados, [*_COLUNAS_AVALIACAO, coluna_total], nome)
    if dados["aluno_id"].isna().any():
        raise ValueError(f"{nome} contém aluno_id nulo.")
    totais = pd.to_numeric(dados[coluna_total], errors="coerce")
    invalidos = dados[coluna_total].notna() & totais.isna()
    fora_da_escala = totais.notna() & ~totais.between(0, maximo)
    if invalidos.any() or fora_da_escala.any():
        raise ValueError(f"{nome} contém pontuações inválidas para a escala do instrumento.")


def _validar_pares(dados: pd.DataFrame, nome: str, maximo: int) -> None:
    """Valida o contrato de uma linha por aluno e fase do pipeline final."""
    exigir_colunas(dados, _COLUNAS_PAREADAS, nome)
    chave = ["aluno_id", "fase"]
    if dados[chave].isna().any().any():
        raise ValueError(f"{nome} contém aluno_id ou fase nulo na chave pareada.")
    if dados.duplicated(subset=chave).any():
        raise ValueError(f"{nome} contém chave pareada duplicada: aluno_id e fase.")
    completos_pre = normalizar_booleano(dados["completo_pre"])
    completos_pos = normalizar_booleano(dados["completo_pos"])
    if completos_pre.isna().any() or completos_pos.isna().any():
        raise ValueError(f"{nome} contém indicador de completude inválido ou ausente.")
    for coluna in ["total_pre", "total_pos"]:
        valores = pd.to_numeric(dados[coluna], errors="coerce")
        if (dados[coluna].notna() & valores.isna()).any() or (
            valores.notna() & ~valores.between(0, maximo)
        ).any():
            raise ValueError(f"{nome} contém {coluna} inválido para a escala do instrumento.")
    pares_completos = completos_pre.eq(True) & completos_pos.eq(True)
    if dados.loc[pares_completos, ["total_pre", "total_pos"]].isna().any().any():
        raise ValueError(f"{nome} contém par completo sem total pré ou pós-teste.")


def _validar_dimensao(dados: pd.DataFrame) -> None:
    exigir_colunas(dados, _COLUNAS_DIMENSAO, "dim_aluno")
    if dados["aluno_id"].isna().any() or dados["aluno_id"].duplicated().any():
        raise ValueError("dim_aluno deve conter exatamente um registro não nulo por aluno_id.")


def carregar_dados() -> dict[str, pd.DataFrame | dict | str]:
    """Lê uma vez os datasets publicados e mantém apenas colunas utilizáveis na análise."""
    vocabulary = _ler_csv_obrigatorio(FINAL_DIR / "dataset_vocabulario.csv")
    tde = _ler_csv_obrigatorio(FINAL_DIR / "dataset_tde.csv")
    paired_vocabulary = _ler_csv_obrigatorio(FINAL_DIR / "pareado_vocabulario.csv")
    paired_tde = _ler_csv_obrigatorio(FINAL_DIR / "pareado_tde.csv")
    students = _ler_csv_obrigatorio(FINAL_DIR / "dim_aluno.csv")

    _validar_avaliacoes(vocabulary, "dataset_vocabulario", "total_vocabulario", 50)
    _validar_avaliacoes(tde, "dataset_tde", "total_tde", 80)
    _validar_pares(paired_vocabulary, "pareado_vocabulario", 50)
    _validar_pares(paired_tde, "pareado_tde", 80)
    _validar_dimensao(students)

    # Identificadores e trilhas técnicas não entram em tabelas nem visualizações.
    for frame in [vocabulary, tde, paired_vocabulary, paired_tde, students]:
        frame.drop(
            columns=["arquivo_id", "linha_origem", "fonte_socio"], errors="ignore", inplace=True
        )

    metrics = pd.DataFrame()
    if METRICS_FILE.exists():
        metrics = pd.read_csv(METRICS_FILE, encoding="utf-8-sig")
        exigir_colunas(metrics, _COLUNAS_METRICAS, "O arquivo de métricas da modelagem")

    metadata: dict[str, dict] = {}
    if MODEL_DIR.exists():
        for path in MODEL_DIR.glob("*_metadata.json"):
            metadata[path.stem.removesuffix("_metadata")] = json.loads(
                path.read_text(encoding="utf-8")
            )

    quality_path = FINAL_DIR / "relatorio_qualidade.md"
    quality = quality_path.read_text(encoding="utf-8") if quality_path.exists() else ""
    analytics = {nome: _ler_csv_analitico(nome) for nome in _ARQUIVOS_ANALITICOS}
    return {
        "vocabulario": vocabulary,
        "tde": tde,
        "pareado_vocabulario": paired_vocabulary,
        "pareado_tde": paired_tde,
        "dim_aluno": students,
        "metricas": metrics,
        "metadados_modelos": metadata,
        "relatorio_qualidade": quality,
        **analytics,
    }
