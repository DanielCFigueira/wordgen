"""Exporta o catálogo para CSV (abre no Excel) e Parquet (para o pipeline)."""

from dataclasses import asdict
from enum import Enum
from pathlib import Path

import pandas as pd

from wordgen.domain.catalogo import RegistroCatalogo


def catalogo_para_dataframe(registros: list[RegistroCatalogo]) -> pd.DataFrame:
    linhas = [
        {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(r).items()}
        for r in registros
    ]
    df = pd.DataFrame(linhas)
    df["fase"] = df["fase"].astype("Int64")
    return df


def exportar_catalogo(registros: list[RegistroCatalogo], pasta: Path) -> pd.DataFrame:
    pasta.mkdir(parents=True, exist_ok=True)
    df = catalogo_para_dataframe(registros)
    df.to_csv(pasta / "catalogo.csv", sep=";", index=False, encoding="utf-8-sig")
    df.to_parquet(pasta / "catalogo.parquet", index=False)
    return df
