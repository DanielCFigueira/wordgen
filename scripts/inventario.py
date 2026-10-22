"""Gera o catálogo dos arquivos brutos em data/interim e imprime um resumo.

Uso:  uv run python scripts/inventario.py [pasta_raiz_dos_csvs]
"""

import sys
from pathlib import Path

from wordgen.application.inventario import construir_catalogo
from wordgen.infrastructure.exportacao import exportar_catalogo
from wordgen.infrastructure.leitura_csv import (
    ler_primeiras_linhas,
    listar_csvs,
    sha256_arquivo,
)

RAIZ_PADRAO = Path("data/raw/Dados Word Gen/saida_csv_anonimizada")
SAIDA = Path("data/interim")


def main() -> None:
    raiz = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ_PADRAO
    arquivos = listar_csvs(raiz)
    registros = construir_catalogo(arquivos, raiz, ler_primeiras_linhas, sha256_arquivo)
    df = exportar_catalogo(registros, SAIDA)

    print(f"Arquivos catalogados: {len(df)}")
    print("\nPor instrumento (decisão final):")
    print(df["instrumento"].value_counts().to_string())
    print("\nPor status:")
    print(df["status"].value_counts().to_string())
    print("\nVocabulário e TDE por fase e momento:")
    avaliacoes = df[df["instrumento"].isin(["vocabulario", "tde"])]
    print(
        avaliacoes.groupby(["instrumento", "fase", "momento"], dropna=False)
        .size()
        .rename("arquivos")
        .to_string()
    )
    print(f"\nCópias idênticas (mesmo hash): {df['duplicado_de'].notna().sum()}")
    print(f"Arquivos com itens repetidos no cabeçalho: {int(df['itens_repetidos'].sum())}")
    print(f"Catálogo gravado em {SAIDA.resolve()}")


if __name__ == "__main__":
    main()
