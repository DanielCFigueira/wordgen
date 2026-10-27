"""
Pós-processamento: normalizar nomes de escola nos datasets gerados.
Aplica padronização canônica, imputação por moda do mesmo aluno e 'NAO INFORMADA'.
Não remove registros.
"""
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from wordgen.domain.padronizacao import ESCOLAS_CANONICAS, padronizar_escola

OUT_DIR = BASE_DIR / "data" / "processed"

ARQUIVOS = ["dataset_vocabulario.csv", "dataset_tde.csv", "dataset_socioeconomico.csv"]


def main():
    print("\n" + "=" * 60)
    print("  PADRONIZAÇÃO DE ESCOLAS (FIX_ESCOLAS)")
    print("=" * 60)

    # 1. Carregar datasets e padronizar com as 6 escolas canônicas
    dfs = {}
    for fname in ARQUIVOS:
        fpath = OUT_DIR / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, low_memory=False)
        df["escola"] = df["escola"].apply(padronizar_escola)
        dfs[fname] = df

    # 2. Construir mapa de modas de escola por aluno_id combinando todos os datasets
    pares_escola = []
    for df in dfs.values():
        validos = df.loc[df["escola"].notna(), ["aluno_id", "escola"]]
        pares_escola.append(validos)

    mapa_modas = {}
    if pares_escola:
        todos = pd.concat(pares_escola, ignore_index=True)
        # Para cada aluno_id, calcula a moda das escolas conhecidas
        for aluno_id, grupo in todos.groupby("aluno_id"):
            m = grupo["escola"].mode()
            if not m.empty:
                mapa_modas[aluno_id] = m.iloc[0]

    # 3. Preencher nulos com a moda do aluno; restante vira NAO INFORMADA
    valores_permitidos = set(ESCOLAS_CANONICAS) | {"NAO INFORMADA"}

    for fname, df in dfs.items():
        n_antes = len(df)
        n_nulos = df["escola"].isna().sum()

        df["escola"] = df["escola"].fillna(df["aluno_id"].map(mapa_modas)).fillna("NAO INFORMADA")
        n_depois = len(df)

        assert n_antes == n_depois, f"Número de linhas alterado em {fname}: {n_antes} != {n_depois}"
        assert set(df["escola"].unique()).issubset(
            valores_permitidos
        ), f"Valores de escola inválidos em {fname}"

        # Ordenação determinística antes de salvar
        if "fase" in df.columns and "periodo" in df.columns:
            df = df.sort_values(by=["aluno_id", "fase", "periodo"])
        elif "fase" in df.columns:
            df = df.sort_values(by=["aluno_id", "fase"])

        fpath = OUT_DIR / fname
        df.to_csv(fpath, index=False, encoding="utf-8-sig")

        n_recup = n_nulos - (df["escola"] == "NAO INFORMADA").sum()
        print(f"{fname}: {n_antes} linhas preservadas | nulos: {n_nulos} -> recuperados: {n_recup}")
        print(f"  Escolas distintas: {df['escola'].nunique()}")

    print("\nEscolas no dataset_vocabulario:")
    df_v = pd.read_csv(OUT_DIR / "dataset_vocabulario.csv", low_memory=False)
    print(df_v["escola"].value_counts().to_string())
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
