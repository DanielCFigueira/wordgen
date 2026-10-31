"""Módulo de preparação e estruturação das bases de dados para modelagem."""

from pathlib import Path

import pandas as pd


def carregar_bases_brutas(
    data_dir: Path | str = "data",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega as bases de dados consolidadas pareadas e a dimensão do aluno."""
    base_dir = Path(data_dir) / "final"

    # Prioriza parquet se disponível, caso contrário csv
    if (base_dir / "pareado_vocabulario.parquet").exists():
        pareado_vocab = pd.read_parquet(base_dir / "pareado_vocabulario.parquet")
    else:
        pareado_vocab = pd.read_csv(base_dir / "pareado_vocabulario.csv")

    if (base_dir / "pareado_tde.parquet").exists():
        pareado_tde = pd.read_parquet(base_dir / "pareado_tde.parquet")
    else:
        pareado_tde = pd.read_csv(base_dir / "pareado_tde.csv")

    if (base_dir / "dim_aluno.parquet").exists():
        dim_aluno = pd.read_parquet(base_dir / "dim_aluno.parquet")
    else:
        dim_aluno = pd.read_csv(base_dir / "dim_aluno.csv")

    return pareado_vocab, pareado_tde, dim_aluno


def preparar_base_instrumento(
    df_pareado: pd.DataFrame,
    df_dim_aluno: pd.DataFrame,
    instrumento: str,
    experimento: str = "A",
) -> pd.DataFrame:
    """Filtra pares completos, calcula o target e seleciona as features do experimento.

    Parâmetros:
    -----------
    df_pareado: DataFrame pareado com pré e pós-teste
    df_dim_aluno: DataFrame com atributos demográficos e contextuais
    instrumento: 'vocabulario' ou 'tde'
    experimento: 'A' (Acadêmico) ou 'B' (Acadêmico + Demográfico/Contextual)

    Retorna:
    --------
    DataFrame estruturado com aluno_id (grupo), target (ganho) e features permitidas.
    """
    # 1. Filtro estrito de completude (apenas pares completos no pré e pós)
    # Garante suporte a boolean nativo e string "True"
    comp_pre = df_pareado["completo_pre"].astype(str).str.upper() == "TRUE"
    comp_pos = df_pareado["completo_pos"].astype(str).str.upper() == "TRUE"
    df_valido = df_pareado[comp_pre & comp_pos].copy()

    # Validação dos limites das escalas
    limite_max = 50.0 if instrumento.lower() == "vocabulario" else 80.0
    val_tot_pre = (df_valido["total_pre"] >= 0) & (df_valido["total_pre"] <= limite_max)
    val_tot_pos = (df_valido["total_pos"] >= 0) & (df_valido["total_pos"] <= limite_max)
    df_valido = df_valido[val_tot_pre & val_tot_pos].copy()

    # 2. Criação do Target (Ganho)
    df_valido["ganho"] = df_valido["total_pos"] - df_valido["total_pre"]

    # 3. Tratamento e padronização das variáveis acadêmicas
    df_valido["fase"] = "Fase " + df_valido["fase"].astype(str).str.replace("Fase ", "")
    df_valido["escola"] = df_valido["escola"].fillna("NAO INFORMADA").astype(str).str.strip()
    df_valido["serie"] = (
        df_valido["serie"]
        .fillna(-1)
        .astype(float)
        .astype(int)
        .astype(str)
        .replace("-1", "NAO_INFORMADA")
    )
    df_valido["turma"] = df_valido["turma"].fillna("NAO_INFORMADA").astype(str).str.strip()

    # 4. Experimento B: Junção com dim_aluno (demográfico e contextual)
    if experimento.upper() == "B":
        df_valido = df_valido.merge(df_dim_aluno, on="aluno_id", how="left")

        # Preenchimento de ausências com categorias explícitas de não registro
        df_valido["sexo"] = df_valido["sexo"].fillna("NAO_INFORMADO").astype(str).str.strip()
        df_valido["faixa_etaria"] = (
            df_valido["faixa_etaria"].fillna("NAO INFORMADA").astype(str).str.strip()
        )
        df_valido["cor_raca"] = (
            df_valido["cor_raca"].fillna("NAO INFORMADO").astype(str).str.strip()
        )
        df_valido["zona"] = df_valido["zona"].fillna("NAO INFORMADA").astype(str).str.strip()
        df_valido["necessidade_especial"] = (
            df_valido["necessidade_especial"].fillna("NAO_REGISTRADO").astype(str).str.strip()
        )

    # 5. Seleção explícita de colunas e prevenção de Data Leakage
    cols_base = ["aluno_id", "fase", "escola", "serie", "turma", "total_pre", "ganho"]
    if experimento.upper() == "B":
        cols_features = cols_base + [
            "sexo",
            "faixa_etaria",
            "cor_raca",
            "zona",
            "necessidade_especial",
        ]
    else:
        cols_features = cols_base

    df_final = df_valido[cols_features].copy()
    df_final = df_final.reset_index(drop=True)
    return df_final


def carregar_e_preparar_dados(data_dir: Path | str = "data") -> dict[str, pd.DataFrame]:
    """Carrega e prepara os 4 datasets experimentais de modelagem.

    Retorna um dicionário com:
    - 'vocabulario_exp_a'
    - 'vocabulario_exp_b'
    - 'tde_exp_a'
    - 'tde_exp_b'
    """
    pareado_vocab, pareado_tde, dim_aluno = carregar_bases_brutas(data_dir)

    datasets = {
        "vocabulario_exp_a": preparar_base_instrumento(
            pareado_vocab, dim_aluno, "vocabulario", "A"
        ),
        "vocabulario_exp_b": preparar_base_instrumento(
            pareado_vocab, dim_aluno, "vocabulario", "B"
        ),
        "tde_exp_a": preparar_base_instrumento(pareado_tde, dim_aluno, "tde", "A"),
        "tde_exp_b": preparar_base_instrumento(pareado_tde, dim_aluno, "tde", "B"),
    }

    # Salva versões preparadas em data/modelagem
    out_dir = Path(data_dir) / "modelagem"
    out_dir.mkdir(parents=True, exist_ok=True)
    for nome, df in datasets.items():
        df.to_parquet(out_dir / f"{nome}.parquet", index=False)
        df.to_csv(out_dir / f"{nome}.csv", index=False)

    return datasets
