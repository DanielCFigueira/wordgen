"""
Finalização dos Datasets — Projeto WordGen
===========================================
Gera os datasets finais anonimizados em data/final/:
  - dataset_vocabulario (.csv e .parquet)
  - dataset_tde (.csv e .parquet)
  - dim_aluno (.csv e .parquet)

Remove todas as colunas de dados pessoais dos datasets de avaliação.
Consolida metadados demográficos minimizados em dim_aluno.
"""
from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from wordgen.domain.padronizacao import (
    ESCOLAS_CANONICAS,
    faixa_etaria,
    normalizar_id,
    normalizar_sexo,
    padronizar_escola,
    padronizar_turma,
    parse_idade,
)

PROCESSED_DIR = BASE_DIR / "data" / "processed"
FINAL_DIR = BASE_DIR / "data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# Colunas proibidas nos datasets de avaliação
COLUNAS_PROIBIDAS = [
    "matricula",
    "responsavel_legal",
    "data_nascimento",
    "monitor",
    "coordenador",
    "idade",
    "sexo",
    "cor_raca",
    "zona",
    "necessidade_deficiencia",
]

CATEGORIAS_RACA_VALIDAS = {
    "PARDA",
    "BRANCA",
    "PRETA",
    "AMARELA",
    "INDÍGENA",
}


def resolver_moda_com_desempate_recente(pares_fase_valor, padrao="NAO INFORMADO"):
    """
    Calcula a moda dos valores válidos.
    Em caso de empate, escolhe o valor da fase mais recente.
    pares_fase_valor: lista de tuplas (fase, valor) ordenada por fase decrescente.
    """
    validos = [
        (f, v)
        for f, v in pares_fase_valor
        if pd.notna(v) and str(v).strip().upper() not in ("", "0", "0.0", padrao)
    ]
    if not validos:
        return padrao

    valores = [v for _, v in validos]
    contagens = Counter(valores)
    max_freq = max(contagens.values())
    candidatos = [v for v, c in contagens.items() if c == max_freq]

    if len(candidatos) == 1:
        return candidatos[0]

    # Em caso de empate, escolhe o da fase mais recente (primeira na lista ordenada desc)
    for _, v in validos:
        if v in candidatos:
            return v
    return candidatos[0]


def construir_dim_aluno(df_socio: pd.DataFrame) -> pd.DataFrame:
    """Consolida a tabela dim_aluno a partir do dataset socioeconômico."""
    registros = []

    for aluno_id, grupo in df_socio.groupby("aluno_id"):
        id_norm = normalizar_id(aluno_id)
        if not id_norm:
            continue

        grupo_sort = grupo.sort_values(by="fase", ascending=False)

        # 1. Sexo
        pares_sexo = [
            (f, normalizar_sexo(s))
            for f, s in zip(grupo_sort["fase"], grupo_sort["sexo"], strict=False)
        ]
        sexo_final = resolver_moda_com_desempate_recente(pares_sexo, "NAO INFORMADO")

        # 2. Cor/Raça
        def norm_raca(r):
            s = str(r).strip().upper()
            if s == "INDIGENA":
                s = "INDÍGENA"
            return s if s in CATEGORIAS_RACA_VALIDAS else "NAO INFORMADO"

        pares_raca = [
            (f, norm_raca(r))
            for f, r in zip(grupo_sort["fase"], grupo_sort["cor_raca"], strict=False)
        ]
        raca_final = resolver_moda_com_desempate_recente(pares_raca, "NAO INFORMADO")

        # 3. Zona
        def norm_zona(z):
            s = str(z).strip().upper()
            if s in ("URBANA", "ZONA URBANA"):
                return "URBANA"
            if s in ("RURAL", "ZONA RURAL"):
                return "RURAL"
            return "NAO INFORMADA"

        pares_zona = [
            (f, norm_zona(z))
            for f, z in zip(grupo_sort["fase"], grupo_sort["zona"], strict=False)
        ]
        zona_final = resolver_moda_com_desempate_recente(pares_zona, "NAO INFORMADA")

        # 4. Faixa Etária (fase mais recente com idade válida entre 8 e 20)
        faixa_final = "NAO INFORMADA"
        for _, row in grupo_sort.iterrows():
            id_val = parse_idade(row.get("idade"), row.get("data_nascimento"))
            if id_val is not None:
                faixa_final = faixa_etaria(id_val)
                break

        # 5. Necessidade Especial
        tem_nec = False
        for _, row in grupo_sort.iterrows():
            v = row.get("necessidade_deficiencia")
            if pd.notna(v) and str(v).strip() not in ("", "0", "0.0", "NÃO", "NAO", "N"):
                tem_nec = True
                break
        nec_final = "SIM" if tem_nec else "NAO_REGISTRADO"

        # 6. Fonte Socio (fases usadas)
        fases_presentes = sorted(grupo["fase"].unique())
        fonte_final = ", ".join(f"Fase {f}" for f in fases_presentes)

        registros.append({
            "aluno_id": id_norm,
            "sexo": sexo_final,
            "faixa_etaria": faixa_final,
            "cor_raca": raca_final,
            "zona": zona_final,
            "necessidade_especial": nec_final,
            "fonte_socio": fonte_final,
        })

    df_dim = pd.DataFrame(registros).sort_values(by="aluno_id")
    return df_dim


def finalizar_datasets_avaliacao(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas proibidas e valida identificadores do dataset de avaliação."""
    # Remove colunas proibidas
    cols_a_remover = [c for c in COLUNAS_PROIBIDAS if c in df.columns]
    df_limpo = df.drop(columns=cols_a_remover).copy()

    # Valida formato de aluno_id
    df_limpo["aluno_id"] = df_limpo["aluno_id"].apply(normalizar_id)
    df_limpo = df_limpo[df_limpo["aluno_id"].notna()].copy()

    # Ordenação determinística
    df_limpo = df_limpo.sort_values(by=["aluno_id", "fase", "periodo"])
    return df_limpo


def gerar_pareado(df: pd.DataFrame, total_col: str) -> pd.DataFrame:
    """Gera tabela pareada (pré e pós do mesmo aluno e fase) com n_validos > 0."""
    pre = df[df["periodo"] == "pre"].copy()
    pos = df[df["periodo"] == "pos"].copy()

    merged = pd.merge(
        pre,
        pos,
        on=["aluno_id", "fase"],
        suffixes=("_pre", "_pos"),
    )

    escola = merged["escola_pos"].fillna(merged["escola_pre"])
    serie = merged["serie_pos"].fillna(merged["serie_pre"])
    turma = merged["turma_pos"].fillna(merged["turma_pre"])

    df_par = pd.DataFrame({
        "aluno_id": merged["aluno_id"],
        "fase": merged["fase"],
        "escola": escola,
        "serie": serie,
        "turma": turma,
        "total_pre": merged[f"{total_col}_pre"],
        "total_pos": merged[f"{total_col}_pos"],
        "pct_pre": merged["pct_completude_pre"],
        "pct_pos": merged["pct_completude_pos"],
        "completo_pre": merged["completo_pre"].astype(bool),
        "completo_pos": merged["completo_pos"].astype(bool),
    })

    return df_par.sort_values(by=["aluno_id", "fase"])


def main():
    print("\n" + "=" * 60)
    print("  FINALIZAÇÃO E SANITIZAÇÃO DE PRIVACIDADE (FINALIZAR_DATASETS)")
    print("=" * 60)

    # 1. Carregar datasets processados
    fpath_vocab = PROCESSED_DIR / "dataset_vocabulario.csv"
    fpath_tde = PROCESSED_DIR / "dataset_tde.csv"
    fpath_socio = PROCESSED_DIR / "dataset_socioeconomico.csv"

    df_vocab_proc = pd.read_csv(fpath_vocab, low_memory=False)
    df_tde_proc = pd.read_csv(fpath_tde, low_memory=False)
    df_socio_proc = pd.read_csv(fpath_socio, low_memory=False)

    # 2. Finalizar datasets de avaliação
    df_vocab_final = finalizar_datasets_avaliacao(df_vocab_proc)
    df_tde_final = finalizar_datasets_avaliacao(df_tde_proc)

    # Verificar que nenhuma coluna proibida existe
    for col in COLUNAS_PROIBIDAS:
        assert col not in df_vocab_final.columns, f"Coluna proibida {col} em vocabulario!"
        assert col not in df_tde_final.columns, f"Coluna proibida {col} em TDE!"

    # 3. Construir dim_aluno
    df_dim_aluno = construir_dim_aluno(df_socio_proc)

    # 4. Validar critérios T6
    # Unicidade de chaves
    assert not df_vocab_final.duplicated(subset=["aluno_id", "fase", "periodo"]).any(), "Chave duplicada em vocabulario!"
    assert not df_tde_final.duplicated(subset=["aluno_id", "fase", "periodo"]).any(), "Chave duplicada em TDE!"
    assert not df_dim_aluno.duplicated(subset=["aluno_id"]).any(), "aluno_id duplicado em dim_aluno!"

    # Formato de aluno_id
    re_id = re.compile(r"^ALUNO_[0-9A-F]{16}$")
    assert df_vocab_final["aluno_id"].apply(lambda x: bool(re_id.match(str(x)))).all()
    assert df_tde_final["aluno_id"].apply(lambda x: bool(re_id.match(str(x)))).all()
    assert df_dim_aluno["aluno_id"].apply(lambda x: bool(re_id.match(str(x)))).all()

    # Cobertura Fase 4
    alunos_fase4 = set(df_vocab_final[df_vocab_final["fase"] == 4]["aluno_id"]) | set(df_tde_final[df_tde_final["fase"] == 4]["aluno_id"])
    alunos_dim = set(df_dim_aluno["aluno_id"])
    cobertura_f4 = len(alunos_fase4 & alunos_dim) / len(alunos_fase4)
    print(f"\n[COBERTURA] dim_aluno cobre {cobertura_f4:.2%} dos alunos da Fase 4 ({len(alunos_fase4 & alunos_dim)}/{len(alunos_fase4)})")
    assert cobertura_f4 >= 0.90, f"Cobertura da Fase 4 abaixo de 90%: {cobertura_f4:.2%}"

    # 5. Salvar saídas em CSV (utf-8-sig) e Parquet
    # Vocabulário
    df_vocab_final.to_csv(FINAL_DIR / "dataset_vocabulario.csv", index=False, encoding="utf-8-sig")
    df_vocab_final.to_parquet(FINAL_DIR / "dataset_vocabulario.parquet", index=False)
    print(f"[VOCABULÁRIO] Salvo em data/final/: {len(df_vocab_final)} linhas, {len(df_vocab_final.columns)} colunas")

    # TDE
    df_tde_final.to_csv(FINAL_DIR / "dataset_tde.csv", index=False, encoding="utf-8-sig")
    df_tde_final.to_parquet(FINAL_DIR / "dataset_tde.parquet", index=False)
    print(f"[TDE]         Salvo em data/final/: {len(df_tde_final)} linhas, {len(df_tde_final.columns)} colunas")

    # dim_aluno
    df_dim_aluno.to_csv(FINAL_DIR / "dim_aluno.csv", index=False, encoding="utf-8-sig")
    df_dim_aluno.to_parquet(FINAL_DIR / "dim_aluno.parquet", index=False)
    print(f"[DIM_ALUNO]   Salvo em data/final/: {len(df_dim_aluno)} alunos únicos")
    print(f"  Distribuição sexo: {dict(df_dim_aluno['sexo'].value_counts())}")
    print(f"  Distribuição faixa etária: {dict(df_dim_aluno['faixa_etaria'].value_counts())}")
    print(f"  Distribuição cor/raça: {dict(df_dim_aluno['cor_raca'].value_counts())}")
    print(f"  Distribuição zona: {dict(df_dim_aluno['zona'].value_counts())}")
    print(f"  Distribuição necessidade especial: {dict(df_dim_aluno['necessidade_especial'].value_counts())}")

    # 6. Gerar tabelas pareadas (T7)
    df_pareado_vocab = gerar_pareado(df_vocab_final, "total_vocabulario")
    df_pareado_tde = gerar_pareado(df_tde_final, "total_tde")

    df_pareado_vocab.to_parquet(FINAL_DIR / "pareado_vocabulario.parquet", index=False)
    df_pareado_vocab.to_csv(FINAL_DIR / "pareado_vocabulario.csv", index=False, encoding="utf-8-sig")
    df_pareado_tde.to_parquet(FINAL_DIR / "pareado_tde.parquet", index=False)
    df_pareado_tde.to_csv(FINAL_DIR / "pareado_tde.csv", index=False, encoding="utf-8-sig")

    print(f"\n[PAREADOS]    Tabelas pareadas salvas em data/final/:")
    print(f"  pareado_vocabulario: {len(df_pareado_vocab)} linhas")
    print(f"  pareado_tde        : {len(df_pareado_tde)} linhas")

    # Métricas de referência para relatório e testes
    metricas_pareadas = {
        "vocabulario": {},
        "tde": {},
    }

    print("\n--- Validação com referências T7 (apenas registros 'completo' em ambos os momentos) ---")
    vc = df_pareado_vocab[df_pareado_vocab["completo_pre"] & df_pareado_vocab["completo_pos"]]
    for f, g in vc.groupby("fase"):
        metricas_pareadas["vocabulario"][int(f)] = {
            "n": int(len(g)),
            "media_pre": round(float(g["total_pre"].mean()), 2),
            "media_pos": round(float(g["total_pos"].mean()), 2),
        }
        print(f"  Vocab Fase {f}: N={len(g)}, Média Pré={g['total_pre'].mean():.2f}, Média Pós={g['total_pos'].mean():.2f}")

    tc = df_pareado_tde[df_pareado_tde["completo_pre"] & df_pareado_tde["completo_pos"]]
    for f, g in tc.groupby("fase"):
        metricas_pareadas["tde"][int(f)] = {
            "n": int(len(g)),
            "media_pre": round(float(g["total_pre"].mean()), 2),
            "media_pos": round(float(g["total_pos"].mean()), 2),
        }
        print(f"  TDE Fase {f}  : N={len(g)}, Média Pré={g['total_pre'].mean():.2f}, Média Pós={g['total_pos'].mean():.2f}")

    import json
    with open(FINAL_DIR / "metricas_pareadas.json", "w", encoding="utf-8") as f_met:
        json.dump(metricas_pareadas, f_met, indent=2)

    print("\n" + "=" * 60)
    print("  FINALIZAÇÃO T6 E T7 CONCLUÍDA COM SUCESSO")
    print("=" * 60)


if __name__ == "__main__":
    main()
