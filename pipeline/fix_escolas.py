"""
Pós-processamento: normalizar nomes de escola nos datasets gerados.
"""
from pathlib import Path

import pandas as pd

OUT_DIR = str(Path(__file__).resolve().parents[1] / "data" / "processed")

# Mapeamento de normalização de escolas
ESCOLA_MAP = {
    "EMEB PROF. RICARDO VIEIRA DE LIMA":          "EMEB PROFESSOR RICARDO VIEIRA DE LIMA",
    "EMEM PROFESOR RICARDO VIEIRA DE LIMA":        "EMEB PROFESSOR RICARDO VIEIRA DE LIMA",
    "EMEB PROFESSOR RICARDO VIEIRA DE LIMA 9º ANO B": "EMEB PROFESSOR RICARDO VIEIRA DE LIMA",
    "EMEB PROFA. MARIA QUEIROZ FERRO":             "EMEB PROFESSORA MARIA QUEIROZ FERRO",
    "EMEB VER. EXPEDITO PORFÍRIO":                 "EMEB EXPEDITO PORFÍRIO DOS SANTOS",
    "1.0": None,  # valor lixo
    "0.0": None,  # valor lixo
}

for fname in ["dataset_vocabulario.csv", "dataset_tde.csv", "dataset_socioeconomico.csv"]:
    fpath = f"{OUT_DIR}/{fname}"
    df = pd.read_csv(fpath, low_memory=False)
    
    antes = df["escola"].nunique()
    df["escola"] = df["escola"].replace(ESCOLA_MAP)
    
    # Remover linhas com escola None (lixo)
    df = df[df["escola"].notna()]
    depois = df["escola"].nunique()
    
    df.to_csv(fpath, index=False, encoding="utf-8-sig")
    print(f"{fname}: {antes} escolas -> {depois} escolas ({len(df)} linhas)")

print("\nEscolas finais:")
df = pd.read_csv(f"{OUT_DIR}/dataset_vocabulario.csv", low_memory=False)
print(df["escola"].value_counts().to_string())
