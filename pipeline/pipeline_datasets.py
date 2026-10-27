"""
Pipeline de Engenharia de Dados — Projeto WordGen (IA)
=======================================================
Gera dois datasets consolidados:
  - data/processed/dataset_vocabulario.csv
  - data/processed/dataset_tde.csv

E um dataset auxiliar de dados socioeconômicos:
  - data/processed/dataset_socioeconomico.csv

Regras:
  - Fases 2, 3, 4
  - Apenas arquivos dentro de pastas "Vocabulário" e "TDE" (inclui raiz da Fase4)
  - Exclui: STARI, _dfs, _controle_aplicacao, _lista_de_alunos, _todos_alunos,
            arquivos de turma isolada (_6a, _6b, _7a...), _summary, _dashboard, _copy
  - Pré e Pós são detectados pela localização da pasta OU pelo nome do arquivo
  - Fase 4: sem subpastas — "PRE" no nome = pré-teste; prefixo "2 " = pós-teste
"""

import functools
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Adiciona src ao path para importar wordgen.domain
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from wordgen.domain.padronizacao import classificar_celula

# Contadores de flags de limpeza por instrumento
CONTADORES_FLAGS = {
    "vocabulario": Counter(),
    "tde": Counter(),
}

REVERSE_FILES = os.getenv("WORDGEN_REVERSE_FILES", "0").lower() in ("1", "true")


@functools.lru_cache(maxsize=None)
def calcular_arquivo_id(filepath):
    """Calcula os 12 primeiros caracteres do SHA-256 do arquivo (sem nomes de pessoas)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as fp:
        while chunk := fp.read(65536):
            h.update(chunk)
    return h.hexdigest()[:12]


def listar_csvs(diretorio):
    """Retorna lista de CSVs ordenada determinística, permitindo inversão para teste."""
    return sorted(diretorio.glob("*.csv"), reverse=REVERSE_FILES)

# ─────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR  = BASE_DIR / "data" / "raw" / "Dados Word Gen" / "saida_csv_anonimizada"
OUT_DIR  = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Padrões de exclusão no nome do arquivo (lowercase)
EXCLUIR_PATTERNS = [
    "stari", "_dfs", "controle_aplicacao", "lista_de_alunos",
    "todos_alunos", "todos_adilson",
    # Excluir apenas os arquivos de summary/dashboard/copy que NÃO têm dados de aluno
    "__summary", "__dashboard", "__copy_of", "copy_of_summary",
    # turmas isoladas: _6a, _6b, ..., _9c etc.
    r"_\d[a-z]\.csv$",
]

# Colunas de metadados socioeconômicos do arquivo Fase2 (RelacaoCompleta)
SOCIO_COLS_PADRAO = [
    "Sexo", "Data Nascimento", "Idade", "Cor/Raça", "Zona",
    "Necessidade/Deficiência", "Responsável Legal",
]

# ─────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────

def normalizar_turma(t):
    """'6º ANO C' / '6° ANO C' → '6C'"""
    if pd.isna(t):
        return t
    t = str(t).strip()
    m = re.search(r"(\d+)[°º]?\s*ANO\s+([A-Z])", t, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2).upper()}"
    return t.strip()

def normalizar_escola(e):
    if pd.isna(e):
        return e
    return str(e).strip().upper()

def normalizar_aluno_id(n):
    if pd.isna(n):
        return None
    n = str(n).strip()
    if n.startswith("ALUNO_"):
        return n
    return None

def converter_matricula(m):
    """'2.4022841194E10' → '24022841194'"""
    try:
        return str(int(float(str(m))))
    except Exception:
        return str(m).strip()

def deve_excluir(filename_lower):
    """Retorna True se o arquivo deve ser excluído."""
    for pat in EXCLUIR_PATTERNS:
        if re.search(pat, filename_lower):
            return True
    return False

def detectar_periodo_fase2(folder_path):
    """Detecta período pelo caminho da pasta (Fase2)."""
    p = str(folder_path).lower()
    if "pós-teste" in p or "pos-teste" in p or "pós teste" in p:
        return "pos"
    if "pré-teste" in p or "pre-teste" in p or "pré teste" in p:
        return "pre"
    return "desconhecido"

def detectar_periodo_fase3(folder_path):
    """Detecta período pelo caminho da pasta (Fase3)."""
    p = str(folder_path).lower()
    if "pós-teste" in p or "pos-teste" in p:
        return "pos"
    if "pré-teste" in p or "pre-teste" in p:
        return "pre"
    return "desconhecido"

def detectar_periodo_fase4(filename):
    """Fase4: sem subpastas. PRE no nome = pré; prefixo '2 ' = pós.
    Retorna tupla (periodo, periodo_fonte).
    """
    fn = filename.lower()
    if "(pre)" in fn or "pré)" in fn or "pre)" in fn:
        return "pre", "marcador_pre"
    # arquivos que começam com "2 " são pós-teste
    if re.match(r"^2[\s_]", filename.strip()):
        return "pos", "prefixo_2"
    # outros sem prefixo "2 " e sem "(PRE)" — tratar como pré por default
    return "pre", "padrao_pre"

def extrair_monitor_coord(filename):
    """Extrai monitor e coordenador do nome do arquivo."""
    monitor = ""
    coord = ""
    m = re.search(r"monitor\s*\(([^)]+)\)", filename, re.IGNORECASE)
    if m:
        monitor = m.group(1).strip()
    c = re.search(r"coord[.\s]*([^_,(]+)", filename, re.IGNORECASE)
    if c:
        coord = c.group(1).strip()
    return monitor, coord

# ─────────────────────────────────────────────
# LEITURA GENÉRICA DE CSV
# ─────────────────────────────────────────────

def ler_csv(filepath):
    """Lê CSV com separador ';', preservando a linha original e ignorando linhas completamente vazias."""
    try:
        df = pd.read_csv(
            filepath, sep=";", encoding="utf-8-sig",
            dtype=str, keep_default_na=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            filepath, sep=";", encoding="latin-1",
            dtype=str, keep_default_na=False,
        )
    # Guarda linha de origem no arquivo original (1 é cabeçalho, então index 0 -> linha 2)
    df["_linha_origem"] = df.index + 2
    df = df.replace("", np.nan)
    cols_dados = [c for c in df.columns if c != "_linha_origem"]
    df = df.dropna(how="all", subset=cols_dados)
    return df

def contar_conflitos(df, subset_chave, colunas_itens):
    """Conta chaves que possuem 2+ registros com respostas válidas e itens divergentes."""
    df_validos = df[df["n_validos"] > 0]
    duplicados = df_validos[df_validos.duplicated(subset=subset_chave, keep=False)]
    if duplicados.empty:
        return 0

    cols_existentes = [c for c in colunas_itens if c in df.columns]
    conflitos = 0
    for _, grupo in duplicados.groupby(subset_chave):
        itens_df = grupo[cols_existentes]
        tem_divergencia = False
        for c in cols_existentes:
            valores_distintos = itens_df[c].dropna().unique()
            if len(valores_distintos) > 1:
                tem_divergencia = True
                break
        if tem_divergencia:
            conflitos += 1
    return conflitos

# ─────────────────────────────────────────────
# NORMALIZAÇÃO — VOCABULÁRIO
# ─────────────────────────────────────────────

def normalizar_vocabulario(df, fase, periodo, monitor="", coord="", arquivo_id="", periodo_fonte=""):
    """
    Retorna DataFrame normalizado para dataset vocabulário.
    Aceita os dois esquemas de colunas (Fase2 e Fase3/4).
    """
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # ── Identificar coluna de aluno ──
    aluno_col = None
    for c in ["Nome", "NOME DO ESTUDANTE", "NOME ESTUDANTE"]:
        if c in cols:
            aluno_col = c
            break
    if aluno_col is None:
        return None  # Arquivo não reconhecível

    # ── Identificar colunas de questão Q1..Q50 ──
    q_cols = [c for c in cols if re.match(r"^Q\d+$", c)]
    if not q_cols:
        return None

    # ── Escola e Turma ──
    escola_col = next((c for c in ["Escola", "ESCOLA"] if c in cols), None)
    turma_col  = next((c for c in ["Turma", "TURMA"] if c in cols), None)
    total_col  = next((c for c in ["Total"] if c in cols), None)

    # ── Colunas socioeconômicas (presentes na Fase2) ──
    sexo_col    = next((c for c in ["Sexo"] if c in cols), None)
    nasc_col    = next((c for c in ["Data Nascimento"] if c in cols), None)
    idade_col   = next((c for c in ["Idade"] if c in cols), None)
    raca_col    = next((c for c in ["Cor/Raça"] if c in cols), None)
    zona_col    = next((c for c in ["Zona"] if c in cols), None)
    nec_col     = next((c for c in ["Necessidade/Deficiência"] if c in cols), None)
    resp_col    = next((c for c in ["Responsável Legal"] if c in cols), None)

    # Monitor da planilha (Fase3/4 tem coluna MONITOR)
    monitor_col = next((c for c in ["MONITOR"] if c in cols), None)
    mat_col     = next((c for c in ["MATRÍCULA"] if c in cols), None)

    rows = []
    for _, row in df.iterrows():
        aluno_id = normalizar_aluno_id(row.get(aluno_col))
        if aluno_id is None:
            continue

        escola = normalizar_escola(row.get(escola_col)) if escola_col else np.nan
        turma  = normalizar_turma(row.get(turma_col))  if turma_col  else np.nan
        total  = row.get(total_col, np.nan) if total_col else np.nan

        # Socioeconômico
        sexo   = row.get(sexo_col,  np.nan) if sexo_col  else np.nan
        nasc   = row.get(nasc_col,  np.nan) if nasc_col  else np.nan
        idade  = row.get(idade_col, np.nan) if idade_col else np.nan
        raca   = row.get(raca_col,  np.nan) if raca_col  else np.nan
        zona   = row.get(zona_col,  np.nan) if zona_col  else np.nan
        nec    = row.get(nec_col,   np.nan) if nec_col   else np.nan
        resp   = row.get(resp_col,  np.nan) if resp_col  else np.nan

        # Monitor da planilha
        mon_planilha = row.get(monitor_col, np.nan) if monitor_col else np.nan

        # Matrícula
        mat = converter_matricula(row.get(mat_col)) if mat_col else np.nan

        # Questões — classificar com classificar_celula (máximo 1)
        questoes = {}
        for i in range(1, 51):
            chave = f"Q{i:02d}"
            c_orig = next((c for c in q_cols if c == f"Q{i}" or c == chave), None)
            val_raw = row.get(c_orig, np.nan) if c_orig else np.nan
            val_limpo, flag = classificar_celula(val_raw, max_valor=1)
            CONTADORES_FLAGS["vocabulario"][flag] += 1
            questoes[chave] = val_limpo if flag == "ok" else np.nan

        n_itens = 50
        n_validos = sum(1 for v in questoes.values() if not pd.isna(v))
        pct_completude = round(n_validos / n_itens, 4)
        completo = bool(pct_completude >= 0.80)
        total_val = float(sum(v for v in questoes.values() if not pd.isna(v))) if n_validos > 0 else np.nan

        linha_origem = int(row.get("_linha_origem", 0))

        record = {
            "aluno_id":               aluno_id,
            "matricula":              mat,
            "escola":                 escola,
            "turma":                  turma,
            "fase":                   fase,
            "periodo":                periodo,
            "periodo_fonte":          periodo_fonte,
            "ano_semestre":           {2: "2023.2", 3: "2024.1", 4: "2024.2"}.get(fase, ""),
            "monitor":                str(mon_planilha).strip() if not pd.isna(mon_planilha) else monitor,
            "coordenador":            coord,
            "sexo":                   sexo,
            "data_nascimento":        nasc,
            "idade":                  idade,
            "cor_raca":               raca,
            "zona":                   zona,
            "necessidade_deficiencia":nec,
            "responsavel_legal":      resp,
            "total_vocabulario":      total_val,
            "n_itens":                n_itens,
            "n_validos":              n_validos,
            "pct_completude":         pct_completude,
            "completo":               completo,
            "arquivo_id":             arquivo_id,
            "linha_origem":           linha_origem,
            **questoes,
        }
        rows.append(record)

    return pd.DataFrame(rows) if rows else None

# ─────────────────────────────────────────────
# NORMALIZAÇÃO — TDE
# ─────────────────────────────────────────────

def normalizar_tde(df, fase, periodo, monitor="", coord="", arquivo_id="", periodo_fonte=""):
    """
    Retorna DataFrame normalizado para dataset TDE.
    Aceita os dois esquemas (Fase2 e Fase3/4).
    Trata também o esquema duplo (pré+pós na mesma planilha com 2 blocos P1-P40).
    """
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # ── Detectar esquema duplo (pré+pós na mesma planilha) ──
    # Identificado quando a primeira linha de cabeçalho tem "Préteste" e "Pósteste"
    # e a linha 2 tem as colunas reais. Neste caso o df já foi lido da linha 2.
    # O padrão: existem colunas duplicadas P1, P1.1, P2, P2.1 etc.
    has_duplo = any(re.match(r"^P\d+\.\d+$", c) for c in cols)
    if has_duplo:
        # Separar em dois blocos e processar separado
        result_frames = []
        for per, suffix in [("pre", ""), ("pos", ".1")]:
            sub_cols = {}
            for i in range(1, 41):
                base = f"P{i}"
                col_name = base if suffix == "" else f"{base}{suffix}"
                if col_name in cols:
                    sub_cols[base] = col_name
            if not sub_cols:
                continue
            # Montar sub-df com as colunas certas
            rename_map = {v: k for k, v in sub_cols.items()}
            sub_df = df.rename(columns=rename_map)
            resultado = normalizar_tde(
                sub_df, fase=fase, periodo=per, monitor=monitor, coord=coord,
                arquivo_id=arquivo_id, periodo_fonte=periodo_fonte,
            )
            if resultado is not None:
                result_frames.append(resultado)
        if result_frames:
            return pd.concat(result_frames, ignore_index=True)
        return None

    aluno_col = None
    for c in ["Nome", "NOME DO ESTUDANTE", "NOME ESTUDANTE", "NOME DO ALUNO"]:
        if c in cols:
            aluno_col = c
            break
    if aluno_col is None:
        return None

    p_cols = [c for c in cols if re.match(r"^P\d+$", c)]
    if not p_cols:
        return None

    escola_col = next((c for c in ["Escola", "ESCOLA"] if c in cols), None)
    turma_col  = next((c for c in ["Turma", "TURMA"] if c in cols), None)
    total_col  = next((c for c in ["Total", "TDE (score)"] if c in cols), None)

    sexo_col    = next((c for c in ["Sexo"] if c in cols), None)
    nasc_col    = next((c for c in ["Data Nascimento"] if c in cols), None)
    idade_col   = next((c for c in ["Idade"] if c in cols), None)
    raca_col    = next((c for c in ["Cor/Raça"] if c in cols), None)
    zona_col    = next((c for c in ["Zona"] if c in cols), None)
    nec_col     = next((c for c in ["Necessidade/Deficiência"] if c in cols), None)
    resp_col    = next((c for c in ["Responsável Legal"] if c in cols), None)
    monitor_col = next((c for c in ["MONITOR", "monitor"] if c in cols), None)
    mat_col     = next((c for c in ["MATRÍCULA"] if c in cols), None)

    rows = []
    for _, row in df.iterrows():
        aluno_id = normalizar_aluno_id(row.get(aluno_col))
        if aluno_id is None:
            continue

        escola = normalizar_escola(row.get(escola_col)) if escola_col else np.nan
        turma  = normalizar_turma(row.get(turma_col))  if turma_col  else np.nan
        total  = row.get(total_col, np.nan) if total_col else np.nan

        sexo   = row.get(sexo_col,  np.nan) if sexo_col  else np.nan
        nasc   = row.get(nasc_col,  np.nan) if nasc_col  else np.nan
        idade  = row.get(idade_col, np.nan) if idade_col else np.nan
        raca   = row.get(raca_col,  np.nan) if raca_col  else np.nan
        zona   = row.get(zona_col,  np.nan) if zona_col  else np.nan
        nec    = row.get(nec_col,   np.nan) if nec_col   else np.nan
        resp   = row.get(resp_col,  np.nan) if resp_col  else np.nan

        mon_planilha = row.get(monitor_col, np.nan) if monitor_col else np.nan
        mat = converter_matricula(row.get(mat_col)) if mat_col else np.nan

        # Questões — classificar com classificar_celula (máximo 2)
        questoes = {}
        for i in range(1, 41):
            chave = f"P{i:02d}"
            c_orig = next((c for c in p_cols if c == f"P{i}" or c == chave), None)
            val_raw = row.get(c_orig, np.nan) if c_orig else np.nan
            val_limpo, flag = classificar_celula(val_raw, max_valor=2)
            CONTADORES_FLAGS["tde"][flag] += 1
            questoes[chave] = val_limpo if flag == "ok" else np.nan

        n_itens = 40
        n_validos = sum(1 for v in questoes.values() if not pd.isna(v))
        pct_completude = round(n_validos / n_itens, 4)
        completo = bool(pct_completude >= 0.80)
        total_val = float(sum(v for v in questoes.values() if not pd.isna(v))) if n_validos > 0 else np.nan

        linha_origem = int(row.get("_linha_origem", 0))

        record = {
            "aluno_id":               aluno_id,
            "matricula":              mat,
            "escola":                 escola,
            "turma":                  turma,
            "fase":                   fase,
            "periodo":                periodo,
            "periodo_fonte":          periodo_fonte,
            "ano_semestre":           {2: "2023.2", 3: "2024.1", 4: "2024.2"}.get(fase, ""),
            "monitor":                str(mon_planilha).strip() if not pd.isna(mon_planilha) else monitor,
            "coordenador":            coord,
            "sexo":                   sexo,
            "data_nascimento":        nasc,
            "idade":                  idade,
            "cor_raca":               raca,
            "zona":                   zona,
            "necessidade_deficiencia":nec,
            "responsavel_legal":      resp,
            "total_tde":              total_val,
            "n_itens":                n_itens,
            "n_validos":              n_validos,
            "pct_completude":         pct_completude,
            "completo":               completo,
            "arquivo_id":             arquivo_id,
            "linha_origem":           linha_origem,
            **questoes,
        }
        rows.append(record)

    return pd.DataFrame(rows) if rows else None

# ─────────────────────────────────────────────
# PROCESSAMENTO POR FASE
# ─────────────────────────────────────────────

vocab_frames = []
tde_frames   = []
log = []

def registrar(fase, periodo, filepath, tipo, n_rows, status, obs=""):
    log.append({
        "fase": fase, "periodo": periodo,
        "arquivo": Path(filepath).name,
        "tipo": tipo, "linhas_processadas": n_rows,
        "status": status, "observacao": obs,
    })

# ══════════════════════════════════════════════
# FASE 2
# ══════════════════════════════════════════════
fase2_base = RAW_DIR / "Fase2" / "Fase 2 - Dados"

for periodo_folder in ["Pré-teste", "Pós-teste"]:
    periodo_path = fase2_base / periodo_folder
    if not periodo_path.exists():
        continue
    periodo = detectar_periodo_fase2(periodo_path)
    periodo_fonte = "pasta"

    # Vocabulário
    for vocab_sub in ["1 - Avaliação de Vocabulário", "1 - Avaliação de vocabulário"]:
        vocab_path = periodo_path / vocab_sub
        if vocab_path.exists():
            for f in listar_csvs(vocab_path):
                fn_lower = f.name.lower()
                if deve_excluir(fn_lower):
                    registrar(2, periodo, f, "vocabulario", 0, "IGNORADO", "padrão de exclusão")
                    continue
                arq_id = calcular_arquivo_id(f)
                monitor, coord = extrair_monitor_coord(f.stem)
                df_raw = ler_csv(f)
                df_norm = normalizar_vocabulario(
                    df_raw, fase=2, periodo=periodo, monitor=monitor, coord=coord,
                    arquivo_id=arq_id, periodo_fonte=periodo_fonte,
                )
                if df_norm is not None and len(df_norm) > 0:
                    vocab_frames.append(df_norm)
                    registrar(2, periodo, f, "vocabulario", len(df_norm), "OK")
                else:
                    registrar(2, periodo, f, "vocabulario", 0, "VAZIO_OU_NAO_RECONHECIDO")

    # TDE
    for tde_sub in ["TDE II - Subteste Escrita", "TDE II - Subteste de escrita"]:
        tde_path = periodo_path / tde_sub
        if tde_path.exists():
            for f in listar_csvs(tde_path):
                fn_lower = f.name.lower()
                if deve_excluir(fn_lower):
                    registrar(2, periodo, f, "tde", 0, "IGNORADO", "padrão de exclusão")
                    continue
                arq_id = calcular_arquivo_id(f)
                monitor, coord = extrair_monitor_coord(f.stem)
                df_raw = ler_csv(f)
                df_norm = normalizar_tde(
                    df_raw, fase=2, periodo=periodo, monitor=monitor, coord=coord,
                    arquivo_id=arq_id, periodo_fonte=periodo_fonte,
                )
                if df_norm is not None and len(df_norm) > 0:
                    tde_frames.append(df_norm)
                    registrar(2, periodo, f, "tde", len(df_norm), "OK")
                else:
                    registrar(2, periodo, f, "tde", 0, "VAZIO_OU_NAO_RECONHECIDO")

# ══════════════════════════════════════════════
# FASE 3
# ══════════════════════════════════════════════
fase3_base = RAW_DIR / "Fase3"

for periodo_folder in ["Pré-teste", "Pós-teste"]:
    periodo_path = fase3_base / periodo_folder
    if not periodo_path.exists():
        continue
    periodo = detectar_periodo_fase3(periodo_path)
    periodo_fonte = "pasta"

    # Vocabulário — subpasta "Avaliação de Vocabulário"
    vocab_subpasta_files = set()
    for vocab_sub in ["Avaliação de Vocabulário", "Avaliação de vocabulário"]:
        vocab_path = periodo_path / vocab_sub
        if vocab_path.exists():
            for f in listar_csvs(vocab_path):
                fn_lower = f.name.lower()
                if deve_excluir(fn_lower):
                    registrar(3, periodo, f, "vocabulario", 0, "IGNORADO", "padrão de exclusão")
                    continue
                vocab_subpasta_files.add(f.name)
                arq_id = calcular_arquivo_id(f)
                monitor, coord = extrair_monitor_coord(f.stem)
                df_raw = ler_csv(f)
                df_norm = normalizar_vocabulario(
                    df_raw, fase=3, periodo=periodo, monitor=monitor, coord=coord,
                    arquivo_id=arq_id, periodo_fonte=periodo_fonte,
                )
                if df_norm is not None and len(df_norm) > 0:
                    vocab_frames.append(df_norm)
                    registrar(3, periodo, f, "vocabulario", len(df_norm), "OK")
                else:
                    registrar(3, periodo, f, "vocabulario", 0, "VAZIO_OU_NAO_RECONHECIDO")

    # Vocabulário — arquivos diretos na raiz do Pré/Pós-teste (pean_vocabulario)
    # Evitar duplicar arquivos que já estão na subpasta
    for f in listar_csvs(periodo_path):
        fn_lower = f.name.lower()
        if f.name in vocab_subpasta_files:
            continue
        if "_vocabulario" in fn_lower and not deve_excluir(fn_lower):
            arq_id = calcular_arquivo_id(f)
            monitor, coord = extrair_monitor_coord(f.stem)
            df_raw = ler_csv(f)
            df_norm = normalizar_vocabulario(
                df_raw, fase=3, periodo=periodo, monitor=monitor, coord=coord,
                arquivo_id=arq_id, periodo_fonte=periodo_fonte,
            )
            if df_norm is not None and len(df_norm) > 0:
                vocab_frames.append(df_norm)
                registrar(3, periodo, f, "vocabulario", len(df_norm), "OK", "raiz pré/pós-teste")
            else:
                registrar(3, periodo, f, "vocabulario", 0, "VAZIO_OU_NAO_RECONHECIDO")

    # TDE — subpasta "TDE II"
    tde_subpasta_files = set()
    tde_path = periodo_path / "TDE II"
    if tde_path.exists():
        for f in listar_csvs(tde_path):
            fn_lower = f.name.lower()
            if deve_excluir(fn_lower):
                registrar(3, periodo, f, "tde", 0, "IGNORADO", "padrão de exclusão")
                continue
            tde_subpasta_files.add(f.name)
            arq_id = calcular_arquivo_id(f)
            monitor, coord = extrair_monitor_coord(f.stem)
            df_raw = ler_csv(f)
            df_norm = normalizar_tde(
                df_raw, fase=3, periodo=periodo, monitor=monitor, coord=coord,
                arquivo_id=arq_id, periodo_fonte=periodo_fonte,
            )
            if df_norm is not None and len(df_norm) > 0:
                tde_frames.append(df_norm)
                registrar(3, periodo, f, "tde", len(df_norm), "OK")
            else:
                registrar(3, periodo, f, "tde", 0, "VAZIO_OU_NAO_RECONHECIDO")

    # TDE — arquivos diretos na raiz (pean_ted_ii)
    # Evitar duplicar arquivos que já estão na subpasta
    for f in listar_csvs(periodo_path):
        fn_lower = f.name.lower()
        if f.name in tde_subpasta_files:
            continue
        if ("_ted_ii" in fn_lower or "_tde_ii" in fn_lower) and not deve_excluir(fn_lower):
            arq_id = calcular_arquivo_id(f)
            monitor, coord = extrair_monitor_coord(f.stem)
            df_raw = ler_csv(f)
            df_norm = normalizar_tde(
                df_raw, fase=3, periodo=periodo, monitor=monitor, coord=coord,
                arquivo_id=arq_id, periodo_fonte=periodo_fonte,
            )
            if df_norm is not None and len(df_norm) > 0:
                tde_frames.append(df_norm)
                registrar(3, periodo, f, "tde", len(df_norm), "OK", "raiz pré/pós-teste")
            else:
                registrar(3, periodo, f, "tde", 0, "VAZIO_OU_NAO_RECONHECIDO")

# ══════════════════════════════════════════════
# FASE 4 — tudo numa pasta só
# ══════════════════════════════════════════════
fase4_base = RAW_DIR / "Fase4" / "WordGen - Dados"

if fase4_base.exists():
    for f in listar_csvs(fase4_base):
        fn_lower = f.name.lower()
        fn_orig  = f.name

        if deve_excluir(fn_lower):
            registrar(4, "?", f, "?", 0, "IGNORADO", "padrão de exclusão")
            continue

        periodo, periodo_fonte = detectar_periodo_fase4(fn_orig)
        arq_id = calcular_arquivo_id(f)
        monitor, coord = extrair_monitor_coord(f.stem)

        if "_vocabulario" in fn_lower:
            df_raw = ler_csv(f)
            df_norm = normalizar_vocabulario(
                df_raw, fase=4, periodo=periodo, monitor=monitor, coord=coord,
                arquivo_id=arq_id, periodo_fonte=periodo_fonte,
            )
            if df_norm is not None and len(df_norm) > 0:
                vocab_frames.append(df_norm)
                registrar(4, periodo, f, "vocabulario", len(df_norm), "OK")
            else:
                registrar(4, periodo, f, "vocabulario", 0, "VAZIO_OU_NAO_RECONHECIDO")

        elif "_ted_ii" in fn_lower or "_tde_ii" in fn_lower:
            df_raw = ler_csv(f)
            df_norm = normalizar_tde(
                df_raw, fase=4, periodo=periodo, monitor=monitor, coord=coord,
                arquivo_id=arq_id, periodo_fonte=periodo_fonte,
            )
            if df_norm is not None and len(df_norm) > 0:
                tde_frames.append(df_norm)
                registrar(4, periodo, f, "tde", len(df_norm), "OK")
            else:
                registrar(4, periodo, f, "tde", 0, "VAZIO_OU_NAO_RECONHECIDO")

# ─────────────────────────────────────────────
# CONSOLIDAÇÃO FINAL
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("  CONSOLIDANDO DATASETS")
print("="*60)

metricas_dedup = {
    "vocabulario": {},
    "tde": {},
    "socioeconomico": {},
}

# ── Vocabulário ──
if vocab_frames:
    df_vocab = pd.concat(vocab_frames, ignore_index=True)
    n_bruto_v = len(df_vocab)

    # Ordenar antes do drop_duplicates: aluno_id, fase, periodo, n_validos desc, arquivo_id asc, linha_origem asc
    df_vocab = df_vocab.sort_values(
        by=["aluno_id", "fase", "periodo", "n_validos", "arquivo_id", "linha_origem"],
        ascending=[True, True, True, False, True, True],
    )

    # Contar chaves com conflito
    q_cols_check = [f"Q{i:02d}" for i in range(1, 51)]
    conflitos_v = contar_conflitos(df_vocab, ["aluno_id", "fase", "periodo"], q_cols_check)

    # Deduplicar mantendo o primeiro
    df_vocab = df_vocab.drop_duplicates(subset=["aluno_id", "fase", "periodo"], keep="first")
    n_dedup_v = len(df_vocab)

    # Remover registros com n_validos == 0
    vazios_v = int((df_vocab["n_validos"] == 0).sum())
    df_vocab = df_vocab[df_vocab["n_validos"] > 0].copy()

    # Ordenação fixa final
    df_vocab = df_vocab.sort_values(by=["aluno_id", "fase", "periodo"])

    out_vocab = OUT_DIR / "dataset_vocabulario.csv"
    df_vocab.to_csv(out_vocab, index=False, encoding="utf-8-sig")

    metricas_dedup["vocabulario"] = {
        "linhas_brutas": n_bruto_v,
        "duplicatas_removidas": n_bruto_v - n_dedup_v,
        "conflitos": conflitos_v,
        "vazios_removidos": vazios_v,
        "linhas_finais": len(df_vocab),
        "alunos_unicos": int(df_vocab["aluno_id"].nunique()),
    }

    print(f"\n[VOCABULÁRIO] {len(df_vocab)} registros → {out_vocab}")
    print(f"  Brutas: {n_bruto_v} | Duplicatas: {n_bruto_v - n_dedup_v} | Conflitos: {conflitos_v} | Vazios removidos: {vazios_v}")
    print(f"  Alunos únicos : {df_vocab['aluno_id'].nunique()}")
    print(f"  Escolas       : {df_vocab['escola'].nunique()}")
    print(f"  Turmas        : {df_vocab['turma'].nunique()}")
    print(f"  Por fase/período:")
    print(df_vocab.groupby(["fase","periodo"]).size().to_string())
else:
    print("[VOCABULÁRIO] Nenhum frame coletado!")

# ── TDE ──
if tde_frames:
    df_tde = pd.concat(tde_frames, ignore_index=True)
    n_bruto_t = len(df_tde)

    df_tde = df_tde.sort_values(
        by=["aluno_id", "fase", "periodo", "n_validos", "arquivo_id", "linha_origem"],
        ascending=[True, True, True, False, True, True],
    )

    p_cols_check = [f"P{i:02d}" for i in range(1, 41)]
    conflitos_t = contar_conflitos(df_tde, ["aluno_id", "fase", "periodo"], p_cols_check)

    df_tde = df_tde.drop_duplicates(subset=["aluno_id", "fase", "periodo"], keep="first")
    n_dedup_t = len(df_tde)

    vazios_t = int((df_tde["n_validos"] == 0).sum())
    df_tde = df_tde[df_tde["n_validos"] > 0].copy()

    df_tde = df_tde.sort_values(by=["aluno_id", "fase", "periodo"])

    out_tde = OUT_DIR / "dataset_tde.csv"
    df_tde.to_csv(out_tde, index=False, encoding="utf-8-sig")

    metricas_dedup["tde"] = {
        "linhas_brutas": n_bruto_t,
        "duplicatas_removidas": n_bruto_t - n_dedup_t,
        "conflitos": conflitos_t,
        "vazios_removidos": vazios_t,
        "linhas_finais": len(df_tde),
        "alunos_unicos": int(df_tde["aluno_id"].nunique()),
    }

    print(f"\n[TDE]         {len(df_tde)} registros → {out_tde}")
    print(f"  Brutas: {n_bruto_t} | Duplicatas: {n_bruto_t - n_dedup_t} | Conflitos: {conflitos_t} | Vazios removidos: {vazios_t}")
    print(f"  Alunos únicos : {df_tde['aluno_id'].nunique()}")
    print(f"  Escolas       : {df_tde['escola'].nunique()}")
    print(f"  Turmas        : {df_tde['turma'].nunique()}")
    print(f"  Por fase/período:")
    print(df_tde.groupby(["fase","periodo"]).size().to_string())
else:
    print("[TDE] Nenhum frame coletado!")

# ── Dados Socioeconômicos consolidados ──
socio_cols = [
    "aluno_id", "escola", "turma", "fase", "ano_semestre",
    "sexo", "data_nascimento", "idade", "cor_raca",
    "zona", "necessidade_deficiencia", "responsavel_legal",
    "arquivo_id", "linha_origem",
]

socio_frames = []
for ds_name, ds in [("vocabulario", df_vocab if vocab_frames else None),
                     ("tde", df_tde if tde_frames else None)]:
    if ds is not None:
        avail = [c for c in socio_cols if c in ds.columns]
        tmp = ds[avail].copy()
        socio_frames.append(tmp)

if socio_frames:
    df_socio = pd.concat(socio_frames, ignore_index=True)
    n_bruto_s = len(df_socio)

    campos_socio_validar = [c for c in [
        "sexo", "data_nascimento", "idade", "cor_raca", "zona",
        "necessidade_deficiencia", "responsavel_legal"
    ] if c in df_socio.columns]
    df_socio["n_preenchidos"] = df_socio[campos_socio_validar].notna().sum(axis=1)

    df_socio = df_socio.sort_values(
        by=["aluno_id", "fase", "n_preenchidos", "arquivo_id", "linha_origem"],
        ascending=[True, True, False, True, True],
    )
    df_socio = df_socio.drop_duplicates(subset=["aluno_id", "fase"], keep="first")

    # Manter apenas registros com pelo menos sexo ou cor_raca preenchido
    mask = df_socio["sexo"].notna() | df_socio["cor_raca"].notna() if "sexo" in df_socio.columns else df_socio.notna().any(axis=1)
    df_socio = df_socio[mask].copy()

    df_socio = df_socio.sort_values(by=["aluno_id", "fase"])

    out_socio = OUT_DIR / "dataset_socioeconomico.csv"
    df_socio.to_csv(out_socio, index=False, encoding="utf-8-sig")

    metricas_dedup["socioeconomico"] = {
        "linhas_brutas": n_bruto_s,
        "linhas_finais": len(df_socio),
        "alunos_unicos": int(df_socio["aluno_id"].nunique()),
    }
    print(f"\n[SOCIOEC.]    {len(df_socio)} registros → {out_socio}")

out_dedup_json = OUT_DIR / "metricas_deduplicacao.json"
with open(out_dedup_json, "w", encoding="utf-8") as f_d:
    json.dump(metricas_dedup, f_d, indent=2)

# ── Métricas de limpeza de células ──
metricas_limpeza = {
    "vocabulario": dict(CONTADORES_FLAGS["vocabulario"]),
    "tde": dict(CONTADORES_FLAGS["tde"]),
}
out_metricas = OUT_DIR / "metricas_limpeza.json"
with open(out_metricas, "w", encoding="utf-8") as f_met:
    json.dump(metricas_limpeza, f_met, indent=2)
print(f"\n[LIMPEZA] Métricas de células salvas → {out_metricas}")
print(f"  Vocabulário: {metricas_limpeza['vocabulario']}")
print(f"  TDE        : {metricas_limpeza['tde']}")

# ── Log de processamento ──
df_log = pd.DataFrame(log)
out_log = OUT_DIR / "log_processamento.csv"
df_log.to_csv(out_log, index=False, encoding="utf-8-sig")
print(f"\n[LOG]         {len(df_log)} entradas → {out_log}")
print(f"  Status OK      : {(df_log['status']=='OK').sum()}")
print(f"  Ignorados      : {(df_log['status']=='IGNORADO').sum()}")
print(f"  Vazios/Erro    : {(df_log['status']!='OK').sum() - (df_log['status']=='IGNORADO').sum()}")
print("\n" + "="*60)
print("  PIPELINE CONCLUÍDA COM SUCESSO")
print("="*60 + "\n")
