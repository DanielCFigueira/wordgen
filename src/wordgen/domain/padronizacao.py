"""Regras de padronização dos valores de um arquivo bruto (funções puras, sem I/O)."""

import re
from typing import Any

from wordgen.domain.tipos import Instrumento

ESCALA_MAXIMA = {Instrumento.VOCABULARIO: 1, Instrumento.TDE: 2}

# 6 escolas canônicas do projeto
ESCOLAS_CANONICAS = (
    "EMEB PROFESSOR RICARDO VIEIRA DE LIMA",
    "EMEB PROFESSORA MARIA QUEIROZ FERRO",
    "EMEB PADRE ANCHIETA",
    "EMEF PADRE JOSÉ DOS SANTOS MOUSINHO",
    "EMEB EXPEDITO PORFÍRIO DOS SANTOS",
    "EMEB NATANAEL DA SILVA",
)

# Mapeamentos e regex para padronizar_escola
REGRAS_ESCOLA = [
    (re.compile(r"RICARDO\s+VIEIRA", re.IGNORECASE), "EMEB PROFESSOR RICARDO VIEIRA DE LIMA"),
    (
        re.compile(r"MARIA\s+QUEIROZ|QUEIROZ\s+FERRO", re.IGNORECASE),
        "EMEB PROFESSORA MARIA QUEIROZ FERRO",
    ),
    (re.compile(r"EXPEDITO|PORF[IÍ]RIO", re.IGNORECASE), "EMEB EXPEDITO PORFÍRIO DOS SANTOS"),
    (re.compile(r"MOUSINHO|PADRE\s+JOS[EÉ]", re.IGNORECASE), "EMEF PADRE JOSÉ DOS SANTOS MOUSINHO"),
    (re.compile(r"ANCHIETA", re.IGNORECASE), "EMEB PADRE ANCHIETA"),
    (re.compile(r"NATANAEL", re.IGNORECASE), "EMEB NATANAEL DA SILVA"),
]

_RE_ALUNO_ID = re.compile(r"^ALUNO_[0-9A-F]{16}$")


def classificar_celula(valor: Any, max_valor: int | Instrumento = 1) -> tuple[float | None, str]:
    """Classifica e limpa uma célula de resposta de item.

    Retorna tupla (valor_float, flag), onde flag pertence a:
    - 'ok': valor inteiro válido na escala [0, max_valor]
    - 'vazio': célula nula, vazia, espaço ou traço
    - 'fora_escala': número negativo, não inteiro ou superior ao máximo permitido
    - 'erro_planilha': erro de fórmula do Excel (ex.: #REF!, #N/A)
    - 'texto': letras, palavras ou anotações
    """
    if isinstance(max_valor, Instrumento):
        limite = ESCALA_MAXIMA[max_valor]
    else:
        limite = int(max_valor)

    if valor is None or isinstance(valor, bool):
        if isinstance(valor, bool):
            return None, "texto"
        return None, "vazio"

    s = str(valor).strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a", "-", "—", "–", "_", "."}:
        return None, "vazio"

    if s.startswith("#"):
        return None, "erro_planilha"

    # Tentativa de conversão numérica
    s_num = s.replace(",", ".")
    try:
        num = float(s_num)
    except ValueError:
        return None, "texto"

    if num.is_integer() and 0 <= int(num) <= limite:
        return float(int(num)), "ok"
    return None, "fora_escala"


def normalizar_id(aluno_id: Any) -> str | None:
    """Valida e normaliza o ID do aluno no formato ALUNO_<16_HEX>."""
    if aluno_id is None:
        return None
    s = str(aluno_id).strip().upper()
    if _RE_ALUNO_ID.match(s):
        return s
    return None


def normalizar_sexo(sexo: Any) -> str:
    """Padroniza sexo para 'M', 'F' ou 'NAO INFORMADO'."""
    if sexo is None:
        return "NAO INFORMADO"
    s = str(sexo).strip().upper()
    if s in {"M", "MASCULINO", "MASC"}:
        return "M"
    if s in {"F", "FEMININO", "FEM"}:
        return "F"
    return "NAO INFORMADO"


def parse_idade(idade: Any, data_nasc: Any = None) -> int | None:
    """Converte e valida idade escolar válida (entre 8 e 20 anos)."""
    if idade is not None:
        s = str(idade).replace(",", ".").strip()
        try:
            val = float(s)
            if 8 <= val <= 20:
                return int(val)
        except ValueError:
            pass

    # Tenta derivar da data de nascimento se fornecida
    if data_nasc is not None:
        s_nasc = str(data_nasc).strip()
        try:
            serial = float(s_nasc)
            if 30000 <= serial <= 45000:
                ano = 1900 + int(serial / 365.25)
                idade_calc = 2024 - ano
                if 8 <= idade_calc <= 20:
                    return idade_calc
        except ValueError:
            m = re.search(r"(\d{4})", s_nasc)
            if m:
                ano = int(m.group(1))
                idade_calc = 2024 - ano
                if 8 <= idade_calc <= 20:
                    return idade_calc

    return None


def faixa_etaria(idade: int | None) -> str:
    """Calcula a faixa etária conforme especificação."""
    if idade is None:
        return "NAO INFORMADA"
    if idade <= 10:
        return "<=10"
    if 11 <= idade <= 12:
        return "11-12"
    if 13 <= idade <= 14:
        return "13-14"
    if idade >= 15:
        return "15+"
    return "NAO INFORMADA"


def padronizar_escola(escola: Any) -> str | None:
    """Padroniza a escola para uma das 6 canônicas ou None se for lixo."""
    if escola is None:
        return None
    s = str(escola).strip()
    if not s or s in {"1.0", "0.0", "//////////////"}:
        return None

    for padrao, cano in REGRAS_ESCOLA:
        if padrao.search(s):
            return cano
    return None


def padronizar_turma(turma: Any) -> tuple[int | None, str | None, str | None]:
    """Padroniza a turma retornando (serie, turma_letra, turma_canonica).

    Apenas séries de 6 a 9 são válidas. Se não houver letra, usa 'U'.
    'Período' sem 'ano' não é considerado série.
    Retorna (None, None, None) se não for turma válida.
    """
    if turma is None:
        return None, None, None
    s = str(turma).strip().upper()
    if not s:
        return None, None, None

    # Se contiver 'ANO', a série é a associada a 'ANO'
    m_ano = re.search(r"([6-9])\s*[º°ªa]?\s*ANO(?:\s*[-–]?\s*([A-Z]))?", s)
    if m_ano:
        serie = int(m_ano.group(1))
        letra = m_ano.group(2)
        if not letra:
            m_letra = re.search(r"ANO[^-A-Z]*[-–]?\s*([A-Z])", s)
            letra = m_letra.group(1) if m_letra else "U"
        return serie, letra, f"{serie}{letra}"

    # Se contiver 'PER' sem 'ANO', não é série
    if "PER" in s:
        return None, None, None

    # Formato direto '6C' ou com indicador ordinal '6º C', '6 B'
    m_direto = re.match(r"^([6-9])\s*[º°ªa]?\s*[-–]?\s*([A-Z])$", s)
    if m_direto:
        serie = int(m_direto.group(1))
        letra = m_direto.group(2)
        return serie, letra, f"{serie}{letra}"

    return None, None, None
