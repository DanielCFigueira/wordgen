"""Regras que classificam um arquivo (instrumento, fase, momento) pelo caminho.

O caminho deve ser relativo à pasta raiz dos CSVs, para que pastas externas
(como C:\\dev\\wordgen) não interfiram na classificação.
"""

import re

from wordgen.domain.texto import normalizar
from wordgen.domain.tipos import Instrumento, Momento

# Os lookarounds tratam "_", dígitos e pontuação como separadores
# (ex.: "prvl_ted_ii"), mas não deixam "ted" casar dentro de uma palavra.
_RE_VOCAB = re.compile(r"vocabul")
_RE_TDE = re.compile(r"(?<![a-z])(tde|ted|stari)(?![a-z])")
_RE_FASE = re.compile(r"fase\s*(\d)")
_RE_PRE = re.compile(r"(?<![a-z])pre[\s_-]?test")
_RE_POS = re.compile(r"(?<![a-z])post?[\s_-]?test")


def instrumento_por_caminho(caminho: str) -> Instrumento:
    """Vocabulário ou TDE; qualquer outro caso (ou caso ambíguo) vira OUTRO."""
    texto = normalizar(caminho)
    eh_vocab = bool(_RE_VOCAB.search(texto))
    eh_tde = bool(_RE_TDE.search(texto))
    if eh_vocab and not eh_tde:
        return Instrumento.VOCABULARIO
    if eh_tde and not eh_vocab:
        return Instrumento.TDE
    return Instrumento.OUTRO


def fase_por_caminho(caminho: str) -> int | None:
    """Número da fase (a primeira menção no caminho) ou None."""
    achado = _RE_FASE.search(normalizar(caminho))
    return int(achado.group(1)) if achado else None


def momento_por_caminho(caminho: str) -> Momento:
    """Pré ou pós-teste; sem evidência, ou com evidências conflitantes, DESCONHECIDO."""
    texto = normalizar(caminho)
    eh_pre = bool(_RE_PRE.search(texto))
    eh_pos = bool(_RE_POS.search(texto))
    if eh_pre and not eh_pos:
        return Momento.PRE
    if eh_pos and not eh_pre:
        return Momento.POS
    return Momento.DESCONHECIDO
