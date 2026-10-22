"""Tipos fundamentais do domínio: o que um arquivo de dados pode ser."""

from enum import StrEnum


class Instrumento(StrEnum):
    VOCABULARIO = "vocabulario"
    TDE = "tde"
    OUTRO = "outro"


class Momento(StrEnum):
    PRE = "pre"
    POS = "pos"
    DESCONHECIDO = "desconhecido"
