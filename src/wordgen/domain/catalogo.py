"""Entidades e regras do catálogo: o que cada arquivo de dados é.

Princípio: o CONTEÚDO (colunas de itens no cabeçalho) define o instrumento;
o caminho só corrobora. Se os dois discordam, o arquivo vai para revisão.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from wordgen.domain.texto import normalizar
from wordgen.domain.tipos import Instrumento, Momento

_RE_ITEM = re.compile(r"^([QP])(\d+)$", re.IGNORECASE)


class Status(StrEnum):
    OK = "ok"  # caminho e conteúdo concordam
    INFERIDO_PELO_CONTEUDO = "inferido_pelo_conteudo"  # caminho sem pista; itens definem
    DIVERGENTE = "divergente"  # caminho e conteúdo discordam: revisar
    SEM_ITENS = "sem_itens"  # caminho sugere avaliação, mas não há colunas de itens
    OUTRO = "outro"  # nem caminho nem conteúdo indicam avaliação


@dataclass(frozen=True)
class AnaliseCabecalho:
    linha: int  # índice (base 0) da linha de cabeçalho dentro do arquivo
    n_itens_q: int
    n_itens_p: int
    itens_repetidos: bool  # o mesmo item (ex.: P1) aparece mais de uma vez
    colunas_meta: tuple[str, ...]  # colunas que não são itens, já normalizadas


@dataclass(frozen=True)
class RegistroCatalogo:
    arquivo: str  # caminho relativo à raiz, com "/"
    fase: int | None
    momento: Momento
    instrumento: Instrumento
    instrumento_caminho: Instrumento
    status: Status
    n_itens_q: int
    n_itens_p: int
    itens_repetidos: bool
    linha_cabecalho: int
    assinatura_esquema: str
    sha256: str
    duplicado_de: str | None = None


def _tipo_item(celula: str) -> str | None:
    achado = _RE_ITEM.match(celula.strip())
    return achado.group(1).upper() if achado else None


def analisar_cabecalho(linhas: Sequence[Sequence[str]]) -> AnaliseCabecalho:
    """Acha a linha de cabeçalho: a que tem mais colunas de item (Q1.. ou P1..)."""
    melhor = (0, 0)  # (nº de itens, índice da linha)
    for indice, linha in enumerate(linhas):
        n = sum(1 for c in linha if _tipo_item(c))
        if n > melhor[0]:
            melhor = (n, indice)
    indice = melhor[1]
    cabecalho = list(linhas[indice]) if linhas else []
    tipos = [_tipo_item(c) for c in cabecalho]
    itens = [c.strip().upper() for c, t in zip(cabecalho, tipos, strict=True) if t]
    meta = tuple(
        normalizar(c) for c, t in zip(cabecalho, tipos, strict=True) if not t and c.strip()
    )
    return AnaliseCabecalho(
        linha=indice,
        n_itens_q=tipos.count("Q"),
        n_itens_p=tipos.count("P"),
        itens_repetidos=len(set(itens)) < len(itens),
        colunas_meta=meta,
    )


def decidir_instrumento(
    por_caminho: Instrumento, analise: AnaliseCabecalho
) -> tuple[Instrumento, Status]:
    """Combina caminho e conteúdo. Só é OK/INFERIDO quando o conteúdo é inequívoco."""
    q, p = analise.n_itens_q, analise.n_itens_p
    if q and p:
        return Instrumento.OUTRO, Status.DIVERGENTE
    if not q and not p:
        if por_caminho is Instrumento.OUTRO:
            return Instrumento.OUTRO, Status.OUTRO
        return Instrumento.OUTRO, Status.SEM_ITENS
    por_conteudo = Instrumento.VOCABULARIO if q else Instrumento.TDE
    if por_caminho is por_conteudo:
        return por_conteudo, Status.OK
    if por_caminho is Instrumento.OUTRO:
        return por_conteudo, Status.INFERIDO_PELO_CONTEUDO
    return Instrumento.OUTRO, Status.DIVERGENTE
