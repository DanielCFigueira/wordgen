"""Caso de uso: montar o catálogo dos arquivos brutos.

A leitura de disco chega por parâmetro (ler_linhas, calcular_hash), então este
módulo depende só do domínio e pode ser testado sem arquivos reais.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from wordgen.domain.catalogo import (
    RegistroCatalogo,
    analisar_cabecalho,
    decidir_instrumento,
)
from wordgen.domain.classificacao import (
    fase_por_caminho,
    instrumento_por_caminho,
    momento_por_caminho,
)


def _assinatura(meta: tuple[str, ...], n_q: int, n_p: int) -> str:
    texto = "|".join(meta) + f"|Q{n_q}|P{n_p}"
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]


def construir_catalogo(
    arquivos: Sequence[Path],
    raiz: Path,
    ler_linhas: Callable[[Path], list[list[str]]],
    calcular_hash: Callable[[Path], str],
) -> list[RegistroCatalogo]:
    registros: list[RegistroCatalogo] = []
    for arquivo in sorted(arquivos, key=lambda p: p.relative_to(raiz).as_posix()):
        rel = arquivo.relative_to(raiz).as_posix()
        analise = analisar_cabecalho(ler_linhas(arquivo))
        por_caminho = instrumento_por_caminho(rel)
        instrumento, status = decidir_instrumento(por_caminho, analise)
        registros.append(
            RegistroCatalogo(
                arquivo=rel,
                fase=fase_por_caminho(rel),
                momento=momento_por_caminho(rel),
                instrumento=instrumento,
                instrumento_caminho=por_caminho,
                status=status,
                n_itens_q=analise.n_itens_q,
                n_itens_p=analise.n_itens_p,
                itens_repetidos=analise.itens_repetidos,
                linha_cabecalho=analise.linha,
                assinatura_esquema=_assinatura(
                    analise.colunas_meta, analise.n_itens_q, analise.n_itens_p
                ),
                sha256=calcular_hash(arquivo),
            )
        )
    return _marcar_duplicatas(registros)


def _marcar_duplicatas(registros: list[RegistroCatalogo]) -> list[RegistroCatalogo]:
    """Arquivos com o mesmo hash são cópias: a 1ª ocorrência (ordem alfabética) é a original."""
    primeiro: dict[str, str] = {}
    saida: list[RegistroCatalogo] = []
    for r in registros:
        original = primeiro.setdefault(r.sha256, r.arquivo)
        saida.append(replace(r, duplicado_de=None if original == r.arquivo else original))
    return saida
