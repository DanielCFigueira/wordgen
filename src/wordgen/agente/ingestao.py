"""Leitura da base de conhecimento RAG já publicada pelo projeto."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ORIGENS_ANALITICAS = {
    "wordgen_visao_geral": "data/analytics/dashboard_kpis.csv",
    "wordgen_longitudinal_transicao": "data/analytics/metricas_longitudinais.csv",
    "wordgen_equidade_territorial": "data/analytics/metricas_equidade.csv",
    "wordgen_perfil_inclusao": "data/analytics/perfil_socioeconomico_desempenho.csv",
}


@dataclass(frozen=True)
class DocumentoConhecimento:
    """Trecho recuperável, com a proveniência necessária para citação."""

    identificador: str
    titulo: str
    categoria: str
    conteudo: str
    tags: tuple[str, ...]
    metadados: dict[str, Any]
    origem: str = "documentos_rag.md"

    @property
    def fonte(self) -> str:
        return f"{self.origem} · {self.identificador}"


def carregar_base_conhecimento(caminho: Path) -> list[DocumentoConhecimento]:
    """Carrega e valida os chunks produzidos no notebook analítico 06."""
    if not caminho.exists():
        raise FileNotFoundError(
            "Base de conhecimento não encontrada. Gere os artefatos RAG antes de iniciar o agente."
        )

    payload = json.loads(caminho.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") if isinstance(payload, dict) else None
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("A base de conhecimento RAG não contém chunks válidos.")

    documentos: list[DocumentoConhecimento] = []
    for indice, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk RAG inválido na posição {indice}.")
        campos = ("chunk_id", "titulo", "categoria", "conteudo")
        if any(not str(chunk.get(campo, "")).strip() for campo in campos):
            raise ValueError(f"Chunk RAG incompleto na posição {indice}.")
        tags = chunk.get("tags", [])
        identificador = str(chunk["chunk_id"])
        origem = _ORIGENS_ANALITICAS.get(identificador, _origem_do_conteudo(chunk))
        documentos.append(
            DocumentoConhecimento(
                identificador=identificador,
                titulo=str(chunk["titulo"]),
                categoria=str(chunk["categoria"]),
                conteudo=str(chunk["conteudo"]),
                tags=tuple(str(tag) for tag in tags if str(tag).strip()),
                metadados={chave: valor for chave, valor in chunk.items() if chave not in campos},
                origem=origem,
            )
        )
    return documentos


def _origem_do_conteudo(chunk: dict[str, Any]) -> str:
    """Recupera a proveniência declarada no chunk, quando presente no conteúdo estruturado."""
    metricas = chunk.get("metricas_exatas")
    if isinstance(metricas, dict) and metricas.get("fonte"):
        return f"data/analytics/{str(metricas['fonte']).replace('.parquet', '.csv')}"
    achado = re.search(r'"fonte"\s*:\s*"([^"]+)"', str(chunk.get("conteudo", "")))
    if achado:
        return f"data/analytics/{achado.group(1).replace('.parquet', '.csv')}"
    return "documentos_rag.md"


def carregar_documentacao_modelagem(diretorio: Path) -> list[DocumentoConhecimento]:
    """Indexa seções metodológicas publicadas sem duplicar ou alterar os documentos de origem."""
    arquivos = (
        "estrategia_validacao.md",
        "definicao_problema.md",
        "avaliacao_modelos.md",
    )
    documentos: list[DocumentoConhecimento] = []
    for nome in arquivos:
        caminho = diretorio / nome
        if not caminho.exists():
            continue
        secoes = _secoes_markdown(caminho.read_text(encoding="utf-8"))
        for posicao, (titulo, conteudo) in enumerate(secoes, start=1):
            if len(conteudo) < 80:
                continue
            documentos.append(
                DocumentoConhecimento(
                    identificador=f"modelagem_{caminho.stem}_{posicao}",
                    titulo=titulo,
                    categoria="documentacao_modelagem",
                    conteudo=conteudo,
                    tags=("modelagem", "metodologia", caminho.stem.replace("_", " ")),
                    metadados={"arquivo": str(caminho.relative_to(diretorio.parents[1]))},
                    origem=str(caminho.relative_to(diretorio.parents[1])),
                )
            )
    return documentos


def _secoes_markdown(texto: str) -> list[tuple[str, str]]:
    """Segmenta seções e subseções para evitar respostas metodológicas excessivamente longas."""
    secoes: list[tuple[str, str]] = []
    titulo = "Documentação de modelagem"
    linhas: list[str] = []
    for linha in texto.splitlines():
        if linha.startswith(("## ", "### ")):
            if linhas:
                secoes.append((titulo, "\n".join(linhas).strip()))
            titulo = linha.removeprefix("## ").strip()
            linhas = []
        else:
            linhas.append(linha)
    if linhas:
        secoes.append((titulo, "\n".join(linhas).strip()))
    return secoes
