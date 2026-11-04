"""Avaliação local e reprodutível contra o benchmark de perguntas publicado."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from wordgen.agente.orquestrador import AgenteWordGen


@dataclass(frozen=True)
class ResultadoBenchmark:
    identificador: str
    categoria: str
    rota: str
    respondeu: bool
    possui_fonte: bool
    fonte_esperada: str
    fonte_corresponde: bool
    valores_correspondem: bool
    rota_adequada: bool
    aprovado: bool
    justificativa: str


def avaliar_benchmark(agente: AgenteWordGen, caminho: Path) -> list[ResultadoBenchmark]:
    """Verifica cobertura, rota e proveniência sem fazer chamadas externas ao LLM."""
    if not caminho.exists():
        raise FileNotFoundError("Benchmark QA não encontrado.")
    itens = json.loads(caminho.read_text(encoding="utf-8"))
    if not isinstance(itens, list):
        raise ValueError("Benchmark QA deve ser uma lista de perguntas.")

    resultados: list[ResultadoBenchmark] = []
    for item in itens:
        resposta = agente.responder(str(item["pergunta"]), usar_llm=False)
        fonte_esperada = str(item.get("fonte", ""))
        fonte_corresponde = any(
            _fonte_equivalente(fonte.nome, fonte_esperada) for fonte in resposta.fontes
        )
        valores_correspondem = _valores_presentes(item.get("valor_exato"), resposta.texto)
        rota_adequada = _rota_adequada(str(item["categoria"]), resposta.rota)
        aprovado = (
            bool(resposta.texto.strip())
            and resposta.rota not in {"sem_evidencia", "fora_escopo", "esclarecimento_escopo"}
            and fonte_corresponde
            and valores_correspondem
            and rota_adequada
        )
        justificativa = _justificativa(
            fonte_corresponde, valores_correspondem, rota_adequada, resposta.rota
        )
        resultados.append(
            ResultadoBenchmark(
                identificador=str(item["id"]),
                categoria=str(item["categoria"]),
                rota=resposta.rota,
                respondeu=bool(resposta.texto.strip()) and resposta.rota != "sem_evidencia",
                possui_fonte=bool(resposta.fontes),
                fonte_esperada=fonte_esperada,
                fonte_corresponde=fonte_corresponde,
                valores_correspondem=valores_correspondem,
                rota_adequada=rota_adequada,
                aprovado=aprovado,
                justificativa=justificativa,
            )
        )
    return resultados


def resumo_benchmark(resultados: list[ResultadoBenchmark]) -> dict[str, int]:
    """Resume cobertura técnica; avaliação humana continua necessária para fidelidade semântica."""
    return {
        "total": len(resultados),
        "respondidas": sum(item.respondeu for item in resultados),
        "com_fontes": sum(item.possui_fonte for item in resultados),
        "aprovadas": sum(item.aprovado for item in resultados),
    }


def _fonte_equivalente(fonte: str, esperada: str) -> bool:
    nome_fonte = fonte.split(" · ", maxsplit=1)[0]
    nome_fonte = re.sub(r"\.(csv|parquet)$", "", nome_fonte.replace("\\", "/").split("/")[-1])
    nome_esperado = re.sub(r"\.(csv|parquet)$", "", esperada.replace("\\", "/").split("/")[-1])
    return bool(nome_esperado) and nome_esperado == nome_fonte


def _valores_presentes(valor: object, resposta: str) -> bool:
    """Confere valores esperados com tolerância de apresentação, sem avaliar linguagem livre."""
    if valor is None:
        return True
    if isinstance(valor, dict):
        return all(_valores_presentes(item, resposta) for item in valor.values())
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        if isinstance(valor, float) and math.isnan(valor):
            return True
        representacoes = {
            str(valor),
            f"{valor:.2f}",
            f"{valor:.3f}",
            f"{valor:,.0f}".replace(",", "."),
            f"{valor:,.0f}",
        }
        representacoes.update(item.replace(".", ",") for item in tuple(representacoes))
        if 0 < abs(float(valor)) < 1:
            representacoes.add(f"{float(valor) * 100:.0f}%")
        return any(item in resposta for item in representacoes)
    return str(valor) in resposta


def _rota_adequada(categoria: str, rota: str) -> bool:
    categoria = categoria.lower()
    if categoria in {"geral", "desempenho", "longitudinal"}:
        return rota.startswith("consulta_") or rota.startswith("rag_")
    return rota.startswith("rag_")


def _justificativa(
    fonte_corresponde: bool, valores_correspondem: bool, rota_adequada: bool, rota: str
) -> str:
    falhas = []
    if not fonte_corresponde:
        falhas.append("fonte esperada não identificada")
    if not valores_correspondem:
        falhas.append("valor esperado ausente ou divergente")
    if not rota_adequada:
        falhas.append(f"rota inadequada: {rota}")
    return "aprovada" if not falhas else "; ".join(falhas)
