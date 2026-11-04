"""Índice vetorial lexical persistente para recuperação local e reproduzível."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from wordgen.agente.ingestao import DocumentoConhecimento

VERSAO_INDICE = 4
_STOPWORDS_PORTUGUES = [
    "a",
    "as",
    "ao",
    "aos",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "essa",
    "esse",
    "foi",
    "o",
    "os",
    "para",
    "por",
    "qual",
    "quais",
    "que",
    "uma",
    "um",
]


def _assinatura(documentos: list[DocumentoConhecimento]) -> str:
    texto = "\n".join(
        f"{documento.identificador}\n{documento.origem}\n{documento.titulo}\n{documento.conteudo}"
        for documento in documentos
    )
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


class IndiceVetorial:
    """TF-IDF persistido; evita download de modelos e torna a recuperação auditável."""

    def __init__(
        self,
        documentos: list[DocumentoConhecimento],
        vetorizar: TfidfVectorizer,
        matriz: object,
    ) -> None:
        self.documentos = documentos
        self.vetorizar = vetorizar
        self.matriz = matriz

    @classmethod
    def construir(cls, documentos: list[DocumentoConhecimento]) -> IndiceVetorial:
        if not documentos:
            raise ValueError("Não é possível indexar uma base de conhecimento vazia.")
        textos = [
            f"{documento.titulo} {' '.join(documento.tags)} {documento.conteudo}"
            for documento in documentos
        ]
        vetorizar = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words=_STOPWORDS_PORTUGUES,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        return cls(documentos, vetorizar, vetorizar.fit_transform(textos))

    @classmethod
    def carregar_ou_construir(
        cls, documentos: list[DocumentoConhecimento], caminho: Path
    ) -> IndiceVetorial:
        assinatura = _assinatura(documentos)
        if caminho.exists():
            try:
                payload = joblib.load(caminho)
                if (
                    payload.get("versao") == VERSAO_INDICE
                    and payload.get("assinatura") == assinatura
                ):
                    documentos_salvos = [
                        DocumentoConhecimento(**item) for item in payload["documentos"]
                    ]
                    return cls(documentos_salvos, payload["vetorizar"], payload["matriz"])
            except (AttributeError, KeyError, TypeError, ValueError):
                pass

        indice = cls.construir(documentos)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "versao": VERSAO_INDICE,
                "assinatura": assinatura,
                "documentos": [asdict(documento) for documento in documentos],
                "vetorizar": indice.vetorizar,
                "matriz": indice.matriz,
            },
            caminho,
        )
        return indice

    def buscar(
        self, consulta: str, limite: int = 4, limiar: float = 0.05
    ) -> list[tuple[DocumentoConhecimento, float]]:
        """Retorna somente chunks acima do limiar de similaridade lexical calibrado."""
        if not consulta.strip() or limite <= 0:
            return []
        vetor_consulta = self.vetorizar.transform([consulta])
        produto = self.matriz @ vetor_consulta.T
        scores = (
            produto.toarray().ravel()
            if hasattr(produto, "toarray")
            else np.asarray(produto).ravel()
        )
        indices = np.argsort(scores)[::-1]
        resultados = []
        for indice in indices:
            documento = self.documentos[indice]
            score = float(scores[indice])
            minimo = (
                max(limiar, 0.12) if documento.identificador == "wordgen_visao_geral" else limiar
            )
            if score < minimo:
                continue
            resultados.append((documento, score))
            if len(resultados) == limite:
                break
        return resultados

    def obter(self, identificador: str) -> DocumentoConhecimento | None:
        """Obtém um trecho conhecido para rotas metodológicas com proveniência explícita."""
        return next(
            (
                documento
                for documento in self.documentos
                if documento.identificador == identificador
            ),
            None,
        )
