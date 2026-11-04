"""Configuração local e segura do agente WordGen."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ConfiguracaoAgente:
    """Caminhos dos artefatos publicados e configuração não secreta do agente."""

    raiz: Path = ROOT
    modelo_groq: str = "openai/gpt-oss-20b"

    @property
    def diretorio_rag(self) -> Path:
        return self.raiz / "data" / "rag"

    @property
    def base_conhecimento(self) -> Path:
        return self.diretorio_rag / "base_conhecimento_wordgen.json"

    @property
    def indice_vetorial(self) -> Path:
        return self.diretorio_rag / "indice_tfidf_wordgen.joblib"

    @property
    def benchmark(self) -> Path:
        return self.diretorio_rag / "benchmark_qa_ground_truth.json"

    @property
    def diretorio_analytics(self) -> Path:
        return self.raiz / "data" / "analytics"

    @property
    def metricas_modelagem(self) -> Path:
        return self.raiz / "reports" / "modelagem" / "metricas" / "metricas_comparativas_gerais.csv"


def obter_chave_groq() -> str | None:
    """Lê a chave exclusivamente do ambiente, sem nunca registrá-la."""
    load_dotenv()
    chave = os.getenv("GROQ_API_KEY", "").strip()
    return chave or None


def obter_modelo_groq() -> str:
    """Obtém o modelo opcionalmente configurado, com padrão documentado."""
    load_dotenv()
    return os.getenv("GROQ_MODEL", "").strip() or ConfiguracaoAgente().modelo_groq
