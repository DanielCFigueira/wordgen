"""Normalização de texto para comparar nomes de pastas e arquivos."""

import re
import unicodedata


def normalizar(texto: str) -> str:
    """Minúsculas, sem acento, barras unificadas e espaços colapsados."""
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c)
    )
    minusculo = sem_acento.casefold().replace("\\", "/")
    return re.sub(r"\s+", " ", minusculo).strip()
