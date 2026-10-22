"""Leitura de baixo nível dos CSVs brutos (delimitador ";", UTF-8 com BOM)."""

import csv
import hashlib
from itertools import islice
from pathlib import Path

LINHAS_CABECALHO = 15


def listar_csvs(raiz: Path) -> list[Path]:
    return sorted(raiz.rglob("*.csv"))


def ler_primeiras_linhas(caminho: Path, n: int = LINHAS_CABECALHO) -> list[list[str]]:
    with caminho.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        return list(islice(csv.reader(f, delimiter=";"), n))


def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()
