from pathlib import Path

import pytest

from wordgen.application.inventario import construir_catalogo
from wordgen.domain.catalogo import Status
from wordgen.domain.tipos import Instrumento, Momento
from wordgen.infrastructure.leitura_csv import (
    ler_primeiras_linhas,
    listar_csvs,
    sha256_arquivo,
)

VOCAB = "Nome;Sexo;Q1;Q2;Q3;Total\nALUNO_1;M;1;0;1;2\n"
TDE = "Nome;P1;P2;P3\nALUNO_1;2;1;0\n"

ARQUIVOS = {
    "Fase2/Pré-teste/1 - Avaliação de Vocabulário/a.csv": VOCAB,
    "Fase2/Pós-teste/TDE II/b.csv": TDE,
    "Fase2/Pós-teste/TDE II/b_copia.csv": TDE,
    "Fase3/Pré-teste/Avaliação de Vocabulário/c__ted_ii.csv": TDE,
    "Fase3/Pré-teste/Avaliação de Vocabulário/d.csv": "Nome;Turma;Total\nALUNO_1;A;1\n",
    "Fase2/Pré-teste/TDE II/e.csv": VOCAB,
    "Fase2/Base de Dados/cadastro.csv": "Nome;Sexo\nALUNO_1;M\n",
    "Fase4/WordGen - Dados/f.csv": "Relatório;;\n;;\nEscola X;;\nNome;Q1;Q2\nALUNO_9;1;1\n",
}


@pytest.fixture
def catalogo(tmp_path: Path):
    for rel, conteudo in ARQUIVOS.items():
        destino = tmp_path / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(conteudo, encoding="utf-8-sig")
    registros = construir_catalogo(
        listar_csvs(tmp_path), tmp_path, ler_primeiras_linhas, sha256_arquivo
    )
    return {r.arquivo.split("/")[-1]: r for r in registros}


def test_todos_os_arquivos_entram_no_catalogo(catalogo):
    assert len(catalogo) == len(ARQUIVOS)


def test_vocabulario_ok(catalogo):
    r = catalogo["a.csv"]
    assert (r.instrumento, r.status, r.fase, r.momento) == (
        Instrumento.VOCABULARIO,
        Status.OK,
        2,
        Momento.PRE,
    )


def test_conteudo_define_arquivo_guardado_na_pasta_errada(catalogo):
    r = catalogo["c__ted_ii.csv"]
    assert r.instrumento is Instrumento.TDE
    assert r.status is Status.INFERIDO_PELO_CONTEUDO


def test_sem_itens_e_divergente_vao_para_revisao(catalogo):
    assert catalogo["d.csv"].status is Status.SEM_ITENS
    assert catalogo["e.csv"].status is Status.DIVERGENTE
    assert catalogo["e.csv"].instrumento is Instrumento.OUTRO


def test_cadastro_fica_como_outro(catalogo):
    assert catalogo["cadastro.csv"].instrumento is Instrumento.OUTRO
    assert catalogo["cadastro.csv"].status is Status.OUTRO


def test_copia_identica_aponta_para_o_original(catalogo):
    assert catalogo["b.csv"].duplicado_de is None
    assert catalogo["b_copia.csv"].duplicado_de.endswith("b.csv")


def test_fase4_sem_momento_e_cabecalho_deslocado(catalogo):
    r = catalogo["f.csv"]
    assert r.momento is Momento.DESCONHECIDO
    assert r.linha_cabecalho == 3
    assert r.n_itens_q == 2
