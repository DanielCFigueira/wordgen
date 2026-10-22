import pytest

from wordgen.domain.classificacao import (
    fase_por_caminho,
    instrumento_por_caminho,
    momento_por_caminho,
)
from wordgen.domain.tipos import Instrumento, Momento

CASOS_INSTRUMENTO = [
    (
        "Fase2/Fase 2 - Dados/Pré-teste/1 - Avaliação de Vocabulário/Turma A.csv",
        Instrumento.VOCABULARIO,
    ),
    (
        "Fase2/Fase 2 - Dados/Pós-teste/TDE II - Subteste de escrita/Turma A.csv",
        Instrumento.TDE,
    ),
    (
        "Fase3/Pré-teste/TDE II/Seleção STARI TDE II.csv",
        Instrumento.TDE,
    ),
    (
        "Fase4/WordGen - Dados/Fase 4 - Monitor X__prvl_ted_ii.csv",
        Instrumento.TDE,
    ),
    (
        "Fase4/WordGen - Dados/Fase 4 - Monitor X__prvl_vocabulario.csv",
        Instrumento.VOCABULARIO,
    ),
    (
        "Fase2/Fase 2 - Dados/Base de Dados/CadastroAlunos.csv",
        Instrumento.OUTRO,
    ),
    (
        "Fase3/Vocabulário e TDE juntos.csv",
        Instrumento.OUTRO,
    ),
]

CASOS_FASE = [
    ("Fase2/Fase 2 - Dados/x.csv", 2),
    ("Fase3/Pré-teste/x.csv", 3),
    ("Fase4/WordGen - Dados/Fase 4 (2024.2) - Monitor.csv", 4),
    ("outra/pasta/x.csv", None),
]

CASOS_MOMENTO = [
    ("Fase2/Fase 2 - Dados/Pré-teste/x.csv", Momento.PRE),
    ("Fase2/Fase 2 - Dados/Pós-teste/x.csv", Momento.POS),
    ("Fase3/Pré-teste/TDE II/result-pretest-TDE-v1__summary.csv", Momento.PRE),
    ("Fase4/WordGen - Dados/PÓS TESTE - Escola X - Fase 4.csv", Momento.POS),
    (
        "Fase4/WordGen - Dados/Fase 4 - Monitor X__prvl_vocabulario.csv",
        Momento.DESCONHECIDO,
    ),
    (
        "Fase4/WordGen - Dados/TDE II - Préteste e Pósteste.csv",
        Momento.DESCONHECIDO,
    ),
]


@pytest.mark.parametrize(("caminho", "esperado"), CASOS_INSTRUMENTO)
def test_instrumento_por_caminho(caminho, esperado):
    assert instrumento_por_caminho(caminho) == esperado


@pytest.mark.parametrize(("caminho", "esperado"), CASOS_FASE)
def test_fase_por_caminho(caminho, esperado):
    assert fase_por_caminho(caminho) == esperado


@pytest.mark.parametrize(("caminho", "esperado"), CASOS_MOMENTO)
def test_momento_por_caminho(caminho, esperado):
    assert momento_por_caminho(caminho) == esperado
