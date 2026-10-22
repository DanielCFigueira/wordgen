import pytest

from wordgen.domain.texto import normalizar


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("Avaliação de Vocabulário", "avaliacao de vocabulario"),
        ("Avaliação de vocabulário", "avaliacao de vocabulario"),
        ("  Pré-teste  ", "pre-teste"),
        ("TDE II  -  Subteste Escrita", "tde ii - subteste escrita"),
        ("Fase 2 - Dados\\Pós-teste", "fase 2 - dados/pos-teste"),
    ],
)
def test_normalizar(entrada, esperado):
    assert normalizar(entrada) == esperado
