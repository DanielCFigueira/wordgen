"""Testes unitários para regras de padronização do domínio."""

import pytest

from wordgen.domain.padronizacao import (
    classificar_celula,
    faixa_etaria,
    normalizar_id,
    normalizar_sexo,
    padronizar_escola,
    padronizar_turma,
    parse_idade,
)
from wordgen.domain.tipos import Instrumento


@pytest.mark.parametrize(
    ("valor", "max_val", "esperado_num", "esperada_flag"),
    [
        (0, 1, 0.0, "ok"),
        (1, 1, 1.0, "ok"),
        ("1", 1, 1.0, "ok"),
        ("1.0", 1, 1.0, "ok"),
        (" 1 ", 1, 1.0, "ok"),
        (2, 2, 2.0, "ok"),
        (2, Instrumento.TDE, 2.0, "ok"),
        (1, Instrumento.VOCABULARIO, 1.0, "ok"),
        # Vazio
        (None, 1, None, "vazio"),
        ("", 1, None, "vazio"),
        ("   ", 1, None, "vazio"),
        ("-", 1, None, "vazio"),
        ("—", 1, None, "vazio"),
        ("nan", 1, None, "vazio"),
        ("null", 1, None, "vazio"),
        # Fora de escala
        (2, 1, None, "fora_escala"),
        (9, 1, None, "fora_escala"),
        (11, 2, None, "fora_escala"),
        (-1, 1, None, "fora_escala"),
        (1.5, 2, None, "fora_escala"),
        # Erro de planilha
        ("#REF!", 2, None, "erro_planilha"),
        ("#N/A", 1, None, "erro_planilha"),
        ("#VALOR!", 1, None, "erro_planilha"),
        # Texto
        ("A", 1, None, "texto"),
        ("D", 1, None, "texto"),
        ("ausente", 1, None, "texto"),
        (True, 1, None, "texto"),
    ],
)
def test_classificar_celula(valor, max_val, esperado_num, esperada_flag):
    num, flag = classificar_celula(valor, max_val)
    assert flag == esperada_flag
    assert num == esperado_num


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("ALUNO_0123456789ABCDEF", "ALUNO_0123456789ABCDEF"),
        (" aluno_0123456789abcdef ", "ALUNO_0123456789ABCDEF"),
        ("ALUNO_123", None),
        ("GABRIEL", None),
        (None, None),
        ("", None),
    ],
)
def test_normalizar_id(entrada, esperado):
    assert normalizar_id(entrada) == esperado


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("M", "M"),
        ("m", "M"),
        ("Masculino", "M"),
        ("MASC", "M"),
        ("F", "F"),
        ("Feminino", "F"),
        ("FEM", "F"),
        ("0", "NAO INFORMADO"),
        (None, "NAO INFORMADO"),
        ("", "NAO INFORMADO"),
    ],
)
def test_normalizar_sexo(entrada, esperado):
    assert normalizar_sexo(entrada) == esperado


@pytest.mark.parametrize(
    ("idade", "data_nasc", "esperado"),
    [
        (12, None, 12),
        ("14", None, 14),
        ("10.0", None, 10),
        (5, None, None),  # fora do intervalo 8-20
        (25, None, None),
        (None, "40544", 13),  # ano ~2011 -> 2024 - 2011 = 13
        (None, "2010-05-15", 14),
        (None, None, None),
    ],
)
def test_parse_idade(idade, data_nasc, esperado):
    assert parse_idade(idade, data_nasc) == esperado


@pytest.mark.parametrize(
    ("idade", "esperado"),
    [
        (9, "<=10"),
        (10, "<=10"),
        (11, "11-12"),
        (12, "11-12"),
        (13, "13-14"),
        (14, "13-14"),
        (15, "15+"),
        (18, "15+"),
        (None, "NAO INFORMADA"),
    ],
)
def test_faixa_etaria(idade, esperado):
    assert faixa_etaria(idade) == esperado


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        (
            "EMEB PROF. RICARDO VIEIRA DE LIMA",
            "EMEB PROFESSOR RICARDO VIEIRA DE LIMA",
        ),
        (
            "EMEB PROFESSOR RICARDO VIEIRA DE LIMA 9º ANO B",
            "EMEB PROFESSOR RICARDO VIEIRA DE LIMA",
        ),
        (
            "EMEB PROFA. MARIA QUEIROZ FERRO",
            "EMEB PROFESSORA MARIA QUEIROZ FERRO",
        ),
        (
            "EMEB VER. EXPEDITO PORFÍRIO",
            "EMEB EXPEDITO PORFÍRIO DOS SANTOS",
        ),
        (
            "EMEF PADRE JOSÉ DOS SANTOS MOUSINHO",
            "EMEF PADRE JOSÉ DOS SANTOS MOUSINHO",
        ),
        ("EMEB PADRE ANCHIETA", "EMEB PADRE ANCHIETA"),
        ("EMEB NATANAEL DA SILVA", "EMEB NATANAEL DA SILVA"),
        ("1.0", None),
        ("0.0", None),
        ("//////////////", None),
        ("GABRIEL BRITO", None),
        (None, None),
    ],
)
def test_padronizar_escola(entrada, esperado):
    assert padronizar_escola(entrada) == esperado


@pytest.mark.parametrize(
    ("entrada", "serie_esp", "letra_esp", "turma_esp"),
    [
        ("6A", 6, "A", "6A"),
        ("7º B", 7, "B", "7B"),
        ("8º ANO - C", 8, "C", "8C"),
        ("8º ANO", 8, "U", "8U"),
        ("6º período (7º ano)", 7, "U", "7U"),
        ("6º período", None, None, None),
        ("10C", None, None, None),
        ("0.0", None, None, None),
        (None, None, None, None),
    ],
)
def test_padronizar_turma(entrada, serie_esp, letra_esp, turma_esp):
    serie, letra, turma = padronizar_turma(entrada)
    assert serie == serie_esp
    assert letra == letra_esp
    assert turma == turma_esp
