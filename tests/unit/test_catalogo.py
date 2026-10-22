from wordgen.domain.catalogo import Status, analisar_cabecalho, decidir_instrumento
from wordgen.domain.tipos import Instrumento


def test_cabecalho_na_primeira_linha():
    analise = analisar_cabecalho([["Nome", "Sexo", "Q1", "Q2", "Q3", "Total"]])
    assert analise.linha == 0
    assert (analise.n_itens_q, analise.n_itens_p) == (3, 0)
    assert analise.colunas_meta == ("nome", "sexo", "total")
    assert not analise.itens_repetidos


def test_cabecalho_deslocado_para_baixo():
    linhas = [["Relatório"], [], ["Escola: X"], ["Nome", "P1", "P2"]]
    analise = analisar_cabecalho(linhas)
    assert analise.linha == 3
    assert (analise.n_itens_q, analise.n_itens_p) == (0, 2)


def test_itens_repetidos_sao_sinalizados():
    analise = analisar_cabecalho([["Nome", "P1", "P2", "P1", "P2"]])
    assert analise.n_itens_p == 4
    assert analise.itens_repetidos


def test_sem_itens_usa_primeira_linha():
    analise = analisar_cabecalho([["Nome", "Turma"], ["a", "b"]])
    assert analise.linha == 0
    assert (analise.n_itens_q, analise.n_itens_p) == (0, 0)


def test_arquivo_vazio_nao_quebra():
    analise = analisar_cabecalho([])
    assert (analise.n_itens_q, analise.n_itens_p) == (0, 0)


def _analise(q, p):
    colunas = ["Nome", *[f"Q{i}" for i in range(q)], *[f"P{i}" for i in range(p)]]
    return analisar_cabecalho([colunas])


def test_decisao_caminho_e_conteudo_concordam():
    assert decidir_instrumento(Instrumento.VOCABULARIO, _analise(3, 0)) == (
        Instrumento.VOCABULARIO,
        Status.OK,
    )


def test_decisao_conteudo_define_quando_caminho_e_ambiguo():
    assert decidir_instrumento(Instrumento.OUTRO, _analise(0, 3)) == (
        Instrumento.TDE,
        Status.INFERIDO_PELO_CONTEUDO,
    )


def test_decisao_divergente_vai_para_revisao():
    assert decidir_instrumento(Instrumento.TDE, _analise(3, 0)) == (
        Instrumento.OUTRO,
        Status.DIVERGENTE,
    )


def test_decisao_cabecalho_misto_e_divergente():
    assert decidir_instrumento(Instrumento.TDE, _analise(2, 2)) == (
        Instrumento.OUTRO,
        Status.DIVERGENTE,
    )


def test_decisao_sem_itens():
    assert decidir_instrumento(Instrumento.VOCABULARIO, _analise(0, 0)) == (
        Instrumento.OUTRO,
        Status.SEM_ITENS,
    )
    assert decidir_instrumento(Instrumento.OUTRO, _analise(0, 0)) == (
        Instrumento.OUTRO,
        Status.OUTRO,
    )
