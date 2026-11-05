"""Testes unitários das regras de agregação e privacidade do dashboard."""

import pandas as pd
import pytest

from wordgen.dashboard import dados as dados_dashboard
from wordgen.dashboard.analise import (
    agregar_demografia,
    associar_metadados_modelagem,
    filtrar_avaliacoes,
    normalizar_booleano,
    resumo_cobertura,
    resumo_modelagem,
)
from wordgen.dashboard.dados import _validar_pares, carregar_dados


def test_filtro_combinado_preserva_apenas_avaliacoes_no_recorte() -> None:
    dados = pd.DataFrame(
        {
            "fase": ["2", "2", "3"],
            "periodo": ["pre", "pos", "pre"],
            "serie": ["6", "6", "7"],
            "escola": ["A", "A", "B"],
            "turma": ["6A", "6A", "7A"],
        }
    )
    resultado = filtrar_avaliacoes(
        dados, fases=["2"], momentos=["pre"], series=["6"], escolas=["A"], turmas=["6A"]
    )
    assert len(resultado) == 1
    assert resultado.iloc[0]["periodo"] == "pre"


def test_combinacao_incompativel_produz_recorte_vazio() -> None:
    dados = pd.DataFrame({"fase": ["2"], "periodo": ["pre"]})
    resultado = filtrar_avaliacoes(dados, fases=["4"], momentos=["pos"])
    assert resultado.empty


def test_booleanos_de_texto_sao_convertidos_sem_verdade_por_nao_vazio() -> None:
    resultado = normalizar_booleano(
        pd.Series([True, False, "True", "False", "1", "0", "sim", "não", None, "inválido"])
    )
    esperado = [True, False, True, False, True, False, True, False, pd.NA, pd.NA]
    assert resultado.tolist() == esperado


def test_resumo_distingue_avaliacoes_estudantes_e_indicadores_ausentes() -> None:
    dados = pd.DataFrame(
        {"aluno_id": ["x", "x", "y", "z"], "completo": [True, "False", None, "inválido"]}
    )
    resumo = resumo_cobertura(dados, total_itens=50)
    assert resumo == {
        "avaliacoes": 4,
        "estudantes": 3,
        "completas": 1,
        "incompletas": 1,
        "sem_indicador": 2,
        "cobertura": 0.5,
        "itens_esperados": 50,
    }


def test_demografia_suprime_categoria_com_menos_de_dez_estudantes_mesmo_com_muitas_avaliacoes() -> (
    None
):
    avaliacoes = pd.DataFrame(
        {
            "aluno_id": [f"a{i}" for i in range(9) for _ in range(3)]
            + [f"b{i}" for i in range(10)],
            "nota": [10] * 27 + [20] * 10,
        }
    )
    estudantes = pd.DataFrame(
        {
            "aluno_id": [f"a{i}" for i in range(9)] + [f"b{i}" for i in range(10)],
            "zona": ["Rural"] * 9 + ["Urbana"] * 10,
        }
    )
    resumo = agregar_demografia(avaliacoes, estudantes, "zona", "nota")
    assert resumo.empty
    assert resumo.attrs["suprimido_por_privacidade"] is True


def test_demografia_exibe_apenas_quando_todas_categorias_tem_dez_ou_mais_estudantes() -> None:
    avaliacoes = pd.DataFrame(
        {
            "aluno_id": [f"a{i}" for i in range(10) for _ in range(2)]
            + [f"b{i}" for i in range(11)],
            "nota": [10] * 20 + [20] * 11,
        }
    )
    estudantes = pd.DataFrame(
        {
            "aluno_id": [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(11)],
            "zona": ["Rural"] * 10 + ["Urbana"] * 11,
        }
    )
    resumo = agregar_demografia(avaliacoes, estudantes, "zona", "nota")
    assert resumo.attrs["suprimido_por_privacidade"] is False
    assert resumo.set_index("zona")["n_estudantes"].to_dict() == {"Rural": 10, "Urbana": 11}
    assert resumo.set_index("zona")["media"].to_dict() == {"Rural": 10.0, "Urbana": 20.0}


def test_demografia_rejeita_chave_duplicada_na_dimensao() -> None:
    avaliacoes = pd.DataFrame({"aluno_id": ["a"], "nota": [10]})
    estudantes = pd.DataFrame({"aluno_id": ["a", "a"], "zona": ["Urbana", "Rural"]})
    with pytest.raises(ValueError, match="um registro não nulo por aluno_id"):
        agregar_demografia(avaliacoes, estudantes, "zona", "nota")


def test_associacao_de_metadados_respeita_instrumento_e_metricas() -> None:
    metricas = pd.DataFrame(
        {
            "Cenário": [
                "Vocabulário — Experimento A (Acadêmico)",
                "TDE II — Experimento A (Acadêmico)",
            ],
            "Modelo": ["Random Forest", "Random Forest"],
            "MAE": [1.2, 2.3],
            "RMSE": [1.5, 2.8],
            "R²": [0.1, 0.2],
        }
    )
    metadados = {
        "vocabulario_exp_a_rf": {
            "instrumento": "Vocabulário",
            "experimento": "A",
            "modelo": "RandomForestRegressor",
            "metricas_teste": {"MAE": 1.2, "RMSE": 1.5, "R2": 0.1},
            "cv_mae_medio": 1.1,
            "amostras_treino": 30,
            "amostras_teste": 10,
        },
        "tde_exp_a_rf": {
            "instrumento": "TDE II",
            "experimento": "A",
            "modelo": "RandomForestRegressor",
            "metricas_teste": {"MAE": 2.3, "RMSE": 2.8, "R2": 0.2},
            "cv_mae_medio": 2.1,
            "amostras_treino": 30,
            "amostras_teste": 10,
        },
        "vocabulario_exp_b_mlp": {
            "instrumento": "Vocabulário",
            "experimento": "B",
            "algoritmo": "MLPRegressor",
            "MAE_teste": 9,
            "RMSE_teste": 9,
            "R2_teste": 9,
        },
    }
    associados, avisos = associar_metadados_modelagem(metadados, metricas, "Vocabulário")
    assert associados[["Artefato", "Instrumento"]].to_dict("records") == [
        {"Artefato": "vocabulario_exp_a_rf", "Instrumento": "Vocabulário"}
    ]
    assert len(avisos) == 1


def _pares_validos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aluno_id": ["a", "a", "b"],
            "fase": ["2", "3", "2"],
            "escola": ["A", "A", "B"],
            "serie": ["6", "7", "6"],
            "turma": ["6A", "7A", "6B"],
            "total_pre": [10, 11, 12],
            "total_pos": [12, 13, 14],
            "completo_pre": [True, True, True],
            "completo_pos": [True, True, True],
        }
    )


def test_validacao_pareada_aceita_mesmo_estudante_em_fases_distintas() -> None:
    _validar_pares(_pares_validos(), "pareado", 50)


@pytest.mark.parametrize("coluna", ["aluno_id", "fase"])
def test_validacao_pareada_rejeita_nulos_na_chave(coluna: str) -> None:
    dados = _pares_validos()
    dados.loc[0, coluna] = None
    with pytest.raises(ValueError, match="chave pareada"):
        _validar_pares(dados, "pareado", 50)


def test_validacao_pareada_rejeita_duplicidade_na_chave_natural() -> None:
    dados = _pares_validos()
    dados.loc[1, "fase"] = "2"
    with pytest.raises(ValueError, match="chave pareada duplicada"):
        _validar_pares(dados, "pareado", 50)


def test_validacao_pareada_rejeita_par_completo_sem_total() -> None:
    dados = _pares_validos()
    dados.loc[0, "total_pos"] = None
    with pytest.raises(ValueError, match="par completo sem total"):
        _validar_pares(dados, "pareado", 50)


def test_validacao_pareada_rejeita_indicador_de_completude_invalido() -> None:
    dados = _pares_validos().astype({"completo_pre": "object"})
    dados.loc[0, "completo_pre"] = "desconhecido"
    with pytest.raises(ValueError, match="indicador de completude inválido"):
        _validar_pares(dados, "pareado", 50)


def test_resumo_modelagem_prioriza_mae_e_compara_com_baseline() -> None:
    metricas = pd.DataFrame(
        {
            "Cenário": [
                "Vocabulário — Experimento A (Acadêmico)",
                "Vocabulário — Experimento A (Acadêmico)",
                "Vocabulário — Experimento A (Acadêmico)",
            ],
            "Modelo": ["Baseline 2 (Média Treino)", "Random Forest", "Rede Neural (MLP)"],
            "MAE": [5.0, 4.0, 4.2],
            "RMSE": [6.0, 5.0, 5.1],
            "R²": [0.0, 0.1, 0.3],
        }
    )
    resumo = resumo_modelagem(metricas)
    assert resumo.iloc[0]["Modelo"] == "Random Forest"
    assert resumo.iloc[0]["Reducao MAE (%)"] == pytest.approx(20.0)


def test_modo_demonstracao_gera_esquemas_e_pares_consistentes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(dados_dashboard, "FINAL_DIR", tmp_path / "sem_dados_reais")
    dados = carregar_dados()

    assert dados["modo_demonstracao"] is True
    assert set(dados["vocabulario"].columns).issuperset(
        {"aluno_id", "fase", "periodo", "total_vocabulario", "Q01", "Q50"}
    )
    assert set(dados["tde"].columns).issuperset(
        {"aluno_id", "fase", "periodo", "total_tde", "P01", "P40"}
    )
    assert dados["vocabulario"]["total_vocabulario"].between(0, 50).all()
    assert dados["tde"]["total_tde"].between(0, 80).all()
    itens_vocabulario = dados["vocabulario"].filter(regex=r"^Q\d{2}$")
    itens_tde = dados["tde"].filter(regex=r"^P\d{2}$")
    assert itens_vocabulario.shape[1] == 50
    assert itens_tde.shape[1] == 40
    assert set(itens_vocabulario.stack().unique()).issubset({0, 1})
    assert set(itens_tde.stack().unique()).issubset({0, 1, 2})
    for nome, maximo in (("pareado_vocabulario", 50), ("pareado_tde", 80)):
        pares = dados[nome]
        _validar_pares(pares, nome, maximo)
        assert not pares.duplicated(["aluno_id", "fase"]).any()
        assert pares["aluno_id"].isin(dados["dim_aluno"]["aluno_id"]).all()


def test_carregamento_real_permanece_prioritario_quando_os_cinco_arquivos_existem(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    final = tmp_path / "final"
    final.mkdir()
    demonstracao = dados_dashboard._gerar_dados_demonstracao()
    demonstracao["vocabulario"].to_csv(final / "dataset_vocabulario.csv", index=False)
    demonstracao["tde"].to_csv(final / "dataset_tde.csv", index=False)
    demonstracao["pareado_vocabulario"].to_csv(final / "pareado_vocabulario.csv", index=False)
    demonstracao["pareado_tde"].to_csv(final / "pareado_tde.csv", index=False)
    demonstracao["dim_aluno"].to_csv(final / "dim_aluno.csv", index=False)
    monkeypatch.setattr(dados_dashboard, "FINAL_DIR", final)

    dados = carregar_dados()

    assert dados["modo_demonstracao"] is False
    assert len(dados["vocabulario"]) == len(demonstracao["vocabulario"])
