"""Testes unitários para o módulo de modelagem preditiva."""

import numpy as np
import pandas as pd
import pytest

from wordgen.modelagem.avaliacao import calcular_metricas
from wordgen.modelagem.baselines import BaselineGanhoZero, BaselineMediaTreino
from wordgen.modelagem.pipelines import criar_pipeline_mlp, criar_pipeline_rf
from wordgen.modelagem.preparacao import preparar_base_instrumento
from wordgen.modelagem.validacao import dividir_treino_teste_agrupado, obter_cv_agrupado


@pytest.fixture
def df_exemplo():
    """Gera um DataFrame sintético para testes unitários."""
    np.random.seed(42)
    n = 100
    alunos = [f"ALUNO_{i:03d}" for i in range(25) for _ in range(4)]

    return pd.DataFrame(
        {
            "aluno_id": alunos,
            "fase": np.random.choice(["Fase 2", "Fase 3", "Fase 4"], n),
            "escola": np.random.choice(["Escola A", "Escola B"], n),
            "serie": np.random.choice(["6", "7", "8", "9"], n),
            "turma": np.random.choice(["6A", "7B", "8A", "9C"], n),
            "total_pre": np.random.uniform(10, 40, n),
            "ganho": np.random.normal(1.0, 3.0, n),
        }
    )


@pytest.fixture
def df_pareado_sintetico():
    """Gera um DataFrame pareado sintético (Exp A/B) para testes de preparação."""
    np.random.seed(42)
    n = 60
    alunos = [f"ALUNO_{i:03d}" for i in range(20) for _ in range(3)]
    return pd.DataFrame(
        {
            "aluno_id": alunos,
            "fase": np.random.choice(["Fase 2", "Fase 3", "Fase 4"], n),
            "escola": np.random.choice(["Escola A", "Escola B"], n),
            "serie": np.random.choice([6.0, 7.0, 8.0, 9.0], n),
            "turma": np.random.choice(["A", "B", "C"], n),
            "total_pre": np.random.uniform(5, 45, n),
            "total_pos": np.random.uniform(5, 45, n),
            "completo_pre": "True",
            "completo_pos": "True",
        }
    )


@pytest.fixture
def df_dim_aluno_sintetico():
    """Gera uma dim_aluno sintética sem duplicatas em aluno_id."""
    alunos_unicos = [f"ALUNO_{i:03d}" for i in range(20)]
    return pd.DataFrame(
        {
            "aluno_id": alunos_unicos,
            "sexo": np.random.choice(["M", "F"], 20),
            "faixa_etaria": np.random.choice(["11-12", "13-14", "15+"], 20),
            "cor_raca": np.random.choice(["PARDA", "BRANCA", "PRETA"], 20),
            "zona": np.random.choice(["URBANA", "RURAL"], 20),
            "necessidade_especial": np.random.choice(["NAO_REGISTRADO", "SIM"], 20),
            "fonte_socio": "teste",
        }
    )


def test_baselines_comportamento(df_exemplo):
    """Verifica se os baselines realizam previsões corretas conforme especificação."""
    X = df_exemplo[["total_pre", "fase", "escola"]]
    y = df_exemplo["ganho"]

    b_zero = BaselineGanhoZero()
    b_zero.fit(X, y)
    pred_zero = b_zero.predict(X)
    assert len(pred_zero) == len(X)
    assert np.all(pred_zero == 0.0)

    b_media = BaselineMediaTreino()
    b_media.fit(X, y)
    pred_media = b_media.predict(X)
    assert len(pred_media) == len(X)
    assert np.isclose(pred_media[0], float(y.mean()))
    assert np.all(pred_media == float(y.mean()))


def test_divisao_agrupada_sem_vazamento(df_exemplo):
    """Garante que a divisão agrupada não compartilhe nenhum aluno entre treino e teste."""
    X_train, X_test, y_train, y_test, g_train, g_test = dividir_treino_teste_agrupado(
        df_exemplo, target_col="ganho", grupo_col="aluno_id", test_size=0.20, random_state=42
    )

    alunos_treino = set(g_train)
    alunos_teste = set(g_test)

    assert len(alunos_treino.intersection(alunos_teste)) == 0
    assert len(X_train) + len(X_test) == len(df_exemplo)
    assert len(y_train) + len(y_test) == len(df_exemplo)


def test_groupkfold_nao_separa_registros_do_mesmo_aluno(df_exemplo):
    grupos = df_exemplo["aluno_id"].to_numpy()
    for train_idx, valid_idx in obter_cv_agrupado(n_splits=5).split(
        df_exemplo, df_exemplo["ganho"], groups=grupos
    ):
        assert set(grupos[train_idx]).isdisjoint(set(grupos[valid_idx]))


def test_calcular_metricas():
    """Valida o cálculo das métricas MAE, RMSE e R²."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 31.0])

    m = calcular_metricas(y_true, y_pred)
    assert "MAE" in m
    assert "RMSE" in m
    assert "R2" in m
    assert np.isclose(m["MAE"], (2.0 + 2.0 + 1.0) / 3.0)


def test_pipelines_execucao(df_exemplo):
    """Verifica se os pipelines Scikit-learn (RF e MLP) realizam fit e predict sem erro."""
    X_train, X_test, y_train, y_test, _, _ = dividir_treino_teste_agrupado(
        df_exemplo, target_col="ganho", grupo_col="aluno_id", test_size=0.20, random_state=42
    )

    num_cols = ["total_pre"]
    cat_cols = ["fase", "escola", "serie", "turma"]

    rf_pipe = criar_pipeline_rf(num_cols, cat_cols, rf_params={"n_estimators": 5, "max_depth": 3})
    rf_pipe.fit(X_train, y_train)
    rf_preds = rf_pipe.predict(X_test)
    assert len(rf_preds) == len(X_test)

    mlp_pipe = criar_pipeline_mlp(
        num_cols, cat_cols, mlp_params={"hidden_layer_sizes": (8,), "max_iter": 20}
    )
    mlp_pipe.fit(X_train, y_train)
    mlp_preds = mlp_pipe.predict(X_test)
    assert len(mlp_preds) == len(X_test)


def test_mlp_sem_early_stopping():
    """Garante que o MLPRegressor padrão não usa early_stopping (evitar vazamento interno)."""
    from wordgen.modelagem.pipelines import criar_pipeline_mlp

    pipe = criar_pipeline_mlp(["total_pre"], ["fase"])
    mlp = pipe.named_steps["regressor"]
    assert mlp.early_stopping is False, (
        "early_stopping deve ser False para não criar divisão interna "
        "sem respeitar agrupamento por aluno_id."
    )
    assert mlp.max_iter >= 500, "max_iter deve ser >= 500 para garantir convergência."


def test_integridade_merge_dim_aluno(df_pareado_sintetico, df_dim_aluno_sintetico):
    """Garante que o merge com dim_aluno não multiplica linhas (many_to_one).

    A validação explicit `validate='many_to_one'` no merge é necessária para detectar
    silenciosamente duplicatas em aluno_id na dimensão que multiplicariam registros.
    """
    n_antes = len(df_pareado_sintetico)

    # Deve executar sem levantar exceção (dim sem duplicatas em aluno_id)
    merged = df_pareado_sintetico.merge(
        df_dim_aluno_sintetico,
        on="aluno_id",
        how="left",
        validate="many_to_one",
    )

    assert len(merged) == n_antes, (
        f"O merge multiplicou registros: {n_antes} antes → {len(merged)} depois. "
        "Verificar duplicatas em aluno_id na dim_aluno."
    )


def test_merge_dim_aluno_com_duplicatas_detectado(df_pareado_sintetico, df_dim_aluno_sintetico):
    """Garante que duplicatas em dim_aluno são detectadas pelo validate='many_to_one'."""
    # Introduz duplicata artificial em aluno_id
    duplicata = df_dim_aluno_sintetico.iloc[[0]].copy()
    dim_com_dup = pd.concat([df_dim_aluno_sintetico, duplicata], ignore_index=True)

    with pytest.raises(pd.errors.MergeError):
        df_pareado_sintetico.merge(
            dim_com_dup,
            on="aluno_id",
            how="left",
            validate="many_to_one",
        )


def test_preparacao_exp_b_rejeita_id_nulo_na_dimensao(df_pareado_sintetico, df_dim_aluno_sintetico):
    dim_com_id_nulo = df_dim_aluno_sintetico.copy()
    dim_com_id_nulo.loc[0, "aluno_id"] = None

    with pytest.raises(ValueError, match="aluno_id nulo"):
        preparar_base_instrumento(
            df_pareado=df_pareado_sintetico,
            df_dim_aluno=dim_com_id_nulo,
            instrumento="vocabulario",
            experimento="B",
        )


def test_preparacao_exp_b_falha_com_chave_duplicada_na_dimensao(
    df_pareado_sintetico, df_dim_aluno_sintetico
):
    dim_duplicada = pd.concat(
        [df_dim_aluno_sintetico, df_dim_aluno_sintetico.iloc[[0]]], ignore_index=True
    )

    with pytest.raises(pd.errors.MergeError):
        preparar_base_instrumento(
            df_pareado=df_pareado_sintetico,
            df_dim_aluno=dim_duplicada,
            instrumento="vocabulario",
            experimento="B",
        )


def test_preparacao_exp_b_sem_multiplicacao(df_pareado_sintetico, df_dim_aluno_sintetico):
    """Verifica que preparar_base_instrumento (Exp B) não multiplica linhas após merge."""
    n_antes = len(df_pareado_sintetico)

    df_resultado = preparar_base_instrumento(
        df_pareado=df_pareado_sintetico,
        df_dim_aluno=df_dim_aluno_sintetico,
        instrumento="vocabulario",
        experimento="B",
    )

    # Resultado deve ter <= n_antes (pode ser menor por filtros de completude, nunca maior)
    assert len(df_resultado) <= n_antes, (
        f"preparar_base_instrumento retornou mais linhas ({len(df_resultado)}) "
        f"do que o input ({n_antes}). Possível multiplicação por merge."
    )
    assert "ganho" in df_resultado.columns
    assert "sexo" in df_resultado.columns
    assert "aluno_id" in df_resultado.columns


def test_preparacao_exp_b_imputa_aluno_ausente_na_dimensao(
    df_pareado_sintetico, df_dim_aluno_sintetico
):
    dim_sem_primeiro_aluno = df_dim_aluno_sintetico.iloc[1:].copy()
    resultado = preparar_base_instrumento(
        df_pareado=df_pareado_sintetico,
        df_dim_aluno=dim_sem_primeiro_aluno,
        instrumento="vocabulario",
        experimento="B",
    )
    nao_encontrado = resultado[resultado["aluno_id"] == "ALUNO_000"]

    assert len(resultado) == len(df_pareado_sintetico)
    assert nao_encontrado["sexo"].eq("NAO_INFORMADO").all()
    assert nao_encontrado["faixa_etaria"].eq("NAO INFORMADA").all()
