"""Testes unitários para o módulo de modelagem preditiva."""

import numpy as np
import pandas as pd
import pytest

from wordgen.modelagem.avaliacao import calcular_metricas
from wordgen.modelagem.baselines import BaselineGanhoZero, BaselineMediaTreino
from wordgen.modelagem.pipelines import criar_pipeline_mlp, criar_pipeline_rf
from wordgen.modelagem.validacao import dividir_treino_teste_agrupado


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
