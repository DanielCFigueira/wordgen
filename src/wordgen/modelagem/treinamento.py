"""Módulo de treinamento e otimização de hiperparâmetros com validação agrupada."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from wordgen.modelagem.pipelines import criar_pipeline_mlp, criar_pipeline_rf
from wordgen.modelagem.validacao import obter_cv_agrupado


def treinar_e_otimizar_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: np.ndarray,
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[any, dict]:
    """Otimiza hiperparâmetros do RandomForestRegressor via GridSearchCV com GroupKFold."""
    base_pipeline = criar_pipeline_rf(
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        random_state=random_state,
    )

    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [6, 10, None],
        "regressor__min_samples_leaf": [2, 5],
    }

    gkf = obter_cv_agrupado(n_splits=cv_splits)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train, groups=groups_train)

    melhores_params = grid.best_params_
    melhor_score = -grid.best_score_  # Converte de negativo para MAE positivo

    info_treino = {
        "melhores_parametros": melhores_params,
        "cv_mae_medio": melhor_score,
        "cv_splits": cv_splits,
    }
    return grid.best_estimator_, info_treino


def treinar_e_otimizar_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: np.ndarray,
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[any, dict]:
    """Otimiza hiperparâmetros da Rede Neural MLPRegressor via GridSearchCV com GroupKFold."""
    base_pipeline = criar_pipeline_mlp(
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        random_state=random_state,
    )

    param_grid = {
        "regressor__hidden_layer_sizes": [(64, 32), (32, 16), (64,)],
        "regressor__alpha": [0.001, 0.01, 0.1],
        "regressor__learning_rate_init": [0.005, 0.01],
    }

    gkf = obter_cv_agrupado(n_splits=cv_splits)

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train, groups=groups_train)

    melhores_params = grid.best_params_
    melhor_score = -grid.best_score_

    info_treino = {
        "melhores_parametros": melhores_params,
        "cv_mae_medio": melhor_score,
        "cv_splits": cv_splits,
    }
    return grid.best_estimator_, info_treino
