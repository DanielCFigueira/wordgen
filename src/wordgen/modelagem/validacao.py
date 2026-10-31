"""Módulo de validação estatística e divisão por grupos de estudantes."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def dividir_treino_teste_agrupado(
    df: pd.DataFrame,
    target_col: str = "ganho",
    grupo_col: str = "aluno_id",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Realiza a separação agrupada entre treino e teste por aluno_id.

    Garante 100% de isolamento: nenhum estudante aparece em treino e teste ao mesmo tempo.

    Retorna:
    --------
    X_train, X_test, y_train, y_test, groups_train, groups_test
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[grupo_col].values

    feature_cols = [c for c in df.columns if c not in [target_col, grupo_col]]
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    # Validação rigorosa de vazamento
    alunos_treino = set(groups_train)
    alunos_teste = set(groups_test)
    intersecao = alunos_treino.intersection(alunos_teste)
    assert len(intersecao) == 0, (
        f"VAZAMENTO DETECTADO! {len(intersecao)} alunos presentes em treino e teste."
    )

    return X_train, X_test, y_train, y_test, groups_train, groups_test


def obter_cv_agrupado(n_splits: int = 5) -> GroupKFold:
    """Retorna um gerador de validação cruzada agrupada para o conjunto de treinamento."""
    return GroupKFold(n_splits=n_splits)
