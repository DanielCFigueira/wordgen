"""Módulo de baselines de referência para avaliação comparativa."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class BaselineGanhoZero(BaseEstimator, RegressorMixin):
    """Baseline 1: Previsão ingênua de que todo estudante terá ganho igual a zero.

    y_pred = 0 para todas as instâncias.
    """

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaselineGanhoZero":
        """Ajusta o baseline (sem parâmetros a aprender além de registrar o fit)."""
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Retorna array com zeros para todas as amostras de entrada."""
        n_samples = len(X)
        return np.zeros(n_samples, dtype=float)


class BaselineMediaTreino(BaseEstimator, RegressorMixin):
    """Baseline 2: Previsão da média aritmética do ganho observada no conjunto de treino.

    y_pred = mean(y_train). Não utiliza informações do conjunto de teste.
    """

    def __init__(self) -> None:
        self.media_treino_: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BaselineMediaTreino":
        """Calcula e armazena a média do target no conjunto de treino."""
        self.media_treino_ = float(np.mean(y))
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Retorna a média aprendida no treino para todas as amostras."""
        if self.media_treino_ is None:
            raise RuntimeError("O modelo ainda não foi ajustado com fit().")
        n_samples = len(X)
        return np.full(n_samples, self.media_treino_, dtype=float)
