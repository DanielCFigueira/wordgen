"""Módulo de avaliação de desempenho, métricas e análise de resíduos."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_metricas(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calcula MAE, RMSE e R² com rigor estatístico."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    # root_mean_squared_error disponível diretamente ou sqrt do mse
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def avaliar_modelo(
    modelo,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    nome_modelo: str,
    cenario: str,
) -> dict:
    """Gera diagnóstico de avaliação completo para um modelo ajustado."""
    y_pred = modelo.predict(X_test)
    metricas = calcular_metricas(y_test, y_pred)

    residuos = y_test.values - y_pred

    diagnostico = {
        "modelo": nome_modelo,
        "cenario": cenario,
        "MAE": metricas["MAE"],
        "RMSE": metricas["RMSE"],
        "R2": metricas["R2"],
        "media_real": float(np.mean(y_test)),
        "media_pred": float(np.mean(y_pred)),
        "desvio_padrao_residuos": float(np.std(residuos)),
        "resíduo_min": float(np.min(residuos)),
        "resíduo_max": float(np.max(residuos)),
        "previsoes": y_pred,
        "residuos": residuos,
    }
    return diagnostico


def consolidar_tabela_metricas(resultados: list[dict]) -> pd.DataFrame:
    """Consolida uma lista de diagnósticos em um DataFrame para relatório."""
    linhas = []
    for r in resultados:
        linhas.append(
            {
                "Cenário": r.get("cenario", ""),
                "Modelo": r.get("modelo", ""),
                "MAE": round(r["MAE"], 4),
                "RMSE": round(r["RMSE"], 4),
                "R²": round(r["R2"], 4),
                "Média Real": round(r["media_real"], 4),
                "Média Prevista": round(r["media_pred"], 4),
            }
        )
    df = pd.DataFrame(linhas)
    return df
