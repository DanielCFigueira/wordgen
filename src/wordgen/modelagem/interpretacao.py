"""Módulo de interpretação de modelos e importância de variáveis."""

import pandas as pd
from sklearn.inspection import permutation_importance


def extrair_nomes_features(pipeline) -> list[str]:
    """Extrai os nomes transformados das features do ColumnTransformer."""
    preprocessor = pipeline.named_steps["preprocessor"]
    nomes = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        if hasattr(transformer, "named_steps"):
            ultimo_passo = list(transformer.named_steps.values())[-1]
            if hasattr(ultimo_passo, "get_feature_names_out"):
                nomes.extend(ultimo_passo.get_feature_names_out(cols))
            else:
                nomes.extend(cols)
        elif hasattr(transformer, "get_feature_names_out"):
            nomes.extend(transformer.get_feature_names_out(cols))
        else:
            nomes.extend(cols)
    return [str(n) for n in nomes]


def calcular_importancia_rf(pipeline) -> pd.DataFrame:
    """Extrai a importância MDI (Mean Decrease in Impurity) das árvores do Random Forest."""
    rf = pipeline.named_steps["regressor"]
    nomes_features = extrair_nomes_features(pipeline)
    importancias = rf.feature_importances_

    df_imp = (
        pd.DataFrame(
            {
                "Feature": nomes_features,
                "Importancia_MDI": importancias,
            }
        )
        .sort_values(by="Importancia_MDI", ascending=False)
        .reset_index(drop=True)
    )
    return df_imp


def calcular_permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Calcula Permutation Importance no conjunto de teste independente."""
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = (
        pd.DataFrame(
            {
                "Feature": X_test.columns,
                "Permutation_Importance_Mean": result.importances_mean,
                "Permutation_Importance_Std": result.importances_std,
            }
        )
        .sort_values(by="Permutation_Importance_Mean", ascending=False)
        .reset_index(drop=True)
    )
    return df_perm
