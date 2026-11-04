"""Módulo de pipelines e pré-processadores Scikit-learn."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def obter_pre_processador(
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    escalar_numericas: bool = True,
) -> ColumnTransformer:
    """Cria um ColumnTransformer com tratamento separado para atributos numéricos e categóricos."""
    if escalar_numericas:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        num_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="NAO_INFORMADO")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if colunas_numericas:
        transformers.append(("num", num_pipe, colunas_numericas))
    if colunas_categoricas:
        transformers.append(("cat", cat_pipe, colunas_categoricas))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def criar_pipeline_rf(
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    rf_params: dict | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Constrói o pipeline Scikit-learn completo para o RandomForestRegressor."""
    params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "random_state": random_state,
        "n_jobs": -1,
    }
    if rf_params:
        params.update(rf_params)

    preprocessor = obter_pre_processador(
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        escalar_numericas=False,
    )

    rf_model = RandomForestRegressor(**params)

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", rf_model),
        ]
    )
    return pipe


def criar_pipeline_mlp(
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    mlp_params: dict | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Constrói o pipeline Scikit-learn completo para a Rede Neural Artificial (MLPRegressor).

    Nota: `early_stopping` está desabilitado intencionalmente.
    O MLPRegressor com `early_stopping=True` realiza uma divisão interna aleatória
    do conjunto de treinamento para validação, sem considerar o agrupamento por
    `aluno_id`. Isso conflita com a estratégia de validação agrupada adotada no
    projeto e pode causar vazamento de dados por estudante na etapa interna do MLP.
    A regularização é controlada exclusivamente via `alpha` (L2) e a parada é
    definida por `max_iter`, suficiente para convergência com os dados disponíveis.
    A seleção de hiperparâmetros é conduzida por `GroupKFold` externo no GridSearchCV.
    """
    params = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.01,
        "learning_rate_init": 0.005,
        "max_iter": 1000,
        "early_stopping": False,
        "tol": 1e-4,
        "random_state": random_state,
    }
    if mlp_params:
        params.update(mlp_params)

    preprocessor = obter_pre_processador(
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        escalar_numericas=True,  # Redes neurais exigem padronização dos inputs
    )

    mlp_model = MLPRegressor(**params)

    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", mlp_model),
        ]
    )
    return pipe
