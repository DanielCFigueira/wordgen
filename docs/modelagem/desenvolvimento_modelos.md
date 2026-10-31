# Desenvolvimento dos Modelos — WordGen

## 1. Algoritmos Obrigatórios

Foram desenvolvidos e avaliados dois paradigmas de aprendizado supervisionado:

### 1.1. Random Forest Regressor
- **Biblioteca**: `sklearn.ensemble.RandomForestRegressor`
- **Fundamentação**: Modelo baseado em ensemble de árvores de decisão com bagging. Excelente capacidade de capturar relações não-lineares, interações complexas entre variáveis categóricas e robustez contra outliers sem necessidade de escalonamento numérico estrito.

### 1.2. Rede Neural Artificial (MLPRegressor)
- **Biblioteca**: `sklearn.neural_network.MLPRegressor`
- **Fundamentação**: Perceptron Multicamadas para regressão supervisionada. Avalia a capacidade de aproximação universal não-linear através de representações intermediárias e ativação ReLU. Exige padronização das entradas (`StandardScaler`) e regularização $L_2$ (`alpha`).

---

## 2. Otimização de Hiperparâmetros

A busca de hiperparâmetros foi conduzida com `GridSearchCV` e `GroupKFold(n_splits=5)` minimizando o Erro Médio Absoluto (MAE):

### 2.1. Grade do Random Forest
- `n_estimators`: `[100, 200]`
- `max_depth`: `[6, 10, None]`
- `min_samples_leaf`: `[2, 5]`
- `random_state`: `42`, `n_jobs`: `-1`

### 2.2. Grade do MLPRegressor
- `hidden_layer_sizes`: `[(64, 32), (32, 16), (64,)]`
- `alpha`: `[0.001, 0.01, 0.1]`
- `learning_rate_init`: `[0.005, 0.01]`
- `activation`: `'relu'`, `solver`: `'adam'`, `max_iter`: `500`, `early_stopping`: `True`

---

## 3. Melhores Hiperparâmetros Selecionados por Cenário

| Cenário | Modelo | Melhores Parâmetros Selecionados | MAE Médio (CV 5-Folds) |
| :--- | :--- | :--- | :--- |
| **Vocabulário — Exp A** | Random Forest | `max_depth: 6, min_samples_leaf: 2, n_estimators: 100` | 4,4537 |
| **Vocabulário — Exp A** | Rede Neural (MLP) | `alpha: 0.001, hidden_layer_sizes: (64, 32), lr_init: 0.005` | 4,4759 |
| **Vocabulário — Exp B** | Random Forest | `max_depth: 6, min_samples_leaf: 2, n_estimators: 100` | 4,4432 |
| **Vocabulário — Exp B** | Rede Neural (MLP) | `alpha: 0.1, hidden_layer_sizes: (64,), lr_init: 0.005` | 4,4771 |
| **TDE II — Exp A** | Random Forest | `max_depth: 10, min_samples_leaf: 2, n_estimators: 100` | 4,8967 |
| **TDE II — Exp A** | Rede Neural (MLP) | `alpha: 0.001, hidden_layer_sizes: (64, 32), lr_init: 0.01` | 4,9422 |
| **TDE II — Exp B** | Random Forest | `max_depth: 10, min_samples_leaf: 2, n_estimators: 100` | 4,8911 |
| **TDE II — Exp B** | Rede Neural (MLP) | `alpha: 0.1, hidden_layer_sizes: (64,), lr_init: 0.01` | 4,9601 |

---

## 4. Persistência de Modelos e Metadados

Todos os 8 pipelines treinados (pré-processador Scikit-learn + modelo ajustado) foram persistidos via `joblib` no diretório `models/modelagem/`:
- `vocabulario_exp_a_rf.joblib` / `_metadata.json`
- `vocabulario_exp_a_mlp.joblib` / `_metadata.json`
- `vocabulario_exp_b_rf.joblib` / `_metadata.json`
- `vocabulario_exp_b_mlp.joblib` / `_metadata.json`
- `tde_exp_a_rf.joblib` / `_metadata.json`
- `tde_exp_a_mlp.joblib` / `_metadata.json`
- `tde_exp_b_rf.joblib` / `_metadata.json`
- `tde_exp_b_mlp.joblib` / `_metadata.json`

Os metadados JSON registram: lista de features, hiperparâmetros ótimos, score de validação cruzada, tamanho das amostras de treino/teste e métricas finais.
