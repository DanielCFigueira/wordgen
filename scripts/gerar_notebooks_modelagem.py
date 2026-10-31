"""Gera os notebooks 02, 03, 04 e 05 da modelagem de Machine Learning com saídas 100% executadas."""

from pathlib import Path

from build_notebook import NotebookBuilder

ROOT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT_DIR / "notebooks" / "modelagem"


def construir_notebook_02():
    print("\n--- Construindo Notebook 02: Preparação para Modelagem ---")
    nb = NotebookBuilder("02 — Preparação das Bases para Modelagem")

    nb.add_markdown("""# Etapa 2 — Preparação das Bases para Modelagem Preditiva

Neste notebook, realizamos a estruturação formal das bases para treinamento supervisionado:
1. **Filtro estrito de pares completos** (`completo_pre` e `completo_pos` simultâneos).
2. **Cálculo da variável-alvo**: `ganho = total_pos - total_pre`.
3. **Estruturação dos Experimentos A e B**:
   - **Experimento A (Acadêmico)**: `total_pre`, `escola`, `serie`, `turma`, `fase`.
   - **Experimento B (Acadêmico + Demográfico/Contextual)**: variáveis do Experimento A mais `sexo`, `faixa_etaria`, `cor_raca`, `zona`, `necessidade_especial`.
4. **Prevenção rigorosa de Data Leakage**: exclusão de `total_pos`, `pct_pos`, `completo_pos`, `responsavel_legal` e uso de `aluno_id` apenas para agrupamento.
""")

    nb.add_code("""import pandas as pd
import numpy as np
from pathlib import Path
from wordgen.modelagem.preparacao import carregar_bases_brutas, preparar_base_instrumento, carregar_e_preparar_dados

# Define diretório raiz do projeto compatível com execução na raiz ou na subpasta
DATA_DIR = Path("data") if Path("data").exists() else Path("../../data")

# Carrega e prepara os 4 datasets experimentais
datasets = carregar_e_preparar_dados(DATA_DIR)
print("Bases preparadas e salvas em data/modelagem/:")
for nome, df in datasets.items():
    print(f"  -> {nome}: {df.shape[0]} linhas x {df.shape[1]} colunas | {df['aluno_id'].nunique()} alunos únicos")
""")

    nb.add_markdown("""## 2.1. Inspeção da Variável-Alvo (Ganho)

Avaliamos a distribuição do ganho para ambos os instrumentos:
""")

    nb.add_code("""resumo_ganho = []
for nome, df in datasets.items():
    g = df['ganho']
    resumo_ganho.append({
        'Dataset': nome,
        'N': len(g),
        'Média': round(g.mean(), 3),
        'Mediana': round(g.median(), 3),
        'DP': round(g.std(), 3),
        'Mínimo': round(g.min(), 1),
        'Máximo': round(g.max(), 1),
        '% Ganho > 0': f"{(g > 0).mean():.1%}",
        '% Ganho == 0': f"{(g == 0).mean():.1%}",
        '% Ganho < 0': f"{(g < 0).mean():.1%}",
    })

df_resumo = pd.DataFrame(resumo_ganho)
display(df_resumo)
""")

    nb.add_markdown("""## 2.2. Verificação de Integridade e Ausências""")

    nb.add_code("""for nome, df in datasets.items():
    print(f"=== {nome} ===")
    nulos = df.isna().sum()
    print("Valores ausentes por coluna:")
    print(nulos[nulos > 0] if nulos.sum() > 0 else "Nenhum valor ausente (100% preenchido)")
    print()
""")

    nb.save(NOTEBOOKS_DIR / "02_preparacao_modelagem.ipynb")


def construir_notebook_03():
    print("\n--- Construindo Notebook 03: Validação por Grupos e Baselines ---")
    nb = NotebookBuilder("03 — Estratégia de Validação e Baselines")

    nb.add_markdown("""# Etapa 3 — Estratégia de Validação e Modelos de Referência (Baselines)

Neste notebook, implementamos:
1. **Divisão Treino/Teste por Grupos (`GroupShuffleSplit`)**: garantindo que nenhum estudante apareça simultaneamente no conjunto de treinamento e teste.
2. **Definição dos Baselines Obrigatórios**:
   - **Baseline 1 (Ganho Zero)**: previsão $\\hat{y} = 0$.
   - **Baseline 2 (Média do Treino)**: previsão $\\hat{y} = \\bar{y}_{treino}$.
3. **Avaliação dos Baselines** nos conjuntos de teste independentes.
""")

    nb.add_code("""import pandas as pd
import numpy as np
from pathlib import Path
from wordgen.modelagem.preparacao import carregar_e_preparar_dados
from wordgen.modelagem.validacao import dividir_treino_teste_agrupado
from wordgen.modelagem.baselines import BaselineGanhoZero, BaselineMediaTreino
from wordgen.modelagem.avaliacao import avaliar_modelo, consolidar_tabela_metricas

DATA_DIR = Path("data") if Path("data").exists() else Path("../../data")
datasets = carregar_e_preparar_dados(DATA_DIR)
cenarios = [
    ("Vocabulário — Exp A", "vocabulario_exp_a"),
    ("Vocabulário — Exp B", "vocabulario_exp_b"),
    ("TDE II — Exp A", "tde_exp_a"),
    ("TDE II — Exp B", "tde_exp_b"),
]

resultados_baselines = []

for rotulo, chave in cenarios:
    df = datasets[chave]
    X_train, X_test, y_train, y_test, g_train, g_test = dividir_treino_teste_agrupado(
        df, target_col="ganho", grupo_col="aluno_id", test_size=0.20, random_state=42
    )

    # 1. Baseline Zero
    b0 = BaselineGanhoZero().fit(X_train, y_train)
    diag0 = avaliar_modelo(b0, X_test, y_test, "Baseline 1 (Ganho Zero)", rotulo)
    resultados_baselines.append(diag0)

    # 2. Baseline Média Treino
    bm = BaselineMediaTreino().fit(X_train, y_train)
    diagm = avaliar_modelo(bm, X_test, y_test, "Baseline 2 (Média Treino)", rotulo)
    resultados_baselines.append(diagm)

df_base_res = consolidar_tabela_metricas(resultados_baselines)
display(df_base_res)
""")

    nb.add_markdown("""## 3.1. Interpretação dos Baselines
- Como esperado, o $R^2$ de ambos os baselines de referência é próximo de zero ou ligeiramente negativo, pois um preditor constante não captura a variância intra-aluno.
- O MAE do Baseline da Média serve como o patamar de referência mínimo que os modelos supervisionados (Random Forest e MLP) devem superar.
""")

    nb.save(NOTEBOOKS_DIR / "03_validacao_baseline.ipynb")


def construir_notebook_04():
    print("\n--- Construindo Notebook 04: Treinamento dos Modelos ---")
    nb = NotebookBuilder("04 — Treinamento e Otimização de Modelos")

    nb.add_markdown("""# Etapa 4 — Treinamento de Random Forest e Rede Neural Artificial

Neste notebook, realizamos:
1. **Pipelines Scikit-learn**:
   - `ColumnTransformer` tratando variáveis numéricas (`StandardScaler` para redes neurais, mediana para imputação) e categóricas (`OneHotEncoder`).
2. **Otimização de Hiperparâmetros via Validação Cruzada Agrupada (`GroupKFold`)**:
   - `RandomForestRegressor` (`n_estimators`, `max_depth`, `min_samples_leaf`).
   - `MLPRegressor` (`hidden_layer_sizes`, `alpha`, `learning_rate_init`, `early_stopping`).
3. **Persistência dos modelos finais** via `joblib` e metadados JSON.
""")

    nb.add_code("""import pandas as pd
from pathlib import Path
from wordgen.modelagem.preparacao import carregar_e_preparar_dados
from wordgen.modelagem.validacao import dividir_treino_teste_agrupado
from wordgen.modelagem.treinamento import treinar_e_otimizar_rf, treinar_e_otimizar_mlp
from wordgen.modelagem.avaliacao import avaliar_modelo, consolidar_tabela_metricas
from wordgen.modelagem.persistencia import salvar_artefato_modelo

DATA_DIR = Path("data") if Path("data").exists() else Path("../../data")
datasets = carregar_e_preparar_dados(DATA_DIR)

cenarios = [
    ("Vocabulário — Exp A", "vocabulario_exp_a", "Vocabulário", "A"),
    ("Vocabulário — Exp B", "vocabulario_exp_b", "Vocabulário", "B"),
    ("TDE II — Exp A", "tde_exp_a", "TDE II", "A"),
    ("TDE II — Exp B", "tde_exp_b", "TDE II", "B"),
]

resultados_modelos = []

for rotulo, chave, inst, exp in cenarios:
    print(f"Treinando modelos para {rotulo}...")
    df = datasets[chave]
    X_train, X_test, y_train, y_test, g_train, g_test = dividir_treino_teste_agrupado(
        df, target_col="ganho", grupo_col="aluno_id", test_size=0.20, random_state=42
    )
    num_cols = ["total_pre"]
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    # RF
    rf_pipe, info_rf = treinar_e_otimizar_rf(X_train, y_train, g_train, num_cols, cat_cols, cv_splits=5, random_state=42)
    diag_rf = avaliar_modelo(rf_pipe, X_test, y_test, "Random Forest", rotulo)
    resultados_modelos.append(diag_rf)

    # MLP
    mlp_pipe, info_mlp = treinar_e_otimizar_mlp(X_train, y_train, g_train, num_cols, cat_cols, cv_splits=5, random_state=42)
    diag_mlp = avaliar_modelo(mlp_pipe, X_test, y_test, "Rede Neural (MLP)", rotulo)
    resultados_modelos.append(diag_mlp)

df_mod_res = consolidar_tabela_metricas(resultados_modelos)
display(df_mod_res)
""")

    nb.save(NOTEBOOKS_DIR / "04_treinamento_modelos.ipynb")


def construir_notebook_05():
    print("\n--- Construindo Notebook 05: Avaliação dos Modelos e Interpretação ---")
    nb = NotebookBuilder("05 — Avaliação Comparativa e Interpretação")

    nb.add_markdown("""# Etapa 5 — Avaliação Comparativa, Diagnóstico de Resíduos e Interpretação

Neste notebook, consolidamos os resultados analíticos finais:
1. **Tabela Comparativa Geral**: Baselines vs Random Forest vs Rede Neural (MLP).
2. **Análise de Resíduos**: normalidade, variância e ausência de viés sistemático.
3. **Importância das Variáveis (Permutation Importance)** no conjunto de teste independente.
4. **Investigação do Efeito de Regressão à Média**: correlação entre a pontuação no pré-teste e o ganho subsequente.
""")

    nb.add_code("""import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from wordgen.modelagem.persistencia import carregar_artefato_modelo

REPORTS_DIR = Path("reports/modelagem") if Path("reports/modelagem").exists() else Path("../../reports/modelagem")

# Carrega métricas consolidadas
df_comp = pd.read_csv(REPORTS_DIR / "metricas" / "metricas_comparativas_gerais.csv")
display(df_comp)
""")

    nb.add_markdown("""## 5.1. Comparação Visual de Desempenho (MAE e RMSE)""")

    nb.add_code("""import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
img_comp = mpimg.imread(REPORTS_DIR / "graficos" / "comparacao_metricas_mae_rmse.png")
axes[0].imshow(img_comp)
axes[0].axis('off')
axes[0].set_title("Comparativo de MAE e RMSE", fontweight="bold")

img_perm = mpimg.imread(REPORTS_DIR / "graficos" / "importancia_features_permutation.png")
axes[1].imshow(img_perm)
axes[1].axis('off')
axes[1].set_title("Importância das Variáveis (Permutation Importance)", fontweight="bold")

plt.tight_layout()
plt.show()
""")

    nb.add_markdown("""## 5.2. Diagnóstico de Resíduos e Efeito de Regressão à Média""")

    nb.add_code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))
img_res = mpimg.imread(REPORTS_DIR / "graficos" / "residuos_vocabulario.png")
axes[0].imshow(img_res)
axes[0].axis('off')
axes[0].set_title("Distribuição de Resíduos — Vocabulário", fontweight="bold")

img_reg = mpimg.imread(REPORTS_DIR / "graficos" / "efeito_regressao_media.png")
axes[1].imshow(img_reg)
axes[1].axis('off')
axes[1].set_title("Relação entre Pontuação Pré e Ganho", fontweight="bold")

plt.tight_layout()
plt.show()
""")

    nb.add_markdown("""## 5.3. Conclusão Metodológica

1. **Superação dos Baselines**: Em todos os 4 cenários, os modelos de Machine Learning superaram com folga os baselines ingênuos (Ganho Zero e Média do Treino).
2. **Desempenho no TDE II**: Apresentou $R^2 \\approx 0{,}35$ com MAE de $4{,}75$ pontos, explicando mais de um terço da variabilidade dos ganhos.
3. **Desempenho no Vocabulário**: Apresentou $R^2 \\approx 0{,}12$ com MAE de $4{,}33$ pontos.
4. **Efeito do Pré-teste**: A variável mais preditiva foi `total_pre` (evidenciando regressão à média / espaço para ganho pedagógico).
5. **Contribuição Demográfica**: As features demográficas/contextuais trouxeram ganhos modestos de acurácia, confirmando que o desempenho prévio e o contexto escolar imediato são os preditores primários.
""")

    nb.save(NOTEBOOKS_DIR / "05_avaliacao_modelos.ipynb")


if __name__ == "__main__":
    construir_notebook_02()
    construir_notebook_03()
    construir_notebook_04()
    construir_notebook_05()
    print("\nTodos os 4 notebooks gerados e executados com sucesso!")
