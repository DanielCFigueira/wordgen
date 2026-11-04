# Estratégia de Validação e Baselines — WordGen

## 1. Desafio Metodológico: Rastreamento Longitudinal de Estudantes

Em dados educacionais com múltiplas fases (Fase 2, Fase 3 e Fase 4), um mesmo estudante (`aluno_id`) pode comparecer em até três momentos avaliativos distintos. 

Uma divisão aleatória simples por linhas geraria **vazamento de dados por estudante (Data Leakage)**: registros do mesmo aluno em diferentes fases estariam presentes simultaneamente no treino e no teste, inflando artificialmente as métricas de avaliação e comprometendo a generalização para novos alunos.

---

## 2. Divisão Agrupada por Estudante (`GroupShuffleSplit`)

Adotou-se o algoritmo `GroupShuffleSplit` da biblioteca Scikit-learn com proporção aproximada de 80% para treinamento e 20% para teste, utilizando `aluno_id` como agrupador:

```python
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["aluno_id"].values))
```

### Composição das Partições

#### Vocabulário (Experimentos A e B)
- **Total de pares completos**: 3.320 registros (1.989 estudantes únicos)
- **Conjunto de Treinamento**: 2.665 registros (1.591 estudantes únicos — 80,0%)
- **Conjunto de Teste**: 655 registros (398 estudantes únicos — 20,0%)
- **Interseção de estudantes entre treino e teste**: **0 estudantes** (isolamento absoluto)

#### TDE II (Experimentos A e B)
- **Total de pares completos**: 3.796 registros (2.069 estudantes únicos)
- **Conjunto de Treinamento**: 3.050 registros (1.655 estudantes únicos — 80,0%)
- **Conjunto de Teste**: 746 registros (414 estudantes únicos — 20,0%)
- **Interseção de estudantes entre treino e teste**: **0 estudantes** (isolamento absoluto)

---

## 3. Validação Cruzada Agrupada no Treinamento (`GroupKFold`)

Durante a etapa de busca de hiperparâmetros (`GridSearchCV`), a validação cruzada utiliza particionamento estrito por grupos (`GroupKFold(n_splits=5)`). O vetor de grupos de treinamento, derivado de `aluno_id`, é passado explicitamente a `fit(..., groups=groups_train)`. O pré-processamento faz parte do pipeline avaliado em cada fold, portanto imputação, escala e codificação são ajustadas apenas no subconjunto de treino daquele fold. Para o MLP, `early_stopping=False` evita uma divisão interna que não respeitaria os grupos.

O conjunto de teste final de 20% permanece fora da busca e do ajuste de hiperparâmetros, sendo usado somente na avaliação final. Os grupos do teste são disjuntos dos grupos de treino.

---

## 4. Modelos de Referência (Baselines)

Para comprovar formalmente o valor agregado dos modelos de Machine Learning, foram implementados dois baselines de referência:

### Baseline 1 — Ganho Zero
Assume a premissa nula de que a intervenção pedagógica não altera o escore do estudante:
$$\hat{y}_i = 0, \quad \forall i$$

### Baseline 2 — Média do Treinamento
Assume uma estimativa constante igual à média aritmética do ganho observada estritamente no conjunto de treino:
$$\hat{y}_i = \bar{y}_{treino} = \frac{1}{N_{treino}} \sum_{j=1}^{N_{treino}} y_j, \quad \forall i$$

> **Regra Rígida de Prevenção de Vazamento**: A média do teste **nunca** é utilizada para construir o baseline.

---

## 5. Resultados Obtidos pelos Baselines no Teste

| Instrumento | Cenário | Baseline | MAE | RMSE | R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vocabulário** | Exp A e B | Baseline 1 (Ganho Zero) | 4,5557 | 6,1371 | -0,0090 |
| **Vocabulário** | Exp A e B | Baseline 2 (Média Treino = 0,639) | 4,5313 | 6,1099 | -0,0001 |
| **TDE II** | Exp A e B | Baseline 1 (Ganho Zero) | 5,7708 | 9,3246 | -0,0010 |
| **TDE II** | Exp A e B | Baseline 2 (Média Treino = 0,120) | 5,7995 | 9,3291 | -0,0020 |

Os baselines confirmam que um preditor constante resulta em $R^2 \approx 0$, definindo o patamar de erro absoluto (MAE de 4,53 em Vocabulário e 5,77 em TDE II) que os modelos supervisionados devem superar para ter utilidade prática.
