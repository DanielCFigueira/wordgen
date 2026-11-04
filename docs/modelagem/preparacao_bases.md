# Preparação das Bases para Modelagem — WordGen

## 1. Critérios de Seleção dos Registros

Para assegurar validade psicométrica e solidez preditiva, os modelos supervisionados utilizam apenas registros que atendem a todos os critérios de elegibilidade:
1. `completo_pre == True`: estudante respondeu validamente a pelo menos 80% dos itens no pré-teste.
2. `completo_pos == True`: estudante respondeu validamente a pelo menos 80% dos itens no pós-teste.
3. `total_pre` e `total_pos` contidos estritamente nos intervalos teóricos do instrumento ([0, 50] para Vocabulário e [0, 80] para TDE II).
4. `aluno_id` válido e não nulo.

### Volume de Registros Selecionados
- **Vocabulário**: 3.320 pares completos (1.989 estudantes únicos) a partir de 3.418 pares brutos.
- **TDE II**: 3.796 pares completos (2.069 estudantes únicos) a partir de 3.948 pares brutos.

> **Limitação de Generalização**: A seleção de pares completos restringe a população modelada aos estudantes que possuem avaliações completas em ambos os momentos (pré e pós-teste). Estudantes com avaliações parciais (< 80% dos itens respondidos) ou sem dados de pós-teste foram excluídos. Consequentemente, os modelos e suas previsões **não devem ser generalizados automaticamente** para estudantes com avaliações incompletas, ausentes do pós-teste, ou de contextos não representados nos dados de treinamento. A população modelada representa 97,1% (Vocabulário) e 96,2% (TDE II) dos registros brutos pareados.

`total_pre` e `total_pos` podem estar disponíveis mesmo quando a avaliação não atingiu o limiar de completude. Ter uma pontuação calculável, portanto, não torna o registro elegível: o filtro preserva apenas pares com ambos os indicadores `completo_*` verdadeiros. A limitação aplica-se à população de pares completos, não apenas a casos sem pontuação.

---

## 2. Construção da Variável-Alvo (Target)

A variável-alvo contínua para regressão supervisionada é calculada intra-aluno:

$$\text{ganho} = \text{total\_pos} - \text{total\_pre}$$

### Estatísticas Descritivas do Target nos Pares Completos

| Instrumento | N | Média | Mediana | Desvio Padrão | Mínimo | Máximo | % Ganhos > 0 | % Ganhos = 0 | % Ganhos < 0 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vocabulário** | 3.320 | +0,627 | +1,0 | 5,984 | -29,0 | +29,0 | 52,47% | 8,31% | 39,22% |
| **TDE II** | 3.796 | +0,039 | 0,0 | 8,770 | -53,0 | +63,0 | 39,86% | 26,34% | 33,80% |

Não há valores nulos ou infinitos em nenhuma das duas bases.

---

## 3. Seleção de Features e Desenho Experimental

A modelagem foi estruturada em dois cenários experimentais:

### Experimento A — Modelo Acadêmico
Utiliza as seguintes variáveis acadêmicas e o desempenho anterior selecionados no pipeline:
- `total_pre`: pontuação numérica prévia do estudante.
- `fase`: fase da intervenção ('Fase 2', 'Fase 3', 'Fase 4').
- `escola`: unidade escolar municipal.
- `serie`: ano de escolaridade ('6', '7', '8', '9').
- `turma`: turma do estudante.

`total_pre` representa o desempenho anterior. `aluno_id` é mantido somente para agrupamento e não é preditor.

### Experimento B — Modelo Acadêmico + Demográfico/Contextual
Adiciona as variáveis consolidadas da dimensão `dim_aluno`:
- Features do Experimento A.
- `sexo`: categoria de sexo registrada ('M', 'F', 'NAO_INFORMADO').
- `faixa_etaria`: faixa etária do estudante.
- `cor_raca`: categoria de cor/raça registrada.
- `zona`: classificação territorial registrada ('URBANA', 'RURAL', 'NAO INFORMADA').
- `necessidade_especial`: indicador de inclusão ('SIM', 'NAO_REGISTRADO').

Essas categorias têm cobertura de 100% dentro de `dim_aluno`, incluindo categorias explícitas de não informação. Isso não equivale a 100% de cobertura entre estudantes avaliados: nos pares completos, a correspondência da dimensão é 99,07% em Vocabulário e 99,42% em TDE II. `zona` é a zona territorial normalizada para URBANA/RURAL; `necessidade_especial` é derivada de registro em `necessidade_deficiencia`, e `NAO_REGISTRADO` não significa ausência confirmada de necessidade. Não há variáveis de renda, escolaridade familiar ou composição do domicílio disponíveis de forma confiável; por isso o Experimento B é demográfico/contextual, não socioeconômico.

A junção com `dim_aluno` utiliza `validate='many_to_one'` para garantir que `dim_aluno` possui no máximo um registro por `aluno_id`. Essa validação explícita previne a multiplicação silenciosa de linhas que ocorreria se a dimensão contivesse duplicatas. A cobertura de `dim_aluno` nos pares completos é de 99,07% (Vocabulário) e 99,42% (TDE II); os estudantes sem cadastro demográfico são tratados com categoria explícita `NAO_INFORMADO` em vez de serem descartados.

> **Limite temporal do Experimento B**: `dim_aluno` é consolidada a partir de registros de múltiplas fases por estudante. Algumas regras usam registros mais recentes (como faixa etária) ou combinam registros de todas as fases (como necessidade especial). Como essa dimensão não guarda a fase de origem de cada atributo individual, a preparação atual não comprova que todos os valores estavam disponíveis no momento da previsão de cada observação. O Experimento A é o cenário acadêmico principal e o B é exploratório complementar. Uma versão futura deverá reconstruir as características demográficas/contextuais respeitando a fase e o momento da previsão antes de qualquer interpretação operacional.

---

## 4. Prevenção Rigorosa de Vazamento de Dados (Data Leakage)

Todas as seguintes variáveis foram **estritamente excluídas das features preditoras**:
- `total_pos`, `pct_pos`, `completo_pos`: dados medidos no momento futuro do pós-teste.
- Questões individuais do pós-teste.
- `aluno_id`: utilizado unicamente como identificador de grupo para a partição treino/teste, nunca como feature preditora.
- `responsavel_legal`: dado pessoal identificável (PII), excluído por conformidade e ética.
- `fonte_socio`: metadado técnico de arquivo, irrelevante e sem significado educacional.

---

## 5. Pipelines de Pré-Processamento Scikit-learn

O pré-processamento é encapsulado com `ColumnTransformer` e `Pipeline` para garantir que transformações e imputações sejam ajustadas **exclusivamente sobre o conjunto de treinamento**:
- **Variáveis Numéricas (`total_pre`)**:
  - `SimpleImputer(strategy='median')`.
  - `StandardScaler()` obrigatório para `MLPRegressor` (estabilização dos gradientes e convergência do otimizador Adam).
- **Variáveis Categóricas**:
  - `SimpleImputer(strategy='constant', fill_value='NAO_INFORMADO')`.
  - `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` para lidar com novas categorias sem falhar em produção.

---

## 6. Arquivos Gerados

Os datasets finais preparados foram exportados em `data/modelagem/` nos formatos Parquet e CSV:
- `vocabulario_exp_a.parquet` / `.csv` (3.320 linhas × 7 colunas)
- `vocabulario_exp_b.parquet` / `.csv` (3.320 linhas × 12 colunas)
- `tde_exp_a.parquet` / `.csv` (3.796 linhas × 7 colunas)
- `tde_exp_b.parquet` / `.csv` (3.796 linhas × 12 colunas)
