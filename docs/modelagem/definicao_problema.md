# Definição do Problema — Modelagem WordGen

## 1. Contextualização do Problema

O projeto WordGen reúne dados de avaliações educacionais realizadas com alunos em diferentes fases, contemplando dois instrumentos: Vocabulário e TDE II.

Cada instrumento possui avaliações de pré-teste e pós-teste, permitindo observar o desempenho dos alunos e analisar sua evolução ao longo do processo avaliativo.

Embora a etapa de Análise de Dados permita compreender o comportamento histórico dos resultados, existe a oportunidade de avançar para uma abordagem preditiva.

A proposta é utilizar técnicas de Machine Learning para estimar a evolução do desempenho dos alunos, considerando seu desempenho anterior, informações acadêmicas e características demográficas e contextuais disponíveis.

Dessa forma, pretende-se complementar as análises descritivas, estatísticas e longitudinais com uma perspectiva preditiva, contribuindo para o acompanhamento do desempenho acadêmico.

## 2. Objetivo da Modelagem

Desenvolver modelos de Machine Learning capazes de prever o ganho dos alunos nas avaliações de Vocabulário e TDE II utilizando informações de desempenho anterior e características acadêmicas. O Experimento B explora adicionalmente variáveis demográficas/contextuais de `dim_aluno`, sem comprovação da disponibilidade temporal individual desses atributos.

A modelagem terá como objetivo estimar a variação da pontuação entre o pré-teste e o pós-teste, permitindo identificar padrões associados à evolução do desempenho dos alunos e complementar as análises descritivas, estatísticas e longitudinais realizadas no projeto WordGen.

### Objetivos específicos

- Desenvolver modelos preditivos independentes para os instrumentos de Vocabulário e TDE II.
- Investigar a relação entre o desempenho no pré-teste e a evolução da pontuação.
- Avaliar a contribuição das características acadêmicas e demográficas/contextuais disponíveis para as previsões.
- Comparar o desempenho de diferentes algoritmos de regressão.
- Avaliar a capacidade preditiva dos modelos por meio de métricas apropriadas.
- Disponibilizar resultados que possam complementar os indicadores e visualizações do dashboard.

## 3. Definição da Variável-Alvo (Target)

A variável-alvo da modelagem será o ganho de desempenho do aluno entre o pré-teste e o pós-teste, calculado pela diferença entre as pontuações obtidas nas duas avaliações.

A definição matemática será:

**Ganho = Pontuação do Pós-teste − Pontuação do Pré-teste**

Serão utilizadas variáveis-alvo independentes para cada instrumento:

- `ganho_vocabulario`: representa a evolução da pontuação do aluno na avaliação de Vocabulário.
- `ganho_tde`: representa a evolução da pontuação do aluno na avaliação TDE II.

A interpretação dos valores será:

- Valores positivos representam aumento da pontuação.
- Valores iguais a zero representam manutenção da pontuação.
- Valores negativos representam redução da pontuação.

A modelagem será formulada como um problema de regressão, no qual os algoritmos deverão estimar o ganho de desempenho dos alunos a partir de informações disponíveis antes da realização do pós-teste.

Somente registros com pontuações válidas de pré-teste e pós-teste poderão ser utilizados no treinamento e na avaliação dos modelos.

## 4. Definição das Variáveis Preditoras (Features)

As variáveis abaixo distinguem o que existe nas bases do que foi efetivamente selecionado pelos pipelines. Os quatro pipelines usam `total_pre` como único preditor numérico; itens individuais e atributos do pós-teste não são usados.

### 4.1. Variáveis efetivamente utilizadas

**Experimento A — cenário acadêmico principal**, em cada instrumento:
- Numérica: `total_pre`.
- Categóricas: `fase`, `escola`, `serie` e `turma`.

**Experimento B — exploratório complementar**, em cada instrumento, utiliza as mesmas variáveis do A e acrescenta `sexo`, `faixa_etaria`, `cor_raca`, `zona` e `necessidade_especial`.

### 4.2. Variáveis disponíveis, mas não utilizadas

As bases pareadas contêm pontuações por item, totais e informações dos dois momentos. Os itens individuais do pré-teste não entram nos pipelines. `total_pos`, `ganho` e medidas derivadas do pós-teste são excluídos das preditoras. A disponibilidade de uma coluna na base não significa que ela foi utilizada.

### 4.3. Características demográficas e contextuais

As variáveis demográficas e contextuais disponíveis na dimensão `dim_aluno` são:
- `sexo`: gênero autodeclarado do estudante.
- `faixa_etaria`: faixa etária do estudante.
- `cor_raca`: classificação étnico-racial autodeclarada.
- `zona`: localização territorial (URBANA / RURAL).
- `necessidade_especial`: indicador de inclusão educacional.

> **Nota**: essas variáveis são de natureza **demográfica e contextual**, não estritamente socioeconômica. Variáveis tipicamente socioeconômicas (como renda familiar, escolaridade dos responsáveis e composição familiar) não estão disponíveis de forma confiável na base de dados.

Como as informações demográficas não estão disponíveis para todos os estudantes, os ausentes são tratados com categoria explícita `NAO_INFORMADO`, sem descartar registros. Dois experimentos são conduzidos:

- Modelo acadêmico: utilizará informações de desempenho anterior e características acadêmicas.
- Modelo acadêmico e demográfico/contextual: adicionará as características demográficas e contextuais disponíveis.

**Limitação temporal do Experimento B**: `dim_aluno` consolida atributos de múltiplas fases e não registra a fase de origem de cada campo. Não está comprovado que todos os atributos demográficos/contextuais estavam disponíveis no momento da previsão. O Experimento B é exploratório; uma versão futura deve reconstruir essas características por fase e no momento da previsão.

### 4.4. Restrições na seleção das features

`total_pre` e as características acadêmicas são usados como preditores conforme os pipelines. A disponibilidade temporal individual de cada atributo de `dim_aluno` não está comprovada.

Não serão utilizadas informações provenientes do pós-teste, da variável-alvo ou de indicadores calculados com informações futuras, evitando vazamento de dados (Data Leakage).

## 5. Tipo de Aprendizado e Tarefa de Machine Learning

A modelagem será desenvolvida utilizando Aprendizado Supervisionado, uma vez que os dados históricos possuem variáveis preditoras e uma variável-alvo conhecida, correspondente ao ganho de desempenho dos alunos.

A tarefa será de Regressão Supervisionada, pois o objetivo é estimar um valor numérico que representa a diferença entre as pontuações do pré-teste e do pós-teste.

### 5.1. Algoritmos selecionados

Serão utilizados dois algoritmos de regressão:

- **MLPRegressor (Rede Neural Artificial):** utilizado para investigar a capacidade de redes neurais em aprender relações entre as características dos alunos e sua evolução de desempenho.
- **RandomForestRegressor (Floresta Aleatória):** utilizado para investigar a capacidade de um conjunto de árvores de decisão em identificar padrões e realizar previsões.

Ambos os algoritmos serão implementados por meio da biblioteca Scikit-learn.

### 5.2. Organização dos experimentos

A modelagem será realizada separadamente para os instrumentos Vocabulário e TDE II.

Para cada instrumento, serão conduzidos dois experimentos:

- **Experimento A — acadêmico (principal)**: `total_pre`, `fase`, `escola`, `serie` e `turma`.
- **Experimento B — acadêmico + demográfico/contextual (exploratório)**: mesmas variáveis do A, acrescidas de `sexo`, `faixa_etaria`, `cor_raca`, `zona` e `necessidade_especial`. A disponibilidade temporal individual dessas características não está comprovada; uma versão futura deverá reconstruí-las por fase e momento da previsão.

Os dois algoritmos serão avaliados em cada experimento, permitindo comparar seus resultados.

## 6. Critérios de Sucesso e Avaliação dos Modelos

A avaliação dos modelos será realizada por meio da comparação com estratégias de referência (*baselines*), métricas de regressão e análise da capacidade de generalização.

### 6.1. Modelos de referência (Baselines)

Serão utilizadas duas estratégias de referência:

- Baseline de ganho zero: considera que todos os alunos terão ganho igual a zero.
- Baseline da média: prevê para todos os alunos a média dos ganhos observados no conjunto de treinamento.

Os algoritmos deverão apresentar resultados superiores a essas referências para demonstrar que agregam valor à previsão.

### 6.2. Métricas de avaliação

Serão utilizadas as seguintes métricas:

- MAE (Mean Absolute Error): mede o erro absoluto médio das previsões.
- RMSE (Root Mean Squared Error): mede a raiz do erro quadrático médio, penalizando mais fortemente erros de maior magnitude.
- R² (Coeficiente de Determinação): avalia a capacidade do modelo de explicar a variação da variável-alvo em relação à referência da média.

A interpretação das métricas será realizada em conjunto, considerando as características de cada instrumento e a distribuição dos ganhos.

### 6.3. Estratégia de validação

Os dados serão separados em conjuntos de treinamento e teste, inicialmente considerando uma proporção de 80% para treinamento e 20% para teste.

A separação será realizada de maneira agrupada por identificador do aluno (`aluno_id`), evitando que registros de um mesmo aluno estejam simultaneamente nos conjuntos de treinamento e teste.

Também poderá ser utilizada validação cruzada agrupada para investigar a estabilidade dos resultados.

### 6.4. Critérios de sucesso

A modelagem será considerada satisfatória quando:

- Os modelos apresentarem resultados superiores aos baselines.
- Demonstrarem capacidade de generalização para alunos não utilizados no treinamento.
- Apresentarem resultados consistentes nas métricas MAE, RMSE e R².
- Permitirem identificar e investigar padrões nos erros de previsão.
- Possibilitarem a comparação entre os instrumentos e os diferentes experimentos.
- Produzirem resultados que possam complementar as análises e os indicadores do dashboard.

### 6.5. Integração com o dashboard

Os resultados da modelagem serão disponibilizados para complementar a análise de dados do projeto WordGen.

Serão considerados indicadores como ganho real, ganho previsto, comparação entre valores reais e previstos e métricas de avaliação dos modelos.

Dessa forma, a etapa de Machine Learning será integrada à solução analítica, ampliando a capacidade de acompanhamento e interpretação dos resultados educacionais.
