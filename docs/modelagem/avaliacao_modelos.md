# Avaliação dos Modelos e Interpretação dos Resultados — WordGen

## 1. Métricas de Avaliação

A performance preditiva foi avaliada sobre conjuntos de teste independentes (sem qualquer vazamento de alunos) através de três métricas complementares:
1. **MAE (Mean Absolute Error)**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$. Medida direta do desvio médio de pontuação na escala original do teste.
2. **RMSE (Root Mean Squared Error)**: $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$. Penaliza desvios de grande magnitude.
3. **$R^2$ (Coeficiente de Determinação)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$. Fração da variabilidade do ganho explicada pelo modelo.

---

## 2. Tabela Comparativa Consolidada de Resultados

| Instrumento | Cenário | Modelo | MAE | RMSE | R² | Média Real | Média Prevista |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vocabulário** | Exp A (Acadêmico) | Baseline 1 (Ganho Zero) | 4,5557 | 6,1371 | -0,0090 | 0,5802 | 0,0000 |
| **Vocabulário** | Exp A (Acadêmico) | Baseline 2 (Média Treino) | 4,5313 | 6,1099 | -0,0001 | 0,5802 | 0,6390 |
| **Vocabulário** | Exp A (Acadêmico) | **Random Forest** | **4,3669** | **5,7665** | **0,1092** | 0,5802 | 0,5801 |
| **Vocabulário** | Exp A (Acadêmico) | Rede Neural (MLP) | 4,3878 | 5,7684 | 0,1086 | 0,5802 | 1,2701 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico) | Baseline 1 (Ganho Zero) | 4,5557 | 6,1371 | -0,0090 | 0,5802 | 0,0000 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico) | Baseline 2 (Média Treino) | 4,5313 | 6,1099 | -0,0001 | 0,5802 | 0,6390 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico) | **Random Forest (Melhor)** | **4,3354** | **5,7325** | **0,1196** | 0,5802 | 0,5677 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico) | Rede Neural (MLP) | 4,4255 | 5,7782 | 0,1056 | 0,5802 | 0,3858 |
| **TDE II** | Exp A (Acadêmico) | Baseline 1 (Ganho Zero) | 5,7708 | 9,3246 | -0,0010 | -0,2936 | 0,0000 |
| **TDE II** | Exp A (Acadêmico) | Baseline 2 (Média Treino) | 5,7995 | 9,3291 | -0,0020 | -0,2936 | 0,1203 |
| **TDE II** | Exp A (Acadêmico) | **Random Forest** | **4,7626** | **7,5154** | **0,3498** | -0,2936 | -0,0545 |
| **TDE II** | Exp A (Acadêmico) | Rede Neural (MLP) | 4,9375 | 7,7009 | 0,3173 | -0,2936 | -0,0361 |
| **TDE II** | Exp B (Acadêmico + Demográfico) | Baseline 1 (Ganho Zero) | 5,7708 | 9,3246 | -0,0010 | -0,2936 | 0,0000 |
| **TDE II** | Exp B (Acadêmico + Demográfico) | Baseline 2 (Média Treino) | 5,7995 | 9,3291 | -0,0020 | -0,2936 | 0,1203 |
| **TDE II** | Exp B (Acadêmico + Demográfico) | **Random Forest (Melhor)** | **4,7578** | **7,4824** | **0,3555** | -0,2936 | -0,0599 |
| **TDE II** | Exp B (Acadêmico + Demográfico) | Rede Neural (MLP) | 5,0896 | 7,6842 | 0,3202 | -0,2936 | -0,5791 |

---

## 3. Interpretação e Principais Achados

### 3.1. Superação dos Baselines
- Em ambos os instrumentos, tanto o Random Forest quanto a Rede Neural Artificial superaram os baselines ingênuos de forma consistente.
- No TDE II, o Random Forest reduziu o MAE de 5,77 para **4,76 pontos** (redução de mais de 1 ponto de erro na escala de 0 a 80) e o RMSE de 9,32 para **7,48 pontos**, alcançando $R^2 = 0{,}3555$ (35,55% da variância do ganho explicada).
- No Vocabulário, o Random Forest alcançou MAE de **4,3354 pontos** (vs. 4,5313 da média) e $R^2 = 0{,}1196$.

### 3.2. Comparação entre Random Forest e MLPRegressor
- O **Random Forest Regressor obteve o melhor desempenho em todos os cenários**, demonstrando maior robustez para dados tabulares educacionais com predomínio de variáveis categóricas esparsas (turma, escola, série).
- A **Rede Neural MLP apresentou desempenho competitivo** (alcançando $R^2 = 0{,}32$ no TDE II e $R^2 = 0{,}11$ no Vocabulário), mas teve maior variabilidade nas previsões e tendência de convergência subótima nas caudas extremas.

### 3.3. Contribuição das Variáveis Demográficas (Experimento A vs. B)
- A inclusão de dados demográficos (`sexo`, `faixa_etaria`, `cor_raca`, `zona`, `necessidade_especial`) no Experimento B gerou um ganho marginal pequeno:
  - Vocabulário: $R^2$ subiu de $0{,}1092$ para $0{,}1196$ (ganho de +0,01).
  - TDE II: $R^2$ subiu de $0{,}3498$ para $0{,}3555$ (ganho de +0,006).
- **Conclusão Pedagógica**: O fator determinante do ganho reside na proficiência inicial do aluno (`total_pre`) e no contexto escolar imediato (`escola`/`turma`), enquanto as características demográficas desempenham papel secundário na previsão do ganho intra-aluno.

---

## 4. Importância das Variáveis e Fenômeno de Regressão à Média

A análise de Permutation Importance no conjunto de teste independente revelou:
1. **`total_pre` é a feature mais importante com folga absoluta**: sua permuta causa o maior aumento no erro de previsão em ambos os instrumentos.
2. **Efeito de Regressão à Média e Efeito Teto/Piso**:
   - Estudantes com escores muito baixos no pré-teste apresentam maior margem para evolução pedagógica (ganho positivo acentuado).
   - Estudantes que já iniciam com escores muito altos sofrem efeito teto (ceiling effect) e flutuações estatísticas que resultam em ganhos nulos ou pequenas reduções.
   - Essa correlação negativa entre escore inicial e ganho é um fenômeno psicométrico amplamente documentado na literatura de avaliação educacional e foi capturado com sucesso pelos modelos.

---

## 5. Limitações e Recomendações

1. **Variância Residual Não Explicada**: Mais de 60% da variância do ganho no TDE II e quase 90% no Vocabulário decorrem de fatores não observados na base (como frequência às aulas, engajamento individual do estudante, qualidade da mediação docente e eventos extraescolares).
2. **Não Causalidade**: A importância preditiva de uma feature (como `escola` ou `total_pre`) reflete correlação estatística para previsão, e **nunca** deve ser interpretada como relação causal direta.
3. **Utilidade Prática**: O modelo é valioso como ferramenta de triagem pedagógica preliminar (Early Warning), permitindo identificar turmas ou perfis com maior risco de estagnação ou declínio de pontuação antes da aplicação final.
