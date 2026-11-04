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
| **Vocabulário** | Exp A (Acadêmico) | Rede Neural (MLP) | 4,3076 | 5,6911 | 0,1323 | 0,5802 | 0,6587 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico/Contextual) | Baseline 1 (Ganho Zero) | 4,5557 | 6,1371 | -0,0090 | 0,5802 | 0,0000 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico/Contextual) | Baseline 2 (Média Treino) | 4,5313 | 6,1099 | -0,0001 | 0,5802 | 0,6390 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico/Contextual) | **Random Forest (Melhor)** | **4,3354** | **5,7325** | **0,1196** | 0,5802 | 0,5677 |
| **Vocabulário** | Exp B (Acadêmico + Demográfico/Contextual) | Rede Neural (MLP) | 4,6769 | 6,0247 | 0,0276 | 0,5802 | 0,8931 |
| **TDE II** | Exp A (Acadêmico) | Baseline 1 (Ganho Zero) | 5,7708 | 9,3246 | -0,0010 | -0,2936 | 0,0000 |
| **TDE II** | Exp A (Acadêmico) | Baseline 2 (Média Treino) | 5,7995 | 9,3291 | -0,0020 | -0,2936 | 0,1203 |
| **TDE II** | Exp A (Acadêmico) | **Random Forest** | **4,7626** | **7,5154** | **0,3498** | -0,2936 | -0,0545 |
| **TDE II** | Exp A (Acadêmico) | Rede Neural (MLP) | 4,8337 | 7,5063 | 0,3513 | -0,2936 | -0,0043 |
| **TDE II** | Exp B (Acadêmico + Demográfico/Contextual) | Baseline 1 (Ganho Zero) | 5,7708 | 9,3246 | -0,0010 | -0,2936 | 0,0000 |
| **TDE II** | Exp B (Acadêmico + Demográfico/Contextual) | Baseline 2 (Média Treino) | 5,7995 | 9,3291 | -0,0020 | -0,2936 | 0,1203 |
| **TDE II** | Exp B (Acadêmico + Demográfico/Contextual) | **Random Forest (Melhor)** | **4,7578** | **7,4824** | **0,3555** | -0,2936 | -0,0599 |
| **TDE II** | Exp B (Acadêmico + Demográfico/Contextual) | Rede Neural (MLP) | 5,1011 | 7,6570 | 0,3250 | -0,2936 | -0,2548 |

---

### Métricas MLP substituídas após a correção

Os valores anteriores do MLP foram substituídos após nova busca agrupada com `early_stopping=False`. A tabela abaixo preserva a rastreabilidade; somente os valores à direita são resultados atuais.

| Cenário | Métricas anteriores (MAE / RMSE / R²) | Métricas atuais (MAE / RMSE / R²) |
| :--- | :--- | :--- |
| Vocabulário — Exp A | 4,3878 / 5,7684 / 0,1086 | **4,3076 / 5,6911 / 0,1323** |
| Vocabulário — Exp B | 4,4255 / 5,7782 / 0,1056 | **4,6769 / 6,0247 / 0,0276** |
| TDE II — Exp A | 4,9375 / 7,7009 / 0,3173 | **4,8337 / 7,5063 / 0,3513** |
| TDE II — Exp B | 5,0896 / 7,6842 / 0,3202 | **5,1011 / 7,6570 / 0,3250** |

---

## 3. Interpretação e Principais Achados

### 3.1. Comparação com os Baselines
- Os Random Forests superam os baselines em ambos os instrumentos e experimentos.
- Os MLPs atuais superam os baselines de média no Experimento A. No Experimento B, o MLP de Vocabulário não supera o baseline de média em MAE; o MLP de TDE II supera-o.
- No TDE II, o Random Forest do Experimento B reduziu o MAE de 5,77 para **4,76 pontos** e alcançou $R^2 = 0{,}3555$. O MLP atualizado nesse cenário apresentou MAE **5,1011** e $R^2 = 0{,}3250$.

### 3.2. Comparação entre Random Forest e MLPRegressor
- O Random Forest tem MAE menor que o MLP nos dois cenários B e no TDE II/Experimento A. No Vocabulário/Experimento A, o MLP tem métricas melhores nesta partição de teste.
- Os resultados não sustentam uma conclusão universal de superioridade de um algoritmo para todos os cenários. A comparação deve considerar em conjunto MAE, RMSE e R² e a partição específica.

### 3.3. Comparação exploratória dos Experimentos A e B
- Nos Random Forests, o R² do Experimento B é ligeiramente maior que o do A: Vocabulário, 0,1092 para 0,1196; TDE II, 0,3498 para 0,3555.
- Essa diferença não pode ser atribuída isoladamente às variáveis demográficas/contextuais. `dim_aluno` agrega registros de várias fases e a disponibilidade temporal dos atributos por observação ainda não foi comprovada; portanto, a comparação A/B é exploratória até haver atributos point-in-time.

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

1. **Restrição de Amostra (Pares Completos)**: Os modelos foram treinados exclusivamente com estudantes que possuem **avaliações completas em pré-teste e pós-teste** (pares completos). Estudantes com avaliações parciais (< 80% de itens respondidos) ou sem pós-teste foram excluídos. Portanto, os resultados e previsões **não devem ser generalizados automaticamente** para estudantes com avaliações incompletas ou para aqueles sem dados de pós-teste disponíveis. A população modelada representa 97,1% (Vocabulário) e 96,2% (TDE II) dos registros brutos pareados.
2. **Variância Residual Não Explicada**: Mais de 60% da variância do ganho no TDE II e quase 90% no Vocabulário decorrem de fatores não observados na base (como frequência às aulas, engajamento individual do estudante, qualidade da mediação docente e eventos extraescolares).
3. **Não Causalidade**: A importância preditiva de uma feature (como `escola` ou `total_pre`) reflete associação estatística para previsão e não deve ser interpretada como relação causal.
4. **Uso futuro como Early Warning**: Os resultados justificam investigar essa possibilidade, mas os modelos são experimentais e ainda não estão validados para orientar triagem ou decisões pedagógicas reais. Antes de qualquer uso, são necessárias validação em novos dados, avaliação de desempenho por grupos demográficos, auditoria de possíveis vieses, confirmação da disponibilidade temporal das variáveis e avaliação da utilidade pedagógica dos alertas. Os modelos não substituem a avaliação dos profissionais da educação.
