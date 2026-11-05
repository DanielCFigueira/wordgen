# WordGen

## Documentação técnico-acadêmica dos trabalhos de Inteligência Artificial

**Projeto:** WordGen - análise educacional, modelagem preditiva e assistente analítico  
**Contexto:** disciplina de Inteligência Artificial, Universidade Federal do Pará  
**Escopo deste documento:** registrar as soluções efetivamente implementadas no repositório, seus roadmaps, resultados comprovados e limitações.

## Resumo executivo

O WordGen foi estruturado como uma solução analítica em três camadas. A primeira consolida e qualifica avaliações educacionais de Vocabulário e TDE II - Escrita. A segunda estima, por regressão, o ganho entre pré e pós-teste. A terceira disponibiliza um assistente que combina recuperação documental, consultas quantitativas determinísticas e síntese opcional por LLM. O dashboard Streamlit integra a exploração dos resultados agregados, das análises pré/pós, da trajetória longitudinal, da qualidade dos dados e dos experimentos de modelagem.

As três entregas compartilham princípios de rastreabilidade, prevenção de vazamento, proteção de dados e distinção entre análises descritivas, preditivas e causais. Os resultados de modelagem são experimentais; não constituem diagnóstico, triagem pedagógica validada nem evidência causal.

## Visão de arquitetura

```text
Arquivos de avaliação autorizados
        |
        v
Inventário e classificação -> limpeza de itens -> deduplicação -> padronização
        |                                                        |
        v                                                        v
Datasets finais minimizados ------------------> notebooks analíticos e dashboard
        |                                                        |
        +--> pares pré/pós -> modelagem de regressão             |
        |                                                        |
        +--> artefatos analíticos e documentação -> RAG ---------+
```

---

# Trabalho 1 - Mineração e análise de dados

## 1. Problema e objetivo

O primeiro trabalho tratou da transformação de planilhas heterogêneas de avaliação em bases analíticas consistentes. Os instrumentos têm escalas diferentes: Vocabulário possui 50 itens binários; TDE II - Escrita possui 40 itens com valores de 0 a 2. As fases 2, 3 e 4 contêm aplicações de pré e pós-teste e podem registrar o mesmo estudante em mais de uma fase.

O objetivo operacional foi produzir bases consolidadas, auditáveis e minimizadas para análise estatística, dashboard e modelagem. A necessidade de engenharia de dados decorreu de erros de planilha, registros duplicados, grafias divergentes de escolas e turmas, campos ausentes e presença de atributos pessoais nas bases intermediárias.

## 2. Dados e organização

O briefing e a decisão de mineração registram 209 arquivos CSV brutos no fluxo original. O pipeline separa Vocabulário, TDE II e informações contextuais. Os datasets finais possuem, entre outros, `aluno_id` pseudonimizado, fase, período, escola, série, turma, itens, total, número de itens válidos, completude e tabelas pareadas.

Os dados finais são minimizados, mas não devem ser descritos como anonimização absoluta: o identificador persistente e a combinação de atributos contextuais requerem controle de acesso e aplicação das regras de privacidade. Campos como matrícula, responsável legal, data de nascimento, monitor e coordenador são removidos das tabelas de avaliação publicadas pelo pipeline.

## 3. Roadmap de mineração

| Etapa | Técnica implementada | Finalidade | Produto |
|---|---|---|---|
| 1. Inventário | classificação por caminho, nome e conteúdo; hash SHA-256 | identificar instrumento, fase, momento e duplicatas de arquivo | catálogo de arquivos e status de classificação |
| 2. Extração | leitura CSV tolerante a codificação e cabeçalhos | preservar a origem e normalizar a leitura | registros estruturados por instrumento |
| 3. Limpeza de itens | função `classificar_celula` | aceitar apenas a escala válida e registrar o tipo de problema | itens limpos, contagem de flags e totais recalculados |
| 4. Consolidação | chaves por estudante, fase e período | integrar aplicações comparáveis | datasets processados por instrumento |
| 5. Deduplicação | ordenação determinística e priorização de completude | evitar que uma linha menos preenchida substitua uma avaliação válida | uma linha por chave natural |
| 6. Padronização territorial | escolas canônicas e turma/série padronizadas | tornar filtros e comparações estáveis | escolas e turmas consistentes |
| 7. Minimização | remoção de PII e dimensão contextual | separar avaliação de atributos contextuais | `dim_aluno` e datasets finais |
| 8. Pareamento e qualidade | junção pré/pós por estudante e fase; validações | sustentar ganhos, effect size e modelagem | tabelas pareadas e relatório de qualidade |

## 4. Extração, padronização e tratamento

O pipeline reconhece itens Q01-Q50 de Vocabulário e P01-P40 do TDE. A limpeza classifica valores como válidos, vazios, fora de escala, erro de planilha ou texto. Somente valores 0/1 são aceitos em Vocabulário e 0/1/2 em TDE; o escore total é recalculado a partir dos itens válidos, em vez de confiar em totais legados. A completude é definida como pelo menos 80% dos itens válidos.

Escolas são convertidas para seis denominações canônicas. Registros sem escola podem receber a moda histórica observada para o mesmo estudante; sem informação disponível, tornam-se `NAO INFORMADA`. Séries válidas vão de 6 a 9; turmas são normalizadas para um formato série-letra, e a letra `U` representa turma única quando a letra não é identificada.

## 5. Deduplicação, validações e qualidade

A chave de deduplicação das avaliações é `(aluno_id, fase, periodo)`. Antes de manter uma ocorrência, o pipeline ordena por chave, número de itens válidos em ordem decrescente, identificador SHA-256 do arquivo e linha de origem. Assim, a escolha é reprodutível e privilegia a avaliação com mais respostas. Registros sem item válido são descartados após a deduplicação; conflitos com respostas divergentes são contabilizados.

As validações incluem a escala dos itens, a unicidade da chave, a forma de turma, a cobertura da dimensão e a compatibilidade dos pares. Os testes de contrato verificam, entre outros pontos, 50 itens em Vocabulário, 40 em TDE, escalas permitidas, ausência de colunas proibidas e chaves sem duplicidade.

## 6. Datasets finais e resultados comprovados

As saídas principais são `dataset_vocabulario`, `dataset_tde`, `dim_aluno`, `pareado_vocabulario` e `pareado_tde`, em CSV e Parquet no fluxo local. `dim_aluno` reúne sexo, faixa etária, cor/raça, zona e necessidade especial em um registro por estudante; conflitos entre fases são resolvidos por moda e, no empate, pela fase mais recente.

| Indicador validado | Resultado registrado |
|---|---:|
| Escolas canônicas | 6 |
| Turmas distintas observadas | 23 |
| Cobertura de `dim_aluno` na Fase 4 | 93,05% (1.766 de 1.898) |
| Vocabulário Fase 2 - pares completos | 1.352 |
| Vocabulário Fase 3 - pares completos | 872 |
| TDE II Fase 2 - pares completos | 1.245 |
| TDE II Fase 3 - pares completos | 1.545 |

As médias pré e pós correspondentes foram confrontadas com referências independentes descritas na decisão de mineração, com concordância de contagem e desvios mínimos de média. Esses resultados referem-se às bases preparadas para análise, não a inferências causais sobre intervenção.

## 7. Técnicas, ferramentas e impacto

Foram utilizadas Python, Pandas, NumPy, Parquet, hashes SHA-256, testes Pytest e regras de domínio próprias para classificação e padronização. A engenharia de dados reduz ambiguidade de arquivos, torna a deduplicação reprodutível, recalcula escores de forma coerente e evita que atributos pessoais entrem como variáveis de avaliação ou preditores. Ela também fornece a chave longitudinal necessária para pares pré/pós e divisão agrupada da modelagem.

## 8. Dashboard analítico

O dashboard Streamlit possui áreas para visão geral, Vocabulário, TDE II - Escrita, contexto dos estudantes, comparação pré/pós, acompanhamento longitudinal, qualidade e modelagem, além do assistente analítico. Os filtros incluem instrumento, fase, momento, série, escola e turma quando aplicáveis. As escalas brutas de Vocabulário e TDE não são combinadas.

A visualização contextual é descritiva. Grupos menores que 10 estudantes distintos são suprimidos e, quando uma categoria não satisfaz a regra, a decomposição inteira é ocultada para reduzir inferência por diferença. Ganhos são calculados apenas em pares pré/pós; o filtro de momento não é aplicado ao cálculo pareado, porque ambos os momentos são necessários.

---

# Trabalho 2 - Aprendizado de Máquina

## 1. Problema e objetivo

O segundo trabalho formula a previsão do ganho de desempenho como regressão supervisionada. Para cada instrumento, o alvo é:

```text
ganho = total_pos - total_pre
```

O objetivo é estimar esse ganho a partir de informações disponíveis antes do pós-teste, comparar algoritmos a referências simples e avaliar generalização sem compartilhar estudantes entre treino e teste.

## 2. Preparação e cenários

Somente pares completos, isto é, com pré e pós completos, são usados. Totais fora da escala são rejeitados. `aluno_id` é preservado exclusivamente para agrupamento e não é feature. `total_pos`, o próprio ganho e variáveis derivadas do pós-teste são excluídos, prevenindo vazamento de informação.

| Cenário | Variáveis utilizadas | Papel metodológico |
|---|---|---|
| Experimento A | `total_pre`, fase, escola, série, turma | cenário acadêmico principal |
| Experimento B | variáveis do A + sexo, faixa etária, cor/raça, zona e necessidade especial | cenário exploratório complementar |

O Experimento B é exploratório porque `dim_aluno` consolida atributos de múltiplas fases e não comprova a disponibilidade temporal individual de todos os atributos no momento da previsão.

## 3. Roadmap de modelagem

```text
Tabelas pareadas completas
        -> validação de escala e criação de ganho
        -> definição de features A/B e exclusão de pós-teste
        -> GroupShuffleSplit por estudante (80/20)
        -> pipelines de pré-processamento
        -> GridSearchCV com GroupKFold (5 folds)
        -> avaliação final no teste independente
        -> persistência de pipelines e metadados
```

Os pipelines usam imputação de mediana para a variável numérica e imputação categórica com `NAO_INFORMADO`, seguida de one-hot encoding. Para MLP, a variável numérica também recebe `StandardScaler`; Random Forest não requer escalonamento numérico.

## 4. Algoritmos, baselines e validação

Foram implementados `RandomForestRegressor` e `MLPRegressor` do Scikit-learn. Random Forest foi ajustado sobre grade de número de árvores, profundidade máxima e folhas mínimas. A MLP avaliou topologias, regularização L2 e taxa de aprendizado; usa ReLU, Adam e `early_stopping=False`. Esta última decisão evita uma divisão interna aleatória que não respeitaria os grupos de estudante.

As referências são: Baseline 1, que prevê ganho zero; e Baseline 2, que prevê a média do ganho calculada somente no treino. A divisão final usa `GroupShuffleSplit` com aproximadamente 80%/20%; o ajuste usa `GroupKFold` com cinco folds. O pré-processamento pertence ao pipeline, sendo ajustado dentro de cada fold. A interseção treino-teste registrada é zero.

| Instrumento | Pares completos | Estudantes únicos | Treino | Teste |
|---|---:|---:|---:|---:|
| Vocabulário | 3.320 | 1.989 | 2.665 registros / 1.591 estudantes | 655 registros / 398 estudantes |
| TDE II | 3.796 | 2.069 | 3.050 registros / 1.655 estudantes | 746 registros / 414 estudantes |

## 5. Métricas e resultados de teste

MAE mede o erro absoluto médio, RMSE penaliza erros maiores e R² resume a variação explicada em comparação com uma referência de média. Menor MAE é o critério prioritário de comparação; R² não é interpretado isoladamente.

| Instrumento | Cenário | Modelo | MAE | RMSE | R² |
|---|---|---|---:|---:|---:|
| Vocabulário | A | Baseline média | 4,5313 | 6,1099 | -0,0001 |
| Vocabulário | A | Random Forest | 4,3669 | 5,7665 | 0,1092 |
| Vocabulário | A | MLP | 4,3076 | 5,6911 | 0,1323 |
| Vocabulário | B | Random Forest | 4,3354 | 5,7325 | 0,1196 |
| Vocabulário | B | MLP | 4,6769 | 6,0247 | 0,0276 |
| TDE II | A | Baseline média | 5,7995 | 9,3291 | -0,0020 |
| TDE II | A | Random Forest | 4,7626 | 7,5154 | 0,3498 |
| TDE II | A | MLP | 4,8337 | 7,5063 | 0,3513 |
| TDE II | B | Random Forest | 4,7578 | 7,4824 | 0,3555 |
| TDE II | B | MLP | 5,1011 | 7,6570 | 0,3250 |

Nos resultados reportados, Random Forest supera os baselines em todos os quatro cenários. A MLP supera o baseline de média nos cenários A e no TDE II/B, mas não no Vocabulário/B. O melhor MAE de Vocabulário é 4,3076 (MLP/A); o melhor MAE de TDE II é 4,7578 (Random Forest/B). O TDE II apresenta R² superior ao Vocabulário nessa partição, mas isso não autoriza comparar as escalas brutas dos instrumentos como se fossem iguais.

Os MAEs médios da validação cruzada para Random Forest foram 4,2901, 4,2728, 4,7776 e 4,8113, respectivamente para Vocabulário A, Vocabulário B, TDE II A e TDE II B. A validação cruzada é apresentada separadamente do teste final.

## 6. Predições, interpretação e agrupamento

As predições são estimativas numéricas de ganho no instrumento correspondente, não classificações de estudantes. Um ganho previsto positivo indica aumento esperado de pontuação, zero indica estabilidade estimada e negativo indica redução estimada; a incerteza individual permanece relevante.

`total_pre` apresentou a maior importância por permutação nos resultados documentados. Esse padrão é compatível com regressão à média e com efeitos de teto/piso, mas não prova que o escore inicial cause o ganho. Escola, turma ou atributos contextuais também não devem receber interpretação causal.

**Agrupamento não supervisionado:** não há implementação de K-Means, DBSCAN, clustering hierárquico ou outro algoritmo de agrupamento no repositório. Portanto, não existem número de clusters, rótulos de grupos ou métricas de clustering a reportar. O agrupamento efetivamente usado é apenas o agrupamento por `aluno_id` na validação, com a finalidade de impedir vazamento entre treino e teste.

## 7. Limitações

Os modelos representam somente estudantes com pares completos: as conclusões não se generalizam automaticamente para avaliações incompletas ou ausência de pós-teste. Permanecem fatores não observados, variância residual e limitação temporal dos atributos do Experimento B. As previsões são associativas, não causais. O projeto registra Early Warning como possibilidade futura; antes de qualquer uso operacional seriam necessárias validação em novos dados, análise por grupos, auditoria de viés, confirmação temporal de atributos e avaliação de utilidade pedagógica.

---

# Trabalho 3 - Chatbot com RAG e LLM

## 1. Problema e objetivo

O terceiro trabalho implementa o Assistente Analítico WordGen para responder perguntas sobre instrumentos, dados, metodologia, resultados e limitações. A solução evita usar geração livre para números: indicadores, effect size, trajetórias e métricas de modelagem são obtidos por consultas fechadas a artefatos analíticos. Conceitos e metodologia são respondidos por recuperação documental.

## 2. Base de conhecimento e preparação documental

O notebook `06_base_analitica_rag.ipynb` produz a base `base_conhecimento_wordgen.json`, acompanhada de `documentos_rag.md` e de um benchmark de perguntas e respostas. A base contém 10 chunks semânticos: visão geral; três por fase para Vocabulário; três por fase para TDE II; transições longitudinais; equidade territorial; e perfil sociodemográfico/inclusão.

Além desses chunks, o agente indexa seções relevantes da documentação de modelagem: estratégia de validação, definição do problema e avaliação dos modelos. Cada documento preserva identificador, título, categoria, tags, metadados e origem. Assim, a resposta apresenta proveniência, em vez de inventar referências.

## 3. Roadmap do assistente

```text
Notebook analítico 06 e documentação de modelagem
        -> chunks com metadados e origem
        -> vetorização TF-IDF e índice persistente Joblib
        -> classificação de escopo e privacidade
        -> rota analítica determinística ou recuperação RAG
        -> síntese opcional com Groq, quando configurado
        -> resposta com fontes, aviso e histórico de sessão
```

## 4. Indexação e recuperação

O índice usa `TfidfVectorizer` com normalização, remoção de acentos, bigramas, sublinear TF e stopwords em português. A assinatura SHA-256 do conjunto documental controla a reutilização do índice: quando conteúdo ou versão mudam, o índice é reconstruído. Na busca, o cosseno TF-IDF implícito pelo produto da matriz é ordenado por relevância; resultados abaixo do limiar são excluídos, e o documento de visão geral recebe limiar mais rigoroso para não funcionar como resposta universal.

O RAG pode retornar ausência de evidência quando não encontra trecho pertinente. Perguntas fora do domínio são recusadas antes da busca. Perguntas ambíguas solicitam esclarecimento. Solicitações de identificadores, registros individuais ou valores pessoais são bloqueadas por privacidade.

## 5. Consultas analíticas determinísticas

O roteador identifica perguntas de KPI, modelagem, effect size, médias pré/pós, transições longitudinais e algumas consultas de equidade. Essas respostas lêem diretamente CSVs agregados, como `dashboard_kpis.csv`, `metricas_effect_size.csv`, `metricas_longitudinais.csv`, `metricas_equidade.csv` e o relatório comparativo de modelagem. O LLM não é autorizado a gerar livremente os números dessas consultas.

## 6. Integração com Groq e interface

`ClienteGroq` lê `GROQ_API_KEY` somente do ambiente. Quando há chave, envia a pergunta e o contexto recuperado ao cliente Groq, com temperatura zero e instrução de responder apenas com base no contexto. Falhas de configuração, autenticação, modelo, limite ou serviço são convertidas em mensagens controladas. Sem chave ou sem disponibilidade do serviço, o agente retorna o trecho recuperado, desde que haja evidência suficiente.

O assistente está integrado ao Streamlit como a página “Assistente analítico”. A interface mantém histórico durante a sessão, exibe fontes e informa indisponibilidade de artefatos ou LLM sem derrubar o dashboard.

## 7. Avaliação e resultados

O benchmark offline tem 12 perguntas de referência. A avaliação verifica se a resposta existe, contém fonte esperada, apresenta valores esperados quando aplicável e usa rota adequada. Os testes automatizados registram 12 de 12 perguntas respondidas, com fonte identificada e aprovadas no benchmark local. Isso demonstra cobertura técnica do benchmark, não substitui avaliação humana de qualidade semântica de respostas geradas por LLM.

Os testes também cobrem recuperação e fonte, reconstrução do índice, consultas a métricas reais, recusa de perguntas fora de escopo, perguntas ambíguas, ausência de evidência, bloqueio de pedidos individuais e tratamento de indisponibilidade da Groq.

## 8. Limitações

O mecanismo de recuperação é lexical TF-IDF, não um modelo de embeddings densos. A geração depende de serviço externo e configuração de ambiente; por isso a solução mantém caminho offline extrativo. A qualidade factual depende da cobertura dos documentos e artefatos publicados. O agente não acessa registros individuais, não deve ser usado para decisões pedagógicas e não substitui profissionais da educação.

---

# Evidências consultadas

| Evidência | Uso neste documento |
|---|---|
| `README.md` | arquitetura geral, execução, dashboard e integração do assistente |
| `docs/decisions/003-mineracao-final.md` | decisões, regras de limpeza, validação e resultados de mineração |
| `pipeline/pipeline_datasets.py`, `fix_escolas.py`, `finalizar_datasets.py`, `run_all.py` | fluxo operacional de engenharia de dados |
| `notebooks/analises/01` a `06` | escopo das análises descritivas, effect size, longitudinal, contexto, dashboard e RAG |
| `src/wordgen/dashboard/` | páginas, filtros, supressão e exibição de artefatos |
| `docs/modelagem/definicao_problema.md` | alvo, features e caráter exploratório do Experimento B |
| `docs/modelagem/estrategia_validacao.md` | GroupShuffleSplit, GroupKFold, baselines e tamanhos de partição |
| `docs/modelagem/desenvolvimento_modelos.md` | grades, hiperparâmetros e MAE de validação cruzada |
| `docs/modelagem/avaliacao_modelos.md` | métricas de teste, interpretação e limitações |
| `src/wordgen/modelagem/` | preparação, pipelines, treinamento, métricas e validação implementados |
| `data/rag/base_conhecimento_wordgen.json`, `benchmark_qa_ground_truth.json` | 10 chunks de conhecimento e benchmark do agente |
| `src/wordgen/agente/` e `tests/test_agente.py` | RAG TF-IDF, orquestração, Groq, privacidade e testes |
| `pyproject.toml` e `uv.lock` | dependências declaradas e ambiente Python |

## Conclusão

O WordGen implementa uma cadeia coerente entre preparação de dados, análise, regressão supervisionada e assistência informacional. A contribuição central da primeira etapa é tornar as bases rastreáveis e aptas a comparações; a segunda demonstra desempenho preditivo acima dos baselines em cenários específicos; a terceira torna métodos e resultados consultáveis com fontes e rotas controladas. As limitações de cobertura, temporalidade, generalização e não causalidade permanecem parte essencial da interpretação acadêmica do projeto.
