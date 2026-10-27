# Briefing: concluir a Mineração de Dados do projeto wordgen

*Para o agente de código (Google Antigravity) · Trabalho de Inteligência Artificial (UFPA), entrega em 03/11/2026 · Escrito em 26/10/2026*

## 1. Objetivo

Concluir de forma definitiva a Mineração de Dados, corrigindo os scripts que já existem em `pipeline/` (sem reescrevê-los do zero). O resultado deve ser:

- `dataset_vocabulario` e `dataset_tde`: um registro por aluno, fase e período, limpo e sem dados pessoais.
- `dim_aluno`: uma linha por aluno com o socioeconômico minimizado.
- `pareado_vocabulario` e `pareado_tde`: pré e pós do mesmo aluno, prontos para effect size.
- Um relatório de qualidade e testes de contrato que provem que tudo está correto.

O usuário precisa disso hoje para seguir com análise, dashboard (Streamlit + Plotly), modelos e agente de IA.

## 2. Regras inegociáveis

**Privacidade e segurança**

- A pasta `data/` contém dados pessoais de menores e de adultos. Não abra, não imprima, não resuma e não cole o conteúdo de arquivos de `data/` em nenhum lugar (logs, comentários, testes, commits, respostas). Rode os scripts e leia apenas saídas agregadas (contagens).
- Não leia nem altere `.env`. Não faça chamadas de rede e não adicione telemetria.
- Testes usam somente fixtures sintéticas em `tests/fixtures/` (inventadas, sem dados reais).
- Se `data/` não estiver disponível no seu workspace, implemente tudo com fixtures sintéticas e deixe `pipeline/run_all.py` pronto: o usuário executará localmente e entregará o relatório agregado.

**Git**

- Trabalhe na branch `feat/correcoes-pipeline`, com um commit por tarefa (mensagens no padrão `tipo(escopo): descrição`).
- Nunca faça `git add data/` nem `.env`. Não use `--no-verify`, não edite `.pre-commit-config.yaml` nem enfraqueça o `.gitignore`. Não faça push para `main` nem force-push. Se um hook modificar arquivos, rode `git add` de novo e repita o commit.

**Código**

- Windows 11, PowerShell, `uv run`, Python 3.12. Comentários e mensagens em português, identificadores no estilo já usado no projeto, UTF-8.
- Caminhos sempre relativos (`Path(__file__)`), sem caminhos absolutos. Resultado determinístico: listas de arquivos ordenadas e chaves de ordenação fixas.
- Código novo em `src/` deve passar em `ruff check` e `ruff format` e ter testes. `pipeline/` e `notebooks/` estão excluídos do Ruff.
- Se algo for ambíguo, escolha o padrão definido neste briefing, registre em `docs/decisions/003-mineracao-final.md` e continue sem esperar resposta.

## 3. Estado atual

- Dados brutos: `data/raw/Dados Word Gen/saida_csv_anonimizada/` (209 CSVs, separador `;`, UTF-8 com BOM). Saídas atuais: `data/processed/` (referência da última execução no Windows: vocabulário 10.593 linhas, TDE 12.033, socioeconômico 3.895, após `fix_escolas.py`). Cópia antiga de referência em `data/legado/`.
- `pipeline/pipeline_datasets.py`: gera `dataset_vocabulario.csv`, `dataset_tde.csv`, `dataset_socioeconomico.csv` e `log_processamento.csv`. Pontos a alterar: sete laços `glob('*.csv')` (linhas \~425 a \~545), três `drop_duplicates(... keep='first')` (linhas \~586, \~601, \~630) e o cálculo de `total_*` nas funções `normalizar_vocabulario` e `normalizar_tde`.
- `pipeline/fix_escolas.py`: a linha 29 (`df = df[df['escola'].notna()]`) apaga registros sem escola.
- Colunas atuais dos datasets: `aluno_id, matricula, escola, turma, fase, periodo, ano_semestre, monitor, coordenador, sexo, data_nascimento, idade, cor_raca, zona, necessidade_deficiencia, responsavel_legal, total_*`, mais os itens `Q1..Q50` (vocabulário) ou `P1..P40` (TDE).
- Funções prontas e testadas em `src/wordgen/domain/padronizacao.py`: `classificar_celula`, `padronizar_turma`, `padronizar_escola` (com `REGRAS_ESCOLA`), `parse_idade`, `normalizar_sexo`, `normalizar_id`, `ESCALA_MAXIMA`. Reutilize; não reimplemente. O arquivo `tests/unit/test_padronizacao.py` não existe: crie-o com casos parametrizados para essas funções.
- Regras do professor já aplicadas (não mudar): só vocabulário e TDE (a Fase 4 não tem pastas e entra por nome de arquivo); STARI, `_dfs` e controles excluídos. Período da Fase 4: prefixo `2` no nome = pós, marcador `(PRE)` = pré, demais = pré por padrão.

## 4. Problemas conhecidos (medidos)

- **Dados pessoais nas saídas:** matrícula, responsável legal, data de nascimento, nomes de monitor e coordenador.
- **`fix_escolas.py` apaga registros:** o vocabulário cai de cerca de 11,5 mil para 10,6 mil linhas. Na Fase 3 pós-teste de vocabulário, \~544 dos \~1.286 alunos com respostas válidas somem.
- **Deduplicação instável:** mantém o primeiro registro da lista de arquivos, que depende da ordem do sistema operacional, e descarta respostas válidas em favor de linhas vazias (\~52 registros no vocabulário e \~113 no TDE).
- **Valores inválidos nos itens:** números fora da escala (ex.: 9, 11), letras (A, B, C, D, M), traços, anotações e erros de planilha (`#REF!`, mais de 64 mil células em 5 arquivos de TDE). Cerca de 25% das linhas não têm nenhuma resposta. A coluna Total diverge da soma dos itens em \~3,5% das linhas.
- **Turmas e escolas:** turmas fora do padrão (`7º B`, `8º ANO - C`, `6º período (7º ano)`), cerca de 45 valores distintos. A coluna escola tem lixo (números, barras, até um nome de pessoa).
- **Socioeconômico:** só existe nas Fases 2 e 3, então a Fase 4 precisa recebê-lo por aluno. `cor_raca` vale `0` em \~32% das linhas, `zona` tem variantes, datas de nascimento vêm como número serial do Excel e idades vão de 1 a 34.

## 5. Tarefas

Cada tarefa termina em commit. A T2 vem antes da T3 porque a deduplicação depende de saber quantas respostas válidas cada registro tem.

### T1. Linha de base e orquestrador

- Execute o pipeline atual e registre as contagens iniciais (linhas por dataset e por fase e período) para o relatório, como 'antes'.
- Crie `pipeline/run_all.py`: executa em ordem `pipeline_datasets.py`, `fix_escolas.py` e `finalizar_datasets.py` com `sys.executable`, para no primeiro erro e imprime o tempo total.
- **Pronto quando:** `uv run python pipeline/run_all.py` roda do início ao fim.

### T2. Limpeza de valores (antes da deduplicação)

- Em `normalizar_vocabulario` e `normalizar_tde` (ou imediatamente antes do `drop_duplicates`), converta cada item com `classificar_celula` (máximo 1 no vocabulário, 2 no TDE). Só o valor com flag `ok` permanece; o restante vira NaN. Conte as células limpas por flag.
- Recalcule `total_vocabulario` e `total_tde` como a soma dos itens válidos (NaN se não houver nenhum). Não confie na coluna Total do arquivo de origem.
- Crie `n_itens`, `n_validos`, `pct_completude` (`n_validos / n_itens`) e `completo` (verdadeiro se `pct_completude >= 0.80`).
- **Pronto quando:** todo item é NaN ou pertence a {0,1} (vocabulário) ou {0,1,2} (TDE).

### T3. Deduplicação determinística

- Troque todos os `glob('*.csv')` por `sorted(...)`.
- Registre `arquivo_id` (os 12 primeiros caracteres do SHA-256 do conteúdo do arquivo, nunca o nome, que contém nomes de pessoas) e `linha_origem`.
- Antes de cada `drop_duplicates`, ordene por `aluno_id, fase, periodo` e depois por `n_validos` decrescente, `arquivo_id` e `linha_origem` crescentes, e mantenha o primeiro. No socioeconômico, o critério é o número de campos preenchidos.
- Conte as chaves com conflito (dois ou mais registros com respostas válidas e itens divergentes) e informe o número no relatório. Não apague nada em silêncio.
- Crie `periodo_fonte` com os valores `pasta` (Fases 2 e 3), `prefixo_2`, `marcador_pre` ou `padrao_pre` (Fase 4).
- Depois da deduplicação, remova os registros com `n_validos == 0` e conte quantos.
- **Pronto quando:** duas execuções seguidas, e uma execução com a ordem dos arquivos invertida (parâmetro ou variável de ambiente só de teste), geram arquivos com o mesmo hash; e nenhuma chave mantém registro vazio quando existe outro com resposta.

### T4. Escolas

- Em `fix_escolas.py`, pare de apagar linhas. Normalize com `padronizar_escola` (6 escolas canônicas); qualquer outro texto (números, barras, nomes de pessoas) vira nulo.
- Preencha a escola nula com a moda das escolas do mesmo aluno em outros registros; o que restar vira `NAO INFORMADA`.
- **Pronto quando:** linhas antes é igual a linhas depois; `escola` pertence às 6 canônicas mais `NAO INFORMADA`; a Fase 3 pós de vocabulário tem cerca de 1.286 alunos com resposta válida (tolerância de 5%).

### T5. Turma

- Use `padronizar_turma` para criar `serie` (6 a 9 ou nulo), `turma_letra` e `turma` no formato `6C`. Sem letra, use `U` (turma única) e registre isso na decisão. 'Período' sem 'ano' não é série.
- **Pronto quando:** `turma` casa com `^[6-9][A-Z]$` ou é nula, com no máximo 25 valores distintos.

### T6. Privacidade e tabela de alunos (`pipeline/finalizar_datasets.py`)

- Leia as saídas, aplique T4 e T5 se ainda não aplicadas e grave em `data/final/` os arquivos `dataset_vocabulario`, `dataset_tde` e `dim_aluno`, cada um em CSV (UTF-8 com BOM) e Parquet.
- Os datasets de avaliação **não** podem conter: `matricula`, `responsavel_legal`, `data_nascimento`, `monitor`, `coordenador`, `idade`, `sexo`, `cor_raca`, `zona`, `necessidade_deficiencia`. A demografia fica só em `dim_aluno`.
- `dim_aluno` (uma linha por `aluno_id`, vinda de `dataset_socioeconomico`): `sexo` (M, F ou NAO INFORMADO); `faixa_etaria` (`<=10`, `11-12`, `13-14`, `15+` ou NAO INFORMADA, a partir de `idade` entre 8 e 20 da fase mais recente com idade válida); `cor_raca` em maiúsculas, com `0` virando NAO INFORMADO; `zona` (URBANA, RURAL ou NAO INFORMADA; só 'URBANA', 'RURAL' e 'ZONA RURAL' são reconhecidos); `necessidade_especial` (SIM se `necessidade_deficiencia` estiver preenchida, senão NAO\_REGISTRADO); `fonte_socio` (fases usadas). Conflitos entre fases: moda, e em empate a fase mais recente.
- **Pronto quando:** `aluno_id` casa com `^ALUNO_[0-9A-F]{16}$` e é único por (`aluno_id`, `fase`, `periodo`) nos datasets e único em `dim_aluno`; nenhuma coluna proibida existe; `dim_aluno` cobre pelo menos 90% dos alunos da Fase 4 (referência: 91%).

### T7. Tabelas pareadas

- Grave `pareado_vocabulario` e `pareado_tde` (Parquet): uma linha por aluno e fase com pré e pós, contendo `aluno_id, fase, escola, serie, turma, total_pre, total_pos, pct_pre, pct_pos, completo_pre, completo_pos`. Só entram alunos com pré e pós, ambos com pelo menos uma resposta válida.
- **Pronto quando:** as contagens e médias abaixo (apenas registros `completo` nos dois momentos) batem com a referência independente:

| Instrumento e fase | Alunos pareados | Média pré | Média pós |
| --- | --- | --- | --- |
| Vocabulário, Fase 2 | \~1.352 | \~26,6 | \~27,6 |
| Vocabulário, Fase 3 | \~872 | \~21,4 | \~21,3 |
| TDE, Fase 2 | \~1.245 | \~37,2 | \~34,6 |
| TDE, Fase 3 | \~1.545 | \~34,1 | \~35,3 |

Tolerância: 8% na contagem e 1,5 ponto nas médias. Fora disso, investigue e explique no relatório; não ajuste os testes para passar. Referência de alunos únicos com pelo menos uma resposta válida: \~2.570 no vocabulário e \~2.499 no TDE (tolerância de 3%).

### T8. Relatório, testes e documentação

- `data/final/relatorio_qualidade.md` (apenas dados agregados, sem nenhum registro individual): contagens antes e depois de cada etapa; registros e alunos por instrumento, fase e período; células limpas por tipo de flag; registros vazios removidos; chaves duplicadas e conflitos; escolas e turmas distintas; cobertura de `dim_aluno` por fase; tabela de pareados com médias; desvios em relação às referências deste briefing.
- `tests/data/test_contrato_datasets.py`: verifica todos os critérios 'Pronto quando' acima e é pulado (`skip`) quando `data/final` não existe. Mais `tests/unit/test_padronizacao.py` e testes unitários das funções novas com fixtures sintéticas.
- `docs/decisions/003-mineracao-final.md` (decisões e suposições) e uma seção 'Como reproduzir' no `README.md`.
- **Pronto quando:** `uv run ruff check .`, `uv run pytest` e `uv run python pipeline/run_all.py` passam.

## 6. Fora de escopo

Não faça análise de dados, effect size, dashboard, modelos nem agente. Não reescreva o pipeline do zero, não altere a regra de período da Fase 4 (apenas registre `periodo_fonte`), não mude o código do M1 e do M2 além de reutilizá-lo, e não adicione dependências sem necessidade.

## 7. Relatório final ao usuário

Ao terminar, entregue um resumo curto: commits criados, comandos executados, números principais do relatório, desvios das referências e decisões tomadas. Sem dados individuais.
