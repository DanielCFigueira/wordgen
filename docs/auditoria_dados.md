# Dossiê de auditoria dos dados WordGen

**Escopo:** perfil linha a linha de todos os CSVs extraídos, conferência do ZIP e manifesto, além da leitura dos arquivos de código, configuração, documentação e teste presentes no projeto. Valores pessoais/respostas individuais não são reproduzidos.

## Resumo executivo

- CSVs examinados: **209** (209 lidos; 0 com erro). O ZIP contém **209** itens (19,926,144 bytes descompactados); arquivos CSV do ZIP e extração coincidem byte a byte: **True**.
- Linhas não vazias somadas entre os arquivos CSV: **60,457**. Este total inclui sobreposições, formulários e arquivos particionados; não representa pessoas únicas.
- Manifesto: 420 registros, status não `ok`: 0; saídas únicas esperadas: 420, presentes: 208, ausentes: 212 (Fase 3: 1; Fase 4: 211).
- Fase por quantidade de CSVs: Fase 2: 68, Fase 3: 61, Fase 4: 79, Raiz/manifesto: 1.
- Classificação preliminar por nome/pasta: vocabulário 98 arquivos; TDE 88; saídas TDE de resumo/dashboard 3; demais auxiliares 19.
- Tamanho total dos CSVs: 19,926,144 bytes. Inventário completo por arquivo em [`inventario_dados.csv`](inventario_dados.csv).

## Organização e significado provável

O material combina fases 2, 3 e 4, coletas pré e pós intervenção, instrumentos de vocabulário, TDE II (escrita), DFS e controles de aplicação. A nomenclatura sugere planilhas de entrada por monitor/turma, exportações por aba e cópias agregadas. Confirme os significados exatos com o dicionário/ protocolo do estudo antes de interpretar escores.

A Fase 2 inclui bases de cadastro e pré/pós-teste, frequentemente divididas por monitor, além de planilhas de escola. Fases 3 e 4 incluem formulários de turma e registros de avaliação organizados por escola, monitor e instrumento. O manifesto documenta origem XLSX, aba, saída CSV, quantidade de linhas, células alvo da anonimização e linha de cabeçalho.

## Classificação para os dois datasets

O inventário marca cada CSV com `familia_candidata`, usando localização/nome do arquivo como regra inicial. Os candidatos identificados são **98 arquivos de Vocabulário** e **88 arquivos de TDE com dados de avaliação**, além de **3 exportações TDE de resumo/dashboard** (uma é dashboard agregado; duas são cópias/summary com risco de sobreposição). Esta classificação deve ser confirmada por conteúdo/cabeçalho antes da ingestão; não equivale a 189/91 bases independentes.

Arquivos DFS, controle de aplicação, acompanhamento semanal da intervenção e listas/cadastros de alunos foram marcados como auxiliares. Podem fornecer dimensões/metadados ou documentação do tratamento, mas não devem ser anexados às respostas de Vocabulário/TDE como se fossem avaliações daquele instrumento. O manifesto também é metadado, não dado de aluno para o dataset analítico.

## Contexto do projeto de engenharia

O README está vazio e não há dicionário de dados nem documentação de pipeline. O código atual é um esqueleto: define categorias gerais de instrumento (vocabulário, TDE, outro), momento (pré, pós, desconhecido) e normalização textual de caminhos; a entrada da aplicação ainda imprime uma mensagem de exemplo. O teste existente verifica apenas importação de dependências. Portanto, não há no código atual regras de leitura, validação, anonimização ou transformação que expliquem completamente a geração do acervo.

As dependências declaradas incluem pandas, DuckDB, Pandera e PyArrow, compatíveis com uma futura camada tabular/analítica. O `.gitignore` exclui `data/`, ZIP e formatos analíticos; o `.gitattributes` preserva CSV como bytes sem conversão de finais de linha. Como os dados contêm informação pessoal, manter essa exclusão e controlar cópias e backups é adequado.

## Qualidade estrutural

- Arquivos com larguras de linha divergentes do cabeçalho: **0**. Consulte `linhas_largura_irregular` e `primeiras_linhas_irregulares` no inventário.
- Arquivos com linhas integralmente repetidas não vazias: **103**, totalizando 1,288 linhas excedentes repetidas. Linhas de preenchimento totalmente vazias foram contabilizadas separadamente e excluídas desta métrica. Em planilhas particionadas, cópias repetidas podem ser legítimas ou duplicação; investigar antes de deduplicar.
- Linhas totalmente vazias nos arquivos somadas: **37,108**; comuns em exportações com intervalo de planilha estendido.
- Arquivos com cabeçalho duplicado/vazio ou colunas totalmente sem dados: **81**.
- Arquivos com pelo menos 50% de células vazias: **43**. Em formulários de avaliação, vazio pode significar não aplicado, ausência ou resposta em branco; não imputar como erro/incorreto sem regra do protocolo.
- Formatos de entrada incluem CSV delimitado por ponto e vírgula, cabeçalhos com variações de caixa/acentos e diferentes nomes para o mesmo conceito. Algumas planilhas parecem ter linhas introdutórias/legendas em vez de uma tabela retangular; validar a linha de cabeçalho por arquivo.
- **Divergência manifesto/pacote:** o manifesto declara 420 caminhos de saída distintos, dos quais 212 não estão presentes na pasta/ZIP. Os status `ok` do manifesto indicam que o processamento foi marcado como bem-sucedido, mas não que o arquivo foi incluído no pacote atual. Verificar se a extração foi intencionalmente parcial ou se faltam entregáveis.
- Há colunas de escores totais junto das respostas por item. Recalcular totais a partir dos itens e comparar, respeitando itens omitidos e escalas distintas (vocabulário pode usar 0/1; TDE pode usar 0/1/2). O inventário inclui, por arquivo, faixas numéricas observadas e cardinalidade de campos textuais; são perfis observados, não validações de domínio.

## Identidade, privacidade e chaves

O manifesto declara anonimização para células em colunas de nome. Os CSVs usam tokens do tipo `ALUNO_<hash>` em vários conjuntos; isso é pseudonimização, não anonimização irreversível. O próprio cadastro e algumas avaliações ainda carregam atributos sensíveis ou identificadores indiretos: escola/turma, sexo, data de nascimento, idade, responsável legal, raça/cor, zona, necessidade/deficiência e, em certas fases, matrícula. Há risco de reidentificação por combinação desses campos. O manifesto e a cópia CSV do cadastro devem ter acesso restrito.

Não encontramos no manifesto garantia explícita de anonimização de matrícula, nome de responsável ou demais identificadores diretos. Trate o acervo como dado pessoal educacional sensível; para análise compartilhada, gere uma camada derivada minimizada, com chave aleatória protegida, idade/faixa etária no lugar de nascimento e supressão de pequenas células.

## Ligações entre tabelas e desenho longitudinal

As colunas de aluno, escola, turma, monitor, fase e instrumento parecem potenciais chaves de ligação. Antes de juntar, medir duplicidade e estabilidade dos tokens entre arquivos: a presença de um token aluno não garante que a mesma função/hash tenha sido aplicada em todas as fases. Matrícula pode aparecer em notação científica em alguns CSVs, o que pode perder precisão; leia sempre como texto e compare com a origem. A repetição de alunos em diferentes abas e avaliações é esperada.

Para comparar pré e pós, construir primeiro uma tabela de mapeamento fase–instrumento–turma–arquivo, validar cohortes e datas e determinar se as observações são pareadas. Não inferir causalidade a partir de comparação simples entre fases; considerar seleção, perdas, diferenças de aplicação, turma/escola e monitor.

## Alertas prioritários

1. **Validar o processo de anonimização:** amostrar colunas diretas e procurar dados pessoais residuais em todas as fases, incluindo matrícula, responsável, observações e nomes em abas/arquivos. O manifesto atual só lista coluna(s) de nome por origem.
2. **Resolver identificadores:** documentar algoritmo/escopo do token ALUNO e verificar colisões, consistência entre pré/pós e fases, e se matrículas foram convertidas para notação científica.
3. **Harmonizar esquemas:** criar dicionário canônico de colunas e tipos, categorias de raça/cor, sexo, zona, escola, série/turma, instrumento, momento e semântica de missing.
4. **Validar escore:** regras por instrumento, máximo/mínimo, itens invertidos, total versus soma, zeros versus branco, códigos textuais (ex.: múltipla resposta) e datas.
5. **Controlar cobertura:** identificar sobreposição entre consolidados e arquivos por monitor; medir cobertura de itens/alunos por fase e perdas entre pré/pós sem contar duplicatas como participantes.
6. **Rastreabilidade:** conservar o ZIP original como fonte imutável; associar cada CSV ao arquivo/aba de origem por manifesto e registrar hashes/versão do pipeline em futuras extrações.

## Próxima sequência recomendada

1. Revisar anomalias listadas no inventário, começando por linhas irregulares, cabeçalhos vazios/duplicados e schemas únicos. 2. Aprovar dicionário de dados e regras de codificação com responsáveis pelo instrumento. 3. Criar camada bronze fiel ao arquivo, camada prata normalizada e tabela fato longa (`aluno`, `fase`, `momento`, `instrumento`, `item`, `resposta`), mantendo metadados de origem. 4. Gerar validações de domínio, consistência de totais, unicidade e pareamento longitudinal. 5. Produzir camada analítica desidentificada e controlar acesso à camada individual.

## Limitações desta auditoria

A contagem de linhas é estrutural e deliberadamente não equivale a amostra única. Sem dicionário de dados, protocolo de avaliação e documentação do algoritmo de anonimização, a interpretação semântica de categorias, códigos e comparabilidade entre fases permanece provisória. O inventário registra cabeçalhos e métricas por arquivo, sem valores de células.

## Arquivos de saída

- `docs/auditoria_dados.md`: este dossiê.
- `docs/inventario_dados.csv`: perfil de todos os CSVs, classificação preliminar de família, esquema, tamanho, missingness, faixas numéricas observadas, cardinalidades e anomalias estruturais.
