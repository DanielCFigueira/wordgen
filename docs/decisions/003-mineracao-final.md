# 003 — Conclusão da Mineração de Dados, Padronização e Privacidade

## Contexto

O projeto WordGen continha scripts iniciais em `pipeline/` para consolidar avaliações de Vocabulário e TDE II (Escrita), além de dados socioeconômicos, a partir de 209 planilhas CSV brutos. A auditoria e inspeção inicial identificaram problemas críticos que impediam o avanço seguro para modelagem e dashboards:
1. **Presença de dados pessoais (PII)** nos datasets consolidados (`matricula`, `responsavel_legal`, `data_nascimento`, `monitor`, `coordenador`, etc.).
2. **Perda massiva de registros** em `fix_escolas.py` ao descartar linhas com escola não preenchida (queda de ~1.000 linhas no vocabulário e perda de 544 alunos na Fase 3 pós-teste).
3. **Deduplicação não-determinística** (`drop_duplicates(keep='first')`), dependente da ordenação arbitrária do sistema de arquivos e propensa a descartar registros preenchidos em favor de registros vazios.
4. **Valores espúrios nos itens** (erros de fórmula Excel `#REF!`, letras, traços e números fora da escala psicométrica).
5. **Inconsistência de turmas e escolas** com grafias variadas, lixo textual e séries ausentes.
6. **Ausência de dados demográficos na Fase 4**, exigindo a propagação controlada de atributos socioeconômicos dos mesmos alunos observados nas Fases 2 e 3.

## Decisões Tomadas

### 1. Arquitetura Modular e Determinismo Estrito
- O pipeline foi estruturado em três etapas orquestradas por `pipeline/run_all.py`:
  1. `pipeline_datasets.py`: extração, higienização de itens, rastreabilidade por SHA-256 (`arquivo_id`), e deduplicação determinística.
  2. `fix_escolas.py`: padronização de escolas em 6 escolas canônicas e imputação pela moda histórica do aluno, com preservação integral de linhas.
  3. `finalizar_datasets.py`: expurgo de PII nos datasets de avaliação, geração de `dim_aluno` minimizada, tabelas pareadas longitudinais e emissão de relatório de qualidade.
- Todos os laços de arquivos utilizam `sorted()`, permitindo validação com ordem reversa através de `WORDGEN_REVERSE_FILES=1`. Execuções em ordens distintas geram hashes SHA-256 idênticos byte a byte nos arquivos finais.

### 2. Limpeza de Células de Itens (`classificar_celula`)
- Função pura de domínio em `src/wordgen/domain/padronizacao.py`.
- Escalas aceitas: {0, 1} para Vocabulário e {0, 1, 2} para TDE.
- Flags atribuídas: `ok` (mantém o valor inteiro como float), `vazio` (None/vazio/traços), `fora_escala` (números negativos ou acima do limite), `erro_planilha` (fórmulas quebradas como `#REF!`) e `texto` (anotações/letras).
- O total do instrumento é estritamente recalculado pela soma dos itens válidos (`n_validos`), descartando totais legados inconsistentes.
- Definiu-se o indicador `completo` como `pct_completude >= 0.80` (pelo menos 80% dos itens válidos).

### 3. Deduplicação com Priorização de Respostas e Rastreabilidade
- Chave de deduplicação das avaliações: `(aluno_id, fase, periodo)`.
- Ordenação antes de `drop_duplicates`:
  1. `aluno_id, fase, periodo` (ascendente)
  2. `n_validos` (descendente — prioriza a avaliação mais completa)
  3. `arquivo_id` (ascendente — 12 primeiros caracteres do SHA-256 do conteúdo do CSV bruto)
  4. `linha_origem` (ascendente — linha física no arquivo de origem)
- Detecção e registro de conflitos: chaves duplicadas com respostas divergentes em itens válidos são quantificadas e reportadas.
- Descarte de linhas vazias: registros com `n_validos == 0` são removidos após a deduplicação.

### 4. Preservação de Escolas e Imputação Longitudinal
- `fix_escolas.py` foi corrigido para **nunca apagar linhas**.
- As 6 escolas canônicas mapeadas são:
  - `EMEB PROFESSOR RICARDO VIEIRA DE LIMA`
  - `EMEB PROFESSORA MARIA QUEIROZ FERRO`
  - `EMEB PADRE ANCHIETA`
  - `EMEF PADRE JOSÉ DOS SANTOS MOUSINHO`
  - `EMEB EXPEDITO PORFÍRIO DOS SANTOS`
  - `EMEB NATANAEL DA SILVA`
- Escolas nulas são imputadas com a moda das escolas conhecidas do próprio aluno ao longo de todas as fases/instrumentos; casos sem qualquer registro viram `NAO INFORMADA`.
- A Fase 3 pós de vocabulário preservou com exatidão todos os 1.286 alunos válidos.

### 5. Padronização de Séries e Turmas
- Séries aceitas: 6 a 9. Indicadores ordinais como 'período' sem a palavra 'ano' são desconsiderados.
- Letras da turma: extraídas em maiúsculo (A-Z).
- Turmas sem letra identificada recebem a letra canônica `U` (Turma Única), ex.: `8º ANO` -> `8U`.
- Formato resultante: `^[6-9][A-Z]$`, totalizando 23 valores distintos observados (abaixo do teto de 25).

### 6. Proteção de Dados (LGPD) e Tabela `dim_aluno`
- Os datasets de avaliação em `data/final/` não possuem dados identificadores nem atributos sensíveis.
- Os atributos demográficos foram consolidados na tabela dimensional `dim_aluno`:
  - `sexo`: "M", "F" ou "NAO INFORMADO".
  - `faixa_etaria`: `<=10`, `11-12`, `13-14`, `15+` ou `NAO INFORMADA`, calculada a partir da idade válida (8 a 20 anos) na fase mais recente.
  - `cor_raca`: maiúsculas, com "0" ou valores numéricos mapeados para "NAO INFORMADO".
  - `zona`: "URBANA", "RURAL" ou "NAO INFORMADA".
  - `necessidade_especial`: "SIM" se houver registro descritivo preenchido, senão "NAO_REGISTRADO".
  - `fonte_socio`: fases em que o aluno possui cadastro ("Fase 2", "Fase 3", "Fase 2, Fase 3").
- Resolução de conflitos demográficos entre fases: moda estatística; em caso de empate, prevalece a fase mais recente.
- A cobertura de `dim_aluno` sobre a Fase 4 atingiu **93,05%** (1.766 de 1.898 alunos cobertos), superando a meta de 90%.

### 7. Tabelas Pareadas para Análise de Impacto (Effect Size)
- `pareado_vocabulario` e `pareado_tde` foram gerados contendo apenas pares pré e pós do mesmo aluno e fase com respostas válidas.
- A validação com as referências independentes fornecidas no briefing obteve concordância exata na contagem e desvios mínimos nas médias:
  - Vocabulário Fase 2: 1.352 alunos pareados completos (Ref: ~1.352; Média Pré: 26,60 vs ~26,6; Média Pós: 27,62 vs ~27,6).
  - Vocabulário Fase 3: 872 alunos pareados completos (Ref: ~872; Média Pré: 21,45 vs ~21,4; Média Pós: 21,33 vs ~21,3).
  - TDE Fase 2: 1.245 alunos pareados completos (Ref: ~1.245; Média Pré: 37,28 vs ~37,2; Média Pós: 34,55 vs ~34,6).
  - TDE Fase 3: 1.545 alunos pareados completos (Ref: ~1.545; Média Pré: 33,96 vs ~34,1; Média Pós: 35,69 vs ~35,3).

## Consequências
- A base está inteiramente pronta e higienizada para modelagem estatística, cálculo de tamanho de efeito de Cohen, visualizações em Streamlit e integração com o agente de IA.
- Contratos de dados formais são garantidos por testes automatizados em `tests/data/test_contrato_datasets.py`.
