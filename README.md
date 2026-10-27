# WordGen — Mineração e Engenharia de Dados

Pipeline de Engenharia e Mineração de Dados do Projeto WordGen, desenvolvido para a disciplina de Inteligência Artificial da Universidade Federal do Pará (UFPA).

O objetivo deste pipeline é consolidar, higienizar, deduplicar determinísticamente e anonimizar as avaliações educacionais psicométricas de **Vocabulário** e **TDE II (Subteste de Escrita)**, juntamente com metadados socioeconômicos dos estudantes (Fases 2, 3 e 4).

---

## Estrutura dos Datasets Finais (`data/final/`)

Os arquivos finais estão disponíveis nos formatos CSV (codificação UTF-8 com BOM) e Apache Parquet:

| Arquivo | Descrição | Granularidade |
|---|---|---|
| `dataset_vocabulario` | Respostas aos 50 itens de vocabulário (Q01 a Q50) e escores totais higienizados. | 1 registro por aluno, fase e período |
| `dataset_tde` | Respostas aos 40 itens de escrita do TDE II (P01 a P40) e escores totais higienizados. | 1 registro por aluno, fase e período |
| `dim_aluno` | Dimensão demográfica minimizada (sexo, faixa etária, cor/raça, zona, necessidade especial, fases). | 1 registro por aluno único |
| `pareado_vocabulario` | Avaliações pré e pós pareadas do mesmo aluno e fase para análise de impacto e *effect size*. | 1 registro por aluno e fase |
| `pareado_tde` | Avaliações pré e pós pareadas do mesmo aluno e fase para análise de impacto e *effect size*. | 1 registro por aluno e fase |
| `relatorio_qualidade.md` | Relatório executivo completo de métricas estatísticas agregadas e validações. | Documento Markdown |

> [!NOTE]
> Os datasets de avaliação não contêm dados pessoais identificáveis (LGPD/privacidade). Todos os atributos demográficos minimizados estão concentrados em `dim_aluno`.

---

## Como Reproduzir

### Pré-requisitos
- **Sistema Operacional:** Windows 10/11 ou Linux
- **Python:** 3.12 ou superior
- **Gerenciador de Pacotes:** [`uv`](https://github.com/astral-sh/uv)

### 1. Instalação do Ambiente
Clone o repositório e sincronize as dependências com o `uv`:

```bash
git checkout feat/correcoes-pipeline
uv sync
```

### 2. Execução Completa do Pipeline
Para processar os dados brutos e gerar todas as tabelas em `data/final/`:

```bash
uv run python pipeline/run_all.py
```

O orquestrador executará sequencialmente:
1. `pipeline/pipeline_datasets.py`: Extração dos 209 CSVs, classificação de células psicométricas, rastreabilidade via SHA-256 (`arquivo_id`) e deduplicação determinística.
2. `pipeline/fix_escolas.py`: Padronização das escolas canônicas e imputação longitudinal pela moda histórica de cada estudante.
3. `pipeline/finalizar_datasets.py`: Remoção de PII, consolidação de `dim_aluno`, montagem das tabelas pareadas e geração automática do relatório de qualidade.

### 3. Execução dos Testes Automatizados
Para rodar a suíte completa de testes (unidade e contratos de dados):

```bash
uv run pytest
```

Para verificar conformidade de estilo e formatação com o Ruff:

```bash
uv run ruff check .
uv run ruff format --check .
```
