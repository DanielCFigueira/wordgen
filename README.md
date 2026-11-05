# WordGen — Mineração e Engenharia de Dados

Pipeline de Engenharia e Mineração de Dados do Projeto WordGen, desenvolvido para a disciplina de Inteligência Artificial da Universidade Federal do Pará (UFPA).

O objetivo deste pipeline é consolidar, higienizar, deduplicar determinísticamente e **pseudonimizar/minimizar** as avaliações educacionais psicométricas de **Vocabulário** e **TDE II (Subteste de Escrita)**, juntamente com metadados demográficos e contextuais dos estudantes (Fases 2, 3 e 4).

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
> Os datasets finais não contêm nomes, matrículas, contatos, datas de nascimento ou responsáveis. Contudo, `aluno_id` é um identificador pseudonimizado e, combinado a atributos de contexto, continua sujeito a controles de acesso e à LGPD; não é uma garantia de anonimização absoluta.

---

## Como Reproduzir

### Dashboard Analítico

Com o ambiente sincronizado (`uv sync`), inicie o painel com:

```bash
uv run streamlit run src/wordgen/dashboard/app.py
```

O dashboard lê os arquivos consolidados existentes em `data/final/`, os resultados de modelagem em `reports/modelagem/metricas/` e os metadados em `models/modelagem/`. Ele não reexecuta o pipeline nem o treinamento.

Quando os cinco CSVs finais obrigatórios não estão disponíveis — como na publicação pública no
Streamlit Community Cloud — o painel inicia automaticamente em **modo de demonstração** com dados
inteiramente sintéticos. Esse fallback é determinístico, não lê nem deriva qualquer registro local
e mostra um aviso visível. Artefatos analíticos, de modelagem e RAG ausentes continuam indicados
como indisponíveis; o modo de demonstração não inventa resultados desses artefatos.

Após a inicialização, acesse `http://localhost:8501` no navegador. O painel reúne: visão geral, Vocabulário, TDE II · Escrita, contexto dos estudantes, comparação pré/pós (Effect Size), acompanhamento longitudinal, qualidade dos dados e modelagem. Os filtros atuam sobre avaliações; a seção de ganho sempre usa pares de pré e pós, e os resultados persistidos de modelagem só respondem ao filtro de instrumento.

Os CSVs `dataset_vocabulario.csv`, `dataset_tde.csv`, `pareado_vocabulario.csv`, `pareado_tde.csv` e `dim_aluno.csv` são obrigatórios. Os artefatos analíticos em `data/analytics/` — KPIs, Effect Size e longitudinal — são lidos como resultados pré-calculados pelos notebooks; o dashboard não os recalcula. O relatório de qualidade, as métricas comparativas e os metadados de modelos são complementares: quando estiverem ausentes, a área correspondente informa a indisponibilidade sem executar pipeline ou treinamento. Para reconstruir os datasets finais ausentes, use apenas o orquestrador documentado em [Execução Completa do Pipeline](#2-execução-completa-do-pipeline).

#### Privacidade e limites analíticos

O dashboard não exibe `aluno_id` nem registros individuais. Recortes e decomposições somente são mostrados com, no mínimo, 10 estudantes distintos por grupo. Quando uma categoria não alcança esse limite, toda a decomposição é suprimida para impedir inferência por diferença entre total e categorias visíveis. Essa regra é uma proteção inicial de divulgação, não uma garantia absoluta de anonimização.

Os comparativos demográficos são descritivos, não causais. As pontuações de Vocabulário e TDE II permanecem em escalas separadas. Os resultados de modelagem são experimentais, não devem orientar decisões pedagógicas individuais e dependem de validação adicional; no Experimento B, a disponibilidade temporal de atributos contextuais ainda não foi comprovada.

#### Erros comuns

- **Bases finais não encontradas:** execute `uv run python pipeline/run_all.py` somente quando precisar gerar os artefatos ausentes.
- **Erro de estrutura ou escala:** não altere os CSVs no dashboard; revise a etapa de geração que produziu a base final.
- **Página de modelagem sem resultados:** confirme a presença de `reports/modelagem/metricas/metricas_comparativas_gerais.csv` e dos JSONs em `models/modelagem/`.

Arquivos `.env` são locais, estão ignorados pelo Git e não devem ser incluídos em ZIPs, imagens ou outras distribuições. Use apenas valores vazios ou fictícios em `.env.example`.

### Assistente RAG + LLM

O assistente é integrado como a página **Assistente analítico** no dashboard principal. Ele usa a base curada em `data/rag/base_conhecimento_wordgen.json` para responder perguntas conceituais e metodológicas, e consulta diretamente os CSVs agregados publicados para perguntas quantitativas. Assim, métricas como MAE, *effect size*, KPIs e acompanhamento longitudinal não são geradas pelo modelo de linguagem.

O índice persistente local (`data/rag/indice_tfidf_wordgen.joblib`) é criado automaticamente a partir dos chunks já produzidos pelo notebook `notebooks/analises/06_base_analitica_rag.ipynb`. Ele usa vetorização TF-IDF, sem downloads adicionais. A geração por LLM é opcional: sem `GROQ_API_KEY`, o assistente continua respondendo por extração fundamentada e informa essa limitação.

Configure a chave somente no ambiente local (nunca a registre em código, documentação ou arquivos distribuídos):

```bash
# Windows PowerShell, apenas para a sessão atual
$env:GROQ_API_KEY = "sua_chave_local"
# Opcional: $env:GROQ_MODEL = "openai/gpt-oss-20b"
```

Inicie o dashboard e acesse a página **Assistente analítico** na mesma navegação:

```bash
uv run streamlit run src/wordgen/dashboard/app.py
```

O dashboard e o assistente estarão disponíveis em `http://localhost:8501`. O assistente mantém o histórico apenas durante a sessão, exibe as fontes usadas e bloqueia pedidos por identificadores ou registros individuais. Ele também recusa formatos de identificador pseudonimizado, e-mail, CPF e telefone antes de qualquer chamada ao Groq. Perguntas fora do escopo do WordGen são recusadas antes da recuperação; se não houver evidência suficiente, o agente o informa sem responder com o panorama geral. Não é uma ferramenta validada para decisão pedagógica; os resultados de modelagem e atributos contextuais conservam as limitações metodológicas documentadas no projeto.

Para executar somente a interface do agente em paralelo ao dashboard, use uma porta distinta:

```bash
uv run streamlit run src/wordgen/agente/app.py --server.port 8502
```

Para verificar a cobertura técnica do benchmark local, sem chamadas à API Groq:

```bash
uv run python scripts/avaliar_agente.py
uv run pytest tests/test_agente.py
```

O benchmark mede se as perguntas de referência recebem resposta e fonte identificada. A fidelidade semântica das sínteses geradas pelo LLM requer revisão humana, pois depende do provedor externo.

### Pré-requisitos
- **Sistema Operacional:** Windows 10/11 ou Linux
- **Python:** 3.12 ou superior
- **Gerenciador de Pacotes:** [`uv`](https://github.com/astral-sh/uv)

### 1. Instalação do Ambiente
Clone o repositório e sincronize as dependências com o `uv`:

```bash
git clone https://github.com/DanielCFigueira/wordgen.git
cd wordgen
uv sync --locked
```

> [!IMPORTANT]
> As bases educacionais, artefatos de modelos e arquivos `.env` são locais e não fazem parte do
> repositório. Antes de executar o dashboard, obtenha-os por canal seguro e coloque-os nas pastas
> previstas (`data/`, `models/` e `reports/`). Nunca os adicione ao Git. Para gerar dados locais a
> partir das fontes autorizadas, use o pipeline descrito abaixo.

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

### Colaboração e preparação para deploy

- O repositório deve permanecer privado enquanto contiver documentação operacional do projeto.
- `.env`, `data/`, `models/`, arquivos tabulares e caches são ignorados pelo Git; mantenha apenas
  `.env.example` sem valores reais.
- Antes de enviar alterações, execute `uv run pytest`, `uv run ruff check .` e
  `git diff --check`.
- O hook local bloqueia a inclusão de arquivos tabulares e compactados para reduzir o risco de
  publicar dados educacionais. O hook também remove saídas de notebooks antes do commit, sem
  depender de caminhos absolutos da máquina. Instale os hooks com
  `uv run pre-commit install` após `uv sync`.

## Publicação em produção

O repositório e a imagem Docker **não incluem dados educacionais, modelos, relatórios de métricas,
índices RAG ou credenciais**. O serviço deve executar como usuário não privilegiado e receber
somente os artefatos aprovados por volume privado ou armazenamento autenticado. Não publique uma
imagem com `data/raw/`, `data/processed/`, `data/legado/` ou qualquer arquivo com registros por
estudante.

### Artefatos privados necessários

Monte os diretórios abaixo no servidor. Os cinco CSVs de `final/` são necessários para as páginas
analíticas; os demais habilitam páginas complementares e o assistente.

```text
<dados>/final/dataset_vocabulario.csv
<dados>/final/dataset_tde.csv
<dados>/final/pareado_vocabulario.csv
<dados>/final/pareado_tde.csv
<dados>/final/dim_aluno.csv
<dados>/analytics/dashboard_kpis.csv
<dados>/analytics/metricas_effect_size.csv
<dados>/analytics/metricas_longitudinais.csv
<dados>/analytics/painel_longitudinal.csv
<dados>/rag/base_conhecimento_wordgen.json
<relatorios>/modelagem/metricas/metricas_comparativas_gerais.csv
<modelos>/modelagem/*_metadata.json
```

`relatorio_qualidade.md`, os CSVs analíticos adicionais e os metadados de modelos são
complementares: a página correspondente informa a indisponibilidade se não existirem. Os binários
de modelos `.joblib` não são carregados pela aplicação web e não devem ser montados sem uma
necessidade operacional específica. O índice TF-IDF é reconstruível; mantenha um diretório
gravável separado para o cache.

### Variáveis de ambiente

| Variável | Obrigatória | Finalidade |
|---|---:|---|
| `WORDGEN_DATA_DIR` | Sim no deploy | Diretório privado com `final/`, `analytics/` e `rag/`. |
| `WORDGEN_REPORTS_DIR` | Para página de modelagem | Diretório privado com métricas agregadas. |
| `WORDGEN_MODEL_DIR` | Não | Metadados JSON de modelos. |
| `WORDGEN_RUNTIME_DIR` | Sim no deploy | Diretório gravável apenas para o índice TF-IDF reconstruível. |
| `GROQ_API_KEY` | Não | Secret opcional para síntese por LLM. Sem ele, o assistente usa somente recuperação local. |
| `GROQ_MODEL` | Não | Modelo Groq permitido pela conta; o padrão está em `.env.example`. |

### Container de produção

O projeto inclui um `Dockerfile` para uma plataforma de containers autenticada. A opção mais
adequada é um serviço privado com autenticação na borda — por exemplo, Cloud Run, ECS/Fargate ou
equivalente — com os diretórios privados fornecidos por armazenamento restrito e `GROQ_API_KEY`
injetada pelo gerenciador de secrets da plataforma. Não use hospedagem pública sem autenticação
para dados educacionais pseudonimizados.

```bash
docker build -t wordgen:prod .
docker run --rm -p 8501:8501 \
  -e WORDGEN_DATA_DIR=/run/wordgen/data \
  -e WORDGEN_REPORTS_DIR=/run/wordgen/reports \
  -e WORDGEN_MODEL_DIR=/run/wordgen/models/modelagem \
  -e WORDGEN_RUNTIME_DIR=/var/lib/wordgen \
  -v /caminho-privado/dados:/run/wordgen/data:ro \
  -v /caminho-privado/relatorios:/run/wordgen/reports:ro \
  -v /caminho-privado/modelos:/run/wordgen/models:ro \
  -v wordgen-runtime:/var/lib/wordgen \
  wordgen:prod
```

Em PowerShell, use crase (`` ` ``) no lugar de `\` para quebrar linhas. Configure a chave Groq
somente no painel de secrets da plataforma; nunca em argumentos de comando, imagem ou arquivo
versionado.
