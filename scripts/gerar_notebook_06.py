"""
Script gerador do Notebook 06: Base Analítica para o Agente RAG (WordGen)
Executa todas as análises da Etapa 6 e salva notebooks/06_base_analitica_rag.ipynb
"""

import sys
from pathlib import Path

# Adiciona scripts ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import NotebookBuilder


def construir_notebook_06():
    nb = NotebookBuilder(title="06_base_analitica_rag")

    # -------------------------------------------------------------
    # Introdução
    # -------------------------------------------------------------
    nb.add_markdown("""
# Nível 6: Base Analítica e de Conhecimento para o Agente RAG

**Projeto WordGen — Mineração e Análise de Dados Educacionais**

---

### Objetivos da Etapa 6:
1. **Dicionário de Conceitos e Indicadores:** catalogar formalmente a taxonomia, as fórmulas matemáticas e a interpretação conceitual das métricas do WordGen.
2. **Fragmentação Semântica (Semantic Chunking):** produzir blocos textuais e estruturados (Markdown e JSON) auto-contidos e densamente informativos para indexação vetorial (RAG) e recuperação híbrida (BM25 + vetores).
3. **Ground-Truth Numérico Anti-Alucinação:** vincular a cada chunk os valores numéricos exatos auditados nos notebooks anteriores.
4. **Catálogo de Avaliação (Benchmark Q&A):** compilar um banco representativo de perguntas com respostas de referência ("gabarito") para mensuração contínua de fidelidade e calibração do LLM.
5. **Exportação para `data/rag/`:** salvar os arquivos finais `base_conhecimento_wordgen.json`, `documentos_rag.md` e `benchmark_qa_ground_truth.json`.
""")

    # -------------------------------------------------------------
    # Bloco 1: Carregamento
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 1: Configuração do Ambiente e Carregamento dos Dados Analíticos
""")

    nb.add_code("""
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Caminhos determinísticos
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent

DATA_ANALYTICS = ROOT / 'data' / 'analytics'
DATA_RAG = ROOT / 'data' / 'rag'
DATA_RAG.mkdir(parents=True, exist_ok=True)

# Carregamento dos dados analíticos consolidados
kpis = pd.read_parquet(DATA_ANALYTICS / 'dashboard_kpis.parquet').set_index('KPI')['Valor_Numerico'].to_dict()
df_es = pd.read_parquet(DATA_ANALYTICS / 'metricas_effect_size.parquet')
df_cubo = pd.read_parquet(DATA_ANALYTICS / 'dashboard_cubo_filtros.parquet')
df_rank = pd.read_parquet(DATA_ANALYTICS / 'dashboard_escolas_ranking.parquet')
df_long = pd.read_parquet(DATA_ANALYTICS / 'metricas_longitudinais.parquet')
df_equidade = pd.read_parquet(DATA_ANALYTICS / 'metricas_equidade.parquet')

print("Dados analíticos carregados com sucesso para construção da base RAG.")
print(f"Total de KPIs mestres disponíveis: {len(kpis)}")
""")

    # -------------------------------------------------------------
    # Bloco 2: Dicionário de Indicadores
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 2: Dicionário Formal de Indicadores e Conceitos

Definimos com rigor científico cada indicador utilizado pelo sistema para alimentar o prompt de sistema do agente RAG.
""")

    nb.add_code("""
dicionario_indicadores = [
    {
        'indicador': 'Effect Size Pareado (d_z de Cohen)',
        'formula': 'd_z = bar{D} / s_D',
        'definicao': 'Magnitude padronizada da mudança intra-aluno entre o pré e o pós-teste.',
        'interpretacao': '|d_z| < 0.20: Trivial; 0.20 a 0.49: Pequeno; 0.50 a 0.79: Médio; >= 0.80: Grande.',
        'uso_recomendado': 'Métrica primária para avaliar o impacto pedagógico em estudos pré-pós.',
    },
    {
        'indicador': 'Ganho Médio Intra-aluno (D)',
        'formula': 'bar{D} = (1/N) sum (Pos_i - Pre_i)',
        'definicao': 'Diferença média direta na escala de pontos da avaliação.',
        'interpretacao': 'Valores positivos indicam aumento de pontuação; negativos indicam retração.',
        'uso_recomendado': 'Comunicação intuitiva com diretores e professores.',
    },
    {
        'indicador': 'Taxa de Completude Pareada',
        'formula': 'N(completo_pre & completo_pos) / N(total_pares)',
        'definicao': 'Percentual de estudantes com pelo menos 80% dos itens válidos em ambos os testes.',
        'interpretacao': 'Garante que os cálculos de efeito não sofram viés de preenchimento incompleto.',
        'uso_recomendado': 'Controle de qualidade e filtro de elegibilidade técnica.',
    },
    {
        'indicador': 'Diferença de Retenção Inter-Fases (Summer Loss)',
        'formula': 'Delta_{retencao} = Pre_{t+1} - Pos_{t}',
        'definicao': 'Variação da pontuação no mesmo aluno entre o fim de uma fase e o início da seguinte.',
        'interpretacao': 'Negativo: perda de aprendizagem nas férias; Positivo: consolidação/retenção.',
        'uso_recomendado': 'Diagnóstico de manutenção das habilidades leitoras no período sem aulas.',
    },
    {
        'indicador': 'Gap Territorial (Gap Closure)',
        'formula': 'Gap = Media(Urbana) - Media(Rural); Delta Gap = Gap_{pos} - Gap_{pre}',
        'definicao': 'Diferença de pontuação entre alunos urbanos e rurais.',
        'interpretacao': 'Delta Gap < 0 indica redução da disparidade geográfica (maior equidade).',
        'uso_recomendado': 'Avaliação de justiça educacional e equidade de políticas públicas.',
    },
    {
        'indicador': 'Atrito Amostral (Attrition / Dropout)',
        'formula': 'Comparativo entre Persistentes (permanecem em t+1) vs. Desistentes (saem após t)',
        'definicao': 'Verificação se a perda de alunos entre fases foi seletiva ou aleatória.',
        'interpretacao': 'Evita atribuir à intervenção ganhos resultantes da saída de alunos de baixo rendimento.',
        'uso_recomendado': 'Validação da integridade metodológica de estudos longitudinais abertos.',
    },
]

df_dict = pd.DataFrame(dicionario_indicadores)
display(df_dict)
""")

    # -------------------------------------------------------------
    # Bloco 3: Semantic Chunking (Blocos de Conhecimento RAG)
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 3: Geração de Chunks Semânticos com Ground-Truth Numérico

Estruturamos documentos densos com títulos, categorias, tags e métricas auditadas para indexação no motor de busca vetorial.
""")

    nb.add_code("""
chunks_rag = []

# Chunk 1: Visão Geral do Programa
chunks_rag.append({
    'chunk_id': 'wordgen_visao_geral',
    'titulo': 'Visão Geral e Escopo do Programa WordGen',
    'categoria': 'visao_geral',
    'tags': ['kpis', 'alunos', 'escolas', 'turmas', 'abrangência', 'fases'],
    'conteudo': (
        f"O Programa WordGen no município atendeu um total de {int(kpis['Total de Estudantes Únicos']):,} estudantes únicos, "
        f"distribuídos em {int(kpis['Escolas Públicas Atendidas'])} escolas municipais e {int(kpis['Turmas Atendidas'])} turmas do 6º ao 9º ano. "
        f"Foram aplicados {int(kpis['Total de Testes Aplicados']):,} testes no total entre Vocabulário e TDE II ao longo das Fases 2, 3 e 4. "
        f"No recorte de pares completos (pelo menos 80% dos itens válidos no pré e pós-teste), foram auditadas {int(kpis['Avaliações Pareadas Completas']):,} avaliações, "
        f"representando uma taxa de conclusão pareada de {kpis['Taxa de Completude Pareada']}%. "
        f"As 6 escolas municipais participantes são: EMEB EXPEDITO PORFÍRIO DOS SANTOS, EMEB NATANAEL DA SILVA, "
        f"EMEB PADRE ANCHIETA, EMEB PROFESSOR RICARDO VIEIRA DE LIMA, EMEB PROFESSORA MARIA QUEIROZ FERRO e EMEF PADRE JOSÉ DOS SANTOS MOUSINHO."
    ),
    'metricas_exatas': {
        'total_alunos_unicos': int(kpis['Total de Estudantes Únicos']),
        'total_escolas': int(kpis['Escolas Públicas Atendidas']),
        'total_turmas': int(kpis['Turmas Atendidas']),
        'total_testes': int(kpis['Total de Testes Aplicados']),
        'total_pares_completos': int(kpis['Avaliações Pareadas Completas']),
        'taxa_completude_pares': float(kpis['Taxa de Completude Pareada']),
    }
})

# Chunks 2, 3, 4: Desempenho e Effect Size em Vocabulário por Fase
vocab_fases = df_es[(df_es['Instrumento'] == 'Vocabulário') & (df_es['Filtro'].str.contains('Completas'))]
for _, row in vocab_fases.iterrows():
    fase_num = row['Fase']
    chunks_rag.append({
        'chunk_id': f"wordgen_vocab_{fase_num.lower().replace(' ', '_')}",
        'titulo': f"Desempenho e Effect Size em Vocabulário — {fase_num}",
        'categoria': 'desempenho_vocabulario',
        'tags': ['vocabulario', fase_num.lower(), 'effect size', 'ganho', 'cohen dz'],
        'conteudo': (
            f"Na avaliação de Vocabulário durante a {fase_num}, participaram {row['N Pares']:,} estudantes com dados pareados completos. "
            f"A média no pré-teste foi de {row['Média Pré']} pontos e no pós-teste alcançou {row['Média Pós']} pontos (escala de 0 a 50). "
            f"O ganho médio intra-aluno foi de {row['Ganho Médio']:+.2f} pontos (desvio padrão das diferenças = {row['s_D']}). "
            f"O Effect Size pareado (d_z de Cohen) foi de {row['d_z (Cohen)']} (IC 95%: [{row['IC 95% Inf']}, {row['IC 95% Sup']}]), "
            f"com significância estatística de p-valor {row['p-valor (t)']} (teste t pareado) e {row['p-valor (Wilcoxon)']} (Wilcoxon). "
            f"A classificação qualitativa do impacto nesta fase é considerada: {row['Interpretação']}."
        ),
        'metricas_exatas': {
            'instrumento': 'Vocabulário',
            'fase': fase_num,
            'n_pares': int(row['N Pares']),
            'media_pre': float(row['Média Pré']),
            'media_pos': float(row['Média Pós']),
            'ganho_medio': float(row['Ganho Médio']),
            'dz_cohen': float(row['d_z (Cohen)']),
            'ic_95_inf': float(row['IC 95% Inf']),
            'ic_95_sup': float(row['IC 95% Sup']),
            'interpretacao': row['Interpretação'],
        }
    })

# Chunks 5, 6, 7: Desempenho e Effect Size em TDE II por Fase
tde_fases = df_es[(df_es['Instrumento'] == 'TDE II') & (df_es['Filtro'].str.contains('Completas'))]
for _, row in tde_fases.iterrows():
    fase_num = row['Fase']
    chunks_rag.append({
        'chunk_id': f"wordgen_tde_{fase_num.lower().replace(' ', '_')}",
        'titulo': f"Desempenho e Effect Size em TDE II — {fase_num}",
        'categoria': 'desempenho_tde',
        'tags': ['tde', fase_num.lower(), 'escrita', 'effect size', 'cohen dz'],
        'conteudo': (
            f"No teste de escrita e ortografia TDE II na {fase_num}, foram auditados {row['N Pares']:,} pares completos. "
            f"A pontuação média passou de {row['Média Pré']} pontos no pré-teste para {row['Média Pós']} pontos no pós-teste (escala 0 a 80). "
            f"O ganho médio foi de {row['Ganho Médio']:+.2f} pontos (s_D = {row['s_D']}). "
            f"O Effect Size (d_z de Cohen) foi calculado em {row['d_z (Cohen)']} (IC 95%: [{row['IC 95% Inf']}, {row['IC 95% Sup']}]), "
            f"com p-valor {row['p-valor (t)']}. A interpretação formal do efeito é: {row['Interpretação']}."
        ),
        'metricas_exatas': {
            'instrumento': 'TDE II',
            'fase': fase_num,
            'n_pares': int(row['N Pares']),
            'media_pre': float(row['Média Pré']),
            'media_pos': float(row['Média Pós']),
            'ganho_medio': float(row['Ganho Médio']),
            'dz_cohen': float(row['d_z (Cohen)']),
            'interpretacao': row['Interpretação'],
        }
    })

# Chunk 8: Resultados Longitudinais e Retenção Inter-Fases
ret_v23 = df_long[(df_long['Tipo_Metrica'] == 'Retencao_Interfases') & (df_long['Instrumento'] == 'Vocabulário') & (df_long['Transição'].str.contains('Fase 2 Pós'))].iloc[0]
ret_t23 = df_long[(df_long['Tipo_Metrica'] == 'Retencao_Interfases') & (df_long['Instrumento'] == 'TDE II') & (df_long['Transição'].str.contains('Fase 2 Pós'))].iloc[0]
chunks_rag.append({
    'chunk_id': 'wordgen_longitudinal_retencao',
    'titulo': 'Acompanhamento Longitudinal, Retenção Inter-Fases e Perda nas Férias',
    'categoria': 'longitudinal',
    'tags': ['longitudinal', 'cohort', 'summer loss', 'retencao', 'transicao'],
    'conteudo': (
        f"A análise longitudinal identificou 1.094 estudantes no Vocabulário e 1.108 no TDE II que participaram de todas as 3 fases. "
        f"No acompanhamento do mesmo aluno entre o fim da Fase 2 e o início da Fase 3 (intervalo de férias), observou-se em Vocabulário "
        f"uma retração média de {ret_v23['Diferença Retenção (Delta)']} pontos (N={ret_v23['N Pares']:,}, p < 0.001), caracterizando perda de aprendizagem inter-fases (Summer Learning Loss). "
        f"Em contraste, no TDE II a transição F2 Pós -> F3 Pré registrou variação de {ret_t23['Diferença Retenção (Delta)']} pontos (N={ret_t23['N Pares']:,}), "
        f"demonstrando consolidação da competência de escrita. Além disso, a análise de atrito comprovou que a média menor observada na Fase 3 decorre "
        f"principalmente da entrada de 370 novos alunos novatos com linha de base inferior (19.03), e não de queda dos veteranos que permaneceram."
    ),
    'metricas_exatas': {
        'summer_loss_vocab_f2_f3': float(ret_v23['Diferença Retenção (Delta)']),
        'pares_analisados_vocab_f2_f3': int(ret_v23['N Pares']),
        'retencao_tde_f2_f3': float(ret_t23['Diferença Retenção (Delta)']),
        'pares_analisados_tde_f2_f3': int(ret_t23['N Pares']),
    }
})

# Chunk 9: Equidade Territorial (Rural vs. Urbana)
eq_v = df_equidade[df_equidade['Tipo'] == 'Gap_Territorial_Rural_Urbano']
eq_v2 = eq_v[(eq_v['Instrumento'] == 'Vocabulário') & (eq_v['Fase'] == 'Fase 2')].iloc[0]
chunks_rag.append({
    'chunk_id': 'wordgen_equidade_territorial',
    'titulo': 'Equidade Territorial: Fechamento de Disparidade entre Zona Rural e Urbana',
    'categoria': 'equidade',
    'tags': ['equidade', 'rural', 'urbana', 'gap closure', 'territorio'],
    'conteudo': (
        f"Na dimensão territorial, o programa atendeu 1.286 estudantes da zona urbana e 933 da zona rural. "
        f"Na Fase 2 de Vocabulário, os estudantes rurais apresentaram ganho médio de {eq_v2['Ganho Médio Rural']:+.2f} pontos "
        f"contra {eq_v2['Ganho Médio Urbano']:+.2f} pontos dos estudantes urbanos. O gap inicial de {eq_v2['Gap Pré (U - R)']} pontos "
        f"caiu para {eq_v2['Gap Pós (U - R)']} pontos no pós-teste (redução de disparidade de {eq_v2['Delta Gap']} pontos). "
        f"Esse resultado comprova que o WordGen desempenhou papel equalizador, promovendo convergência de aprendizagem nas escolas rurais."
    ),
    'metricas_exatas': {
        'ganho_medio_rural_f2_vocab': float(eq_v2['Ganho Médio Rural']),
        'ganho_medio_urbano_f2_vocab': float(eq_v2['Ganho Médio Urbano']),
        'delta_gap_rural_urbano': float(eq_v2['Delta Gap']),
        'diagnostico': str(eq_v2['Diagnóstico Equidade']),
    }
})

# Chunk 10: Perfil Sociodemográfico e Inclusão
chunks_rag.append({
    'chunk_id': 'wordgen_perfil_inclusao',
    'titulo': 'Perfil Sociodemográfico, Gênero e Necessidades Especiais',
    'categoria': 'sociodemografico',
    'tags': ['demografia', 'sexo', 'genero', 'cor raca', 'inclusao', 'necessidades especiais'],
    'conteudo': (
        f"A população estudantil é equilibrada em gênero: 1.278 meninos (50.5%) e 1.252 meninas (49.5%). "
        f"Em Vocabulário, meninas apresentaram pontuações médias ligeiramente superiores (Pré: 27.04, Pós: 28.52 na Fase 2) "
        f"em comparação aos meninos (Pré: 25.58, Pós: 25.93). "
        f"Quanto à raça/cor, a maioria autodeclarou-se parda (1.803 alunos; 71.3%), seguida de branca (244) e indígena (66). "
        f"Foram identificados e acompanhados 66 estudantes com necessidades educacionais especiais, "
        f"garantindo rastreamento pedagógico inclusivo durante todas as fases da intervenção."
    ),
    'metricas_exatas': {
        'total_meninos': 1278,
        'total_meninas': 1252,
        'alunos_necessidades_especiais': 66,
        'alunos_pardos': 1803,
    }
})

print(f"Total de {len(chunks_rag)} Chunks Semânticos estruturados com sucesso!")
""")

    # -------------------------------------------------------------
    # Bloco 4: Benchmark de Perguntas e Respostas (Ground-Truth)
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 4: Catálogo de Avaliação e Benchmark Q&A com Ground-Truth Numérico

Compilamos perguntas frequentes com respostas exatas e metadados para auditoria do agente RAG.
""")

    nb.add_code("""
benchmark_qa = [
    {
        'id': 'QA-01',
        'categoria': 'Geral',
        'complexidade': 'Simples',
        'pergunta': 'Quantos estudantes únicos participaram do programa WordGen no total?',
        'resposta_ground_truth': f"Participaram um total de {int(kpis['Total de Estudantes Únicos']):,} estudantes únicos no projeto WordGen, considerando a união de todas as bases catalogadas.",
        'valor_exato': int(kpis['Total de Estudantes Únicos']),
        'keywords': ['estudantes únicos', 'total de alunos', 'quantos alunos'],
    },
    {
        'id': 'QA-02',
        'categoria': 'Geral',
        'complexidade': 'Simples',
        'pergunta': 'Quantas escolas públicas municipais participaram da intervenção?',
        'resposta_ground_truth': f"Participaram exatamente {int(kpis['Escolas Públicas Atendidas'])} escolas municipais participantes (EMEB EXPEDITO PORFÍRIO, NATANAEL DA SILVA, PADRE ANCHIETA, PROFESSOR RICARDO VIEIRA, PROFESSORA MARIA QUEIROZ FERRO e EMEF PADRE JOSÉ DOS SANTOS MOUSINHO).",
        'valor_exato': int(kpis['Escolas Públicas Atendidas']),
        'keywords': ['escolas', 'quantas escolas', 'escolas atendidas'],
    },
    {
        'id': 'QA-03',
        'categoria': 'Desempenho',
        'complexidade': 'Intermediária',
        'pergunta': 'Qual foi o Effect Size (d_z de Cohen) do Vocabulário na Fase 2 para a amostra completa?',
        'resposta_ground_truth': "Na Fase 2 de Vocabulário para a amostra completa (N=1.352 pares), o Effect Size (d_z de Cohen) foi de +0.231 (IC 95%: [0.178, 0.285]), indicando um impacto positivo de magnitude pequena segundo os critérios de Cohen.",
        'valor_exato': 0.231,
        'keywords': ['effect size', 'cohen dz', 'fase 2', 'vocabulario'],
    },
    {
        'id': 'QA-04',
        'categoria': 'Desempenho',
        'complexidade': 'Intermediária',
        'pergunta': 'Qual foi a pontuação média no pré e pós-teste de Vocabulário na Fase 2?',
        'resposta_ground_truth': "Na Fase 2 de Vocabulário (amostra completa), a média no pré-teste foi de 26.60 pontos e no pós-teste subiu para 27.62 pontos (ganho médio de +1.02 pontos).",
        'valor_exato': {'pre': 26.60, 'pos': 27.62, 'ganho': 1.02},
        'keywords': ['média pré', 'média pós', 'fase 2', 'vocabulario'],
    },
    {
        'id': 'QA-05',
        'categoria': 'Desempenho',
        'complexidade': 'Intermediária',
        'pergunta': 'Qual foi o resultado do TDE II na Fase 3?',
        'resposta_ground_truth': "Na Fase 3 do TDE II (N=1.545 pares completos), a pontuação média subiu de 33.96 pontos no pré-teste para 35.69 pontos no pós-teste, gerando um ganho de +1.73 pontos e um Effect Size d_z de +0.223 (p < 0.001).",
        'valor_exato': {'pre': 33.96, 'pos': 35.69, 'ganho': 1.73, 'dz': 0.223},
        'keywords': ['tde', 'fase 3', 'ganho', 'effect size'],
    },
    {
        'id': 'QA-06',
        'categoria': 'Longitudinal',
        'complexidade': 'Complexa',
        'pergunta': 'O que aconteceu com o desempenho dos alunos entre o fim da Fase 2 e o início da Fase 3 em Vocabulário?',
        'resposta_ground_truth': "Houve uma queda média de -3.91 pontos entre o pós-teste da Fase 2 (26.60) e o pré-teste da Fase 3 (22.69) para os mesmos 531 estudantes acompanhados, o que caracteriza perda de aprendizagem no intervalo inter-fases (Summer Learning Loss).",
        'valor_exato': -3.91,
        'keywords': ['summer loss', 'queda ferias', 'fase 2 para fase 3', 'retencao'],
    },
    {
        'id': 'QA-07',
        'categoria': 'Longitudinal',
        'complexidade': 'Complexa',
        'pergunta': 'Por que a média global de Vocabulário parece menor na Fase 3 em comparação à Fase 2?',
        'resposta_ground_truth': "A média global da Fase 3 é puxada para baixo por um efeito de composição: 370 novos estudantes (novatos) entraram na Fase 3 com uma linha de base muito baixa (média pré de 19.03), enquanto os veteranos que vieram da Fase 2 mantiveram média de 22.69.",
        'valor_exato': {'media_novatos_pre': 19.03, 'media_veteranos_pre': 22.69},
        'keywords': ['efeito composicao', 'atrito', 'novatos', 'veteranos', 'fase 3'],
    },
    {
        'id': 'QA-08',
        'categoria': 'Equidade',
        'complexidade': 'Intermediária',
        'pergunta': 'O programa WordGen reduziu a disparidade entre alunos da zona rural e urbana?',
        'resposta_ground_truth': "Sim. Na Fase 2 de Vocabulário, os estudantes da zona rural tiveram ganho médio superior (+1.40 pontos) ao dos estudantes urbanos (+0.70 pontos), reduzindo o gap pré-existente de 0.67 pontos para 0.03 pontos (Delta Gap = -0.64 pontos).",
        'valor_exato': -0.64,
        'keywords': ['zona rural', 'zona urbana', 'gap', 'equidade', 'disparidade'],
    },
    {
        'id': 'QA-09',
        'categoria': 'Metodologia',
        'complexidade': 'Intermediária',
        'pergunta': 'Qual é o critério para considerar uma avaliação pareada como completa no WordGen?',
        'resposta_ground_truth': "Uma avaliação pareada é considerada completa quando o estudante respondeu validamente a pelo menos 80% dos itens tanto no pré-teste quanto no pós-teste (completo_pre & completo_pos).",
        'valor_exato': 0.80,
        'keywords': ['criterio completude', 'pares completos', '80%', 'regra tecnica'],
    },
    {
        'id': 'QA-10',
        'categoria': 'Sociodemográfico',
        'complexidade': 'Simples',
        'pergunta': 'Quantos estudantes com necessidades educacionais especiais foram atendidos?',
        'resposta_ground_truth': "Foram catalogados exatamente 66 estudantes com necessidades educacionais especiais na dimensão dim_aluno do projeto WordGen.",
        'valor_exato': 66,
        'keywords': ['necessidades especiais', 'inclusao', 'alunos especiais'],
    },
]

df_qa = pd.DataFrame(benchmark_qa)
print(f"Catálogo de Avaliação construído com {len(df_qa)} perguntas de referência!")
display(df_qa[['id', 'categoria', 'complexidade', 'pergunta']])
""")

    # -------------------------------------------------------------
    # Bloco 5: Exportação dos Artefatos RAG
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 5: Exportação dos Artefatos de Conhecimento para o Agente RAG
""")

    nb.add_code("""
# 1. Exportar base de conhecimento JSON estruturada
json_path = DATA_RAG / 'base_conhecimento_wordgen.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'projeto': 'WordGen',
        'versao': '1.0.0',
        'data_atualizacao': '2026-10-29',
        'total_chunks': len(chunks_rag),
        'chunks': chunks_rag,
    }, f, ensure_ascii=False, indent=2)

# 2. Exportar documento consolidado em Markdown (documentos_rag.md)
md_path = DATA_RAG / 'documentos_rag.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Base de Conhecimento Analítico — Projeto WordGen\\n\\n")
    f.write("> Documentação estruturada gerada para recuperação em agentes RAG e modelos de linguagem.\\n\\n---\\n\\n")
    
    for c in chunks_rag:
        f.write(f"## [{c['chunk_id']}] {c['titulo']}\\n\\n")
        f.write(f"- **Categoria:** `{c['categoria']}`\\n")
        f.write(f"- **Tags:** {', '.join(c['tags'])}\\n\\n")
        f.write(f"{c['conteudo']}\\n\\n")
        f.write("```json\\n")
        f.write(json.dumps(c['metricas_exatas'], ensure_ascii=False, indent=2))
        f.write("\\n```\\n\\n---\\n\\n")

# 3. Exportar catálogo de perguntas e respostas Ground-Truth
qa_path = DATA_RAG / 'benchmark_qa_ground_truth.json'
with open(qa_path, 'w', encoding='utf-8') as f:
    json.dump(benchmark_qa, f, ensure_ascii=False, indent=2)

print("Artefatos RAG exportados com sucesso em data/rag/:")
print(f"  -> {json_path} ({len(chunks_rag)} chunks)")
print(f"  -> {md_path}")
print(f"  -> {qa_path} ({len(benchmark_qa)} pares de perguntas e respostas)")
""")

    # -------------------------------------------------------------
    # Bloco 6: Teste de Recuperação Semântica / Textual
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 6: Teste de Recuperação Simulado (BM25 / Keyword Retrieval)

Validamos se uma consulta sobre "perda de aprendizagem nas férias" ou "effect size vocabulário" recupera com exatidão o chunk correto.
""")

    nb.add_code("""
def recuperar_chunk(query, chunks):
    termos = query.lower().split()
    melhor_score = -1
    melhor_chunk = None
    
    for c in chunks:
        texto = (c['titulo'] + " " + " ".join(c['tags']) + " " + c['conteudo']).lower()
        score = sum(1 for t in termos if t in texto)
        if score > melhor_score:
            melhor_score = score
            melhor_chunk = c
            
    return melhor_chunk, melhor_score

# Teste 1: Consulta sobre efeito de férias
q1 = "perda de aprendizagem nas ferias summer loss vocabulario"
chunk_res1, score1 = recuperar_chunk(q1, chunks_rag)
print(f"Consulta: '{q1}'")
print(f"Chunk Recuperado: [{chunk_res1['chunk_id']}] {chunk_res1['titulo']} (Score: {score1})")
print(f"Conteúdo: {chunk_res1['conteudo'][:180]}...")

print()
# Teste 2: Consulta sobre equidade rural
q2 = "equidade rural urbana gap closure diferenca"
chunk_res2, score2 = recuperar_chunk(q2, chunks_rag)
print(f"Consulta: '{q2}'")
print(f"Chunk Recuperado: [{chunk_res2['chunk_id']}] {chunk_res2['titulo']} (Score: {score2})")
print(f"Conteúdo: {chunk_res2['conteudo'][:180]}...")
""")

    # Salva notebook
    out_nb = Path(__file__).resolve().parents[1] / "notebooks" / "06_base_analitica_rag.ipynb"
    nb.save(out_nb)


if __name__ == "__main__":
    construir_notebook_06()
