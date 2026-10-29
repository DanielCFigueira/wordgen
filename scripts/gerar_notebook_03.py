"""
Script gerador do Notebook 03: Análise Longitudinal (WordGen)
Executa todas as análises da Etapa 3 e salva notebooks/03_analise_longitudinal.ipynb
"""

import sys
from pathlib import Path

# Adiciona scripts ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import NotebookBuilder


def construir_notebook_03():
    nb = NotebookBuilder(title="03_analise_longitudinal")

    # -------------------------------------------------------------
    # Introdução
    # -------------------------------------------------------------
    nb.add_markdown("""
# Nível 3: Análise Longitudinal e Acompanhamento de Cohorts

**Projeto WordGen — Mineração e Análise de Dados Educacionais**

---

### Objetivos da Etapa 3:
1. **Recorrência de alunos:** quantificar a participação estudantil ao longo das Fases 2, 3 e 4 (alunos que participaram de 1, 2 ou 3 fases).
2. **Sobreposição e Fluxo:** mapear a interseção e a permanência de estudantes entre as fases da intervenção pedagógica.
3. **Cohorts longitudinais:** acompanhar painéis de estudantes ao longo do tempo (F2 $\\to$ F3, F3 $\\to$ F4, F2 $\\to$ F4 e o cohort contínuo F2 $\\to$ F3 $\\to$ F4).
4. **Evolução dos escores:** comparar as trajetórias educacionais dos estudantes acompanhados continuamente versus os participantes de fases isoladas.
5. **Variação inter-fases e retenção de aprendizagem:** quantificar a diferença de pontuação observada entre o término de uma fase e o início da fase seguinte ($Pré_{t+1} - Pós_{t}$) no retorno letivo, avaliando hipóteses de recesso escolar e calibração de instrumentos.
6. **Efeito Intra-aluno vs. Efeito de Composição:** avaliar se as variações de médias entre fases derivam de evolução intra-aluno ou de mudanças na composição da amostra (alunos novatos vs. veteranos e análise de atrito amostral).
7. **Exportação analítica:** gerar as tabelas `painel_longitudinal.parquet` e `metricas_longitudinais.parquet` em `data/analytics/`.
""")

    # -------------------------------------------------------------
    # Bloco 1: Configuração e Carregamento
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 1: Configuração do Ambiente e Carregamento dos Dados
""")

    nb.add_code("""
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Caminhos determinísticos
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent

DATA_FINAL = ROOT / 'data' / 'final'
DATA_ANALYTICS = ROOT / 'data' / 'analytics'
DATA_ANALYTICS.mkdir(parents=True, exist_ok=True)

# Carregamento de dados finais e pareados
df_v = pd.read_parquet(DATA_FINAL / 'dataset_vocabulario.parquet')
df_t = pd.read_parquet(DATA_FINAL / 'dataset_tde.parquet')
df_pv = pd.read_parquet(DATA_FINAL / 'pareado_vocabulario.parquet')
df_pt = pd.read_parquet(DATA_FINAL / 'pareado_tde.parquet')
df_aluno = pd.read_parquet(DATA_FINAL / 'dim_aluno.parquet')

print(f"Base Vocabulário: {len(df_v):,} registros ({df_v['aluno_id'].nunique():,} alunos únicos)")
print(f"Base TDE II:      {len(df_t):,} registros ({df_t['aluno_id'].nunique():,} alunos únicos)")
print(f"Dimensão Alunos:  {len(df_aluno):,} alunos únicos catalogados")
""")

    # -------------------------------------------------------------
    # Bloco 2: Recorrência e Mapeamento de Presença
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 2: Recorrência de Alunos entre Fases (Fase 2, Fase 3 e Fase 4)

Investigamos quantos alunos participaram de 1 única fase, de 2 fases ou de todas as 3 fases do projeto WordGen, distinguindo presença censitária de pares completos.
""")

    nb.add_code("""
def mapear_recorrencia(df_raw, df_pareado, nome_instrumento):
    # Presença geral (qualquer teste realizado)
    fases_geral = df_raw.groupby('aluno_id')['fase'].apply(lambda s: tuple(sorted(s.unique())))
    
    # Presença pareada completa (pre e pos realizados com >=80% itens)
    pareado_comp = df_pareado[df_pareado['completo_pre'] & df_pareado['completo_pos']]
    fases_comp = pareado_comp.groupby('aluno_id')['fase'].apply(lambda s: tuple(sorted(s.unique())))
    
    resumo_geral = fases_geral.map(lambda f: f"{len(f)} Fase(s)").value_counts().sort_index()
    resumo_comp = fases_comp.map(lambda f: f"{len(f)} Fase(s)").value_counts().sort_index()
    
    tabela = pd.DataFrame({
        'N Alunos (Presença Geral)': resumo_geral,
        '% Geral': (resumo_geral / resumo_geral.sum() * 100).round(1),
        'N Alunos (Pares Completos)': resumo_comp,
        '% Completos': (resumo_comp / resumo_comp.sum() * 100).round(1),
    }).fillna(0)
    
    return tabela, fases_geral, fases_comp

tab_rec_v, padroes_v_geral, padroes_v_comp = mapear_recorrencia(df_v, df_pv, 'Vocabulário')
tab_rec_t, padroes_t_geral, padroes_t_comp = mapear_recorrencia(df_t, df_pt, 'TDE II')

print("=== RECORRÊNCIA DE PARTICIPAÇÃO: VOCABULÁRIO ===")
display(tab_rec_v)

print()
print("=== RECORRÊNCIA DE PARTICIPAÇÃO: TDE II ===")
display(tab_rec_t)
""")

    nb.add_markdown("""
### Detalhamento das Combinações de Presença (Fase 2, 3 e 4)
""")

    nb.add_code("""
def detalhar_combinacoes(padroes, nome_instrumento):
    df_padroes = padroes.value_counts().reset_index()
    df_padroes.columns = ['Fases Presente', 'Total Alunos']
    df_padroes['Fases Presente'] = df_padroes['Fases Presente'].apply(lambda x: " + ".join([f"Fase {f}" for f in x]))
    df_padroes['% do Total'] = (df_padroes['Total Alunos'] / df_padroes['Total Alunos'].sum() * 100).round(1)
    df_padroes['Instrumento'] = nome_instrumento
    return df_padroes[['Instrumento', 'Fases Presente', 'Total Alunos', '% do Total']]

comb_v = detalhar_combinacoes(padroes_v_geral, 'Vocabulário')
comb_t = detalhar_combinacoes(padroes_t_geral, 'TDE II')

tab_combinacoes = pd.concat([comb_v, comb_t], ignore_index=True)
display(tab_combinacoes)
""")

    # -------------------------------------------------------------
    # Bloco 3: Visualização da Sobreposição
    # -------------------------------------------------------------
    nb.add_markdown("""
### Visualização: Sobreposição e Tamanho das Interseções entre Fases
""")

    nb.add_code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, (nome, df_comb, cor) in enumerate([('Vocabulário', comb_v, 'teal'), ('TDE II', comb_t, 'darkslateblue')]):
    ax = axes[i]
    df_plot = df_comb.sort_values(by='Total Alunos', ascending=True)
    bars = ax.barh(df_plot['Fases Presente'], df_plot['Total Alunos'], color=cor, alpha=0.85, edgecolor='black')
    
    for bar in bars:
        w = bar.get_width()
        pct = 100 * w / df_comb['Total Alunos'].sum()
        ax.annotate(f"{int(w):,} ({pct:.1f}%)",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_title(f"Padrões de Presença Longitudinal — {nome}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Número de Estudantes Únicos")
    ax.set_xlim(0, max(df_plot['Total Alunos']) * 1.25)
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Bloco 4: Cohort Contínuo (Trajetória nas 3 Fases)
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 3: Acompanhamento de Cohorts Longitudinais

Acompanhamos painéis de estudantes ao longo das transições:
1. **Cohort F2 $\\to$ F3:** avaliados na Fase 2 e na Fase 3.
2. **Cohort F3 $\\to$ F4:** avaliados na Fase 3 e na Fase 4.
3. **Cohort F2 $\\to$ F4:** avaliados na Fase 2 e na Fase 4.
4. **Cohort Contínuo (F2 $\\to$ F3 $\\to$ F4):** núcleo acompanhado nas três etapas consecutivas.

*Nota metodológica:* a chave de agregação do painel é estritamente `aluno_id`, permitindo rastrear o estudante mesmo que tenha havido migração entre escolas da rede municipal durante o período.
""")

    nb.add_code("""
def montar_painel_longitudinal_aluno(df_pareado, nome_instrumento):
    piv = df_pareado.pivot_table(
        index='aluno_id',
        columns='fase',
        values=['total_pre', 'total_pos', 'completo_pre', 'completo_pos', 'escola', 'serie'],
        aggfunc='first'
    )
    
    # Achatando colunas de forma determinística
    piv.columns = [f"{col}_{fase}" for col, fase in piv.columns]
    piv = piv.reset_index()
    piv['instrumento'] = nome_instrumento
    
    # Escola mais recente documentada para o aluno
    piv['escola_recente'] = piv['escola_4'].fillna(piv['escola_3']).fillna(piv['escola_2'])
    
    # Indicadores de presença em cohorts
    piv['in_f2'] = piv['total_pre_2'].notna() | piv['total_pos_2'].notna()
    piv['in_f3'] = piv['total_pre_3'].notna() | piv['total_pos_3'].notna()
    piv['in_f4'] = piv['total_pre_4'].notna() | piv['total_pos_4'].notna()
    
    piv['cohort_f2_f3'] = piv['in_f2'] & piv['in_f3']
    piv['cohort_f3_f4'] = piv['in_f3'] & piv['in_f4']
    piv['cohort_f2_f4'] = piv['in_f2'] & piv['in_f4']
    piv['cohort_continuo_3f'] = piv['in_f2'] & piv['in_f3'] & piv['in_f4']
    
    piv['n_fases'] = piv['in_f2'].astype(int) + piv['in_f3'].astype(int) + piv['in_f4'].astype(int)
    
    return piv

painel_v = montar_painel_longitudinal_aluno(df_pv, 'Vocabulário')
painel_t = montar_painel_longitudinal_aluno(df_pt, 'TDE II')

print(f"Painel Longitudinal Vocabulário: {len(painel_v):,} alunos únicos")
print(f"  -> No Cohort F2->F3:           {painel_v['cohort_f2_f3'].sum():,} alunos")
print(f"  -> No Cohort F3->F4:           {painel_v['cohort_f3_f4'].sum():,} alunos")
print(f"  -> No Cohort Contínuo (3 fases): {painel_v['cohort_continuo_3f'].sum():,} alunos")
print()
print(f"Painel Longitudinal TDE II:      {len(painel_t):,} alunos únicos")
print(f"  -> No Cohort F2->F3:           {painel_t['cohort_f2_f3'].sum():,} alunos")
print(f"  -> No Cohort F3->F4:           {painel_t['cohort_f3_f4'].sum():,} alunos")
print(f"  -> No Cohort Contínuo (3 fases): {painel_t['cohort_continuo_3f'].sum():,} alunos")
""")

    nb.add_markdown("""
### Trajetória Média dos Estudantes do Cohort Contínuo (6 Momentos de Avaliação)
""")

    nb.add_code("""
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

momentos = ['F2 Pré', 'F2 Pós', 'F3 Pré', 'F3 Pós', 'F4 Pré', 'F4 Pós']
colunas_v = ['total_pre_2', 'total_pos_2', 'total_pre_3', 'total_pos_3', 'total_pre_4', 'total_pos_4']

for i, (nome, painel, max_score, cor) in enumerate([('Vocabulário', painel_v, 50, 'navy'), ('TDE II', painel_t, 80, 'darkgreen')]):
    ax = axes[i]
    sub = painel[painel['cohort_continuo_3f']]
    
    medias = [sub[col].mean() for col in colunas_v]
    erros_se = [sub[col].std() / np.sqrt(sub[col].notna().sum()) for col in colunas_v]
    
    ax.errorbar(momentos, medias, yerr=erros_se, fmt='-o', color=cor, linewidth=2.5,
                 capsize=5, markersize=8, label=f'Cohort Contínuo (N={len(sub):,})')
    
    # Anotações de pontuação
    for m, val in zip(momentos, medias):
        ax.annotate(f"{val:.2f}", xy=(m, val), xytext=(0, 10),
                    textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
        
    # Linhas divisórias entre as fases
    ax.axvline(1.5, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(3.5, color='gray', linestyle=':', alpha=0.7)
    
    ax.text(0.5, 0.05, 'Fase 2', transform=ax.get_xaxis_transform(), ha='center', fontsize=11, fontweight='bold', color='gray')
    ax.text(2.5, 0.05, 'Fase 3', transform=ax.get_xaxis_transform(), ha='center', fontsize=11, fontweight='bold', color='gray')
    ax.text(4.5, 0.05, 'Fase 4', transform=ax.get_xaxis_transform(), ha='center', fontsize=11, fontweight='bold', color='gray')
    
    ax.set_title(f"Trajetória Longitudinal no Cohort Contínuo — {nome}", fontsize=13, fontweight='bold')
    ax.set_ylabel(f"Pontuação Média (Escala 0-{max_score})")
    ax.set_ylim(min(medias) * 0.85, max(medias) * 1.15)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Bloco 5: Variação Inter-Fases e Retenção
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 4: Variação Inter-Fases e Retenção de Aprendizagem no Retorno Letivo

Avaliamos a diferença de pontuação nos mesmos estudantes entre o término de uma fase e o início da fase seguinte:

$$\\Delta_{inter\\text{-}fases} = Pré_{t+1} - Pós_{t}$$

*Cuidado com a interpretação:* diferenças negativas observadas no início de uma nova fase podem decorrer de perda de fluência durante o recesso escolar (fenômeno de *summer learning loss*), mas também de alterações na dificuldade dos itens, mudança de nível de proficiência esperada na troca de série ou readaptação ao ritmo de testes. Portanto, reportamos a variação observada sem atribuir isoladamente uma causalidade única.

Documentamos a análise sob dois critérios amostrais:
1. **Pontuações Disponíveis:** pares com pontuações registradas em ambos os momentos.
2. **Completude Estrita:** pares onde ambas as avaliações atenderam a pelo menos 80% dos itens válidos (`completo_pos_t & completo_pre_t+1`).
""")

    nb.add_code("""
def calcular_retencao_inter_fases(painel, nome_instrumento):
    resultados = []
    
    transicoes = [
        ('Fase 2 Pós -> Fase 3 Pré', 'total_pos_2', 'total_pre_3', 'completo_pos_2', 'completo_pre_3'),
        ('Fase 3 Pós -> Fase 4 Pré', 'total_pos_3', 'total_pre_4', 'completo_pos_3', 'completo_pre_4'),
    ]
    
    for nome_trans, col_pos, col_pre, col_comp_pos, col_comp_pre in transicoes:
        # 1. Amostra com pontuações disponíveis
        disp = painel[painel[col_pos].notna() & painel[col_pre].notna()]
        diff_disp = disp[col_pre] - disp[col_pos]
        t_d, p_d = stats.ttest_rel(disp[col_pre], disp[col_pos])
        
        diag_disp = (
            "Redução observada no retorno letivo" if diff_disp.mean() < -0.5
            else ("Aumento observado no retorno letivo" if diff_disp.mean() > 0.5 else "Estabilidade observada")
        )
        
        resultados.append({
            'Instrumento': nome_instrumento,
            'Transição': nome_trans,
            'Critério_Amostra': 'Pontuações Disponíveis',
            'N Pares': len(disp),
            'Média Pós Anterior': round(disp[col_pos].mean(), 2),
            'Média Pré Seguinte': round(disp[col_pre].mean(), 2),
            'Variação Média (Delta)': round(diff_disp.mean(), 2),
            'DP da Diferença': round(diff_disp.std(), 2),
            't (Student)': round(t_d, 2),
            'p-valor': "< 0.001" if p_d < 0.001 else f"{p_d:.4f}",
            'Diagnóstico': diag_disp,
        })
        
        # 2. Amostra com completude estrita (>=80% itens em ambos)
        comp = painel[painel[col_comp_pos].fillna(False) & painel[col_comp_pre].fillna(False)]
        diff_comp = comp[col_pre] - comp[col_pos]
        t_c, p_c = stats.ttest_rel(comp[col_pre], comp[col_pos])
        
        diag_comp = (
            "Redução observada no retorno letivo" if diff_comp.mean() < -0.5
            else ("Aumento observado no retorno letivo" if diff_comp.mean() > 0.5 else "Estabilidade observada")
        )
        
        resultados.append({
            'Instrumento': nome_instrumento,
            'Transição': nome_trans,
            'Critério_Amostra': 'Completude Estrita (>=80% itens)',
            'N Pares': len(comp),
            'Média Pós Anterior': round(comp[col_pos].mean(), 2),
            'Média Pré Seguinte': round(comp[col_pre].mean(), 2),
            'Variação Média (Delta)': round(diff_comp.mean(), 2),
            'DP da Diferença': round(diff_comp.std(), 2),
            't (Student)': round(t_c, 2),
            'p-valor': "< 0.001" if p_c < 0.001 else f"{p_c:.4f}",
            'Diagnóstico': diag_comp,
        })
        
    return pd.DataFrame(resultados)

ret_v = calcular_retencao_inter_fases(painel_v, 'Vocabulário')
ret_t = calcular_retencao_inter_fases(painel_t, 'TDE II')

tabela_retencao = pd.concat([ret_v, ret_t], ignore_index=True)
print("=== ANÁLISE DE VARIAÇÃO INTER-FASES E RETENÇÃO DE APRENDIZAGEM ===")
display(tabela_retencao)
""")

    # -------------------------------------------------------------
    # Bloco 6: Efeito Intra-aluno vs Efeito de Composição (Atrito)
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 5: Efeito Intra-aluno vs. Efeito de Composição (Análise de Atrito / Attrition)

Em estudos de coorte abertos, as médias globais entre fases mudam por dois fatores concorrentes:
1. **Efeito Intra-aluno:** evolução cognitiva dos estudantes que permaneceram na intervenção.
2. **Efeito de Composição da Amostra:**
   - *Desistentes / Dropouts:* alunos que saíram tinham desempenho inicial diferente dos que ficaram?
   - *Novatos / Entrantes:* alunos que ingressaram em fases posteriores puxaram a média para cima ou para baixo?

Investigamos formalmente essas subpopulações para distinguir ganho pedagógico real de efeitos de seleção amostral.
""")

    nb.add_code("""
def analisar_atrito_fases(df_pareado, nome_instrumento):
    fases = sorted(df_pareado['fase'].unique())
    linhas = []
    
    for i in range(len(fases) - 1):
        f_atual = fases[i]
        f_prox = fases[i+1]
        
        atual_ids = set(df_pareado[df_pareado['fase'] == f_atual]['aluno_id'])
        prox_ids = set(df_pareado[df_pareado['fase'] == f_prox]['aluno_id'])
        
        # Alunos na fase atual divididos em: Continuaram vs Desistiram
        sub_atual = df_pareado[df_pareado['fase'] == f_atual].copy()
        sub_atual['destino'] = sub_atual['aluno_id'].apply(lambda x: f"Continuou na Fase {f_prox}" if x in prox_ids else f"Saiu após Fase {f_atual} (Não participou de F{f_prox})")
        
        for dest, grp in sub_atual.groupby('destino'):
            linhas.append({
                'Instrumento': nome_instrumento,
                'Fase de Referência': f"Fase {f_atual}",
                'Grupo': dest,
                'N': len(grp),
                'Média Pré': round(grp['total_pre'].mean(), 2),
                'Média Pós': round(grp['total_pos'].mean(), 2),
                'Ganho Médio': round((grp['total_pos'] - grp['total_pre']).mean(), 2),
            })
            
        # Alunos na fase seguinte divididos em: Veteranos vs Novatos
        sub_prox = df_pareado[df_pareado['fase'] == f_prox].copy()
        sub_prox['origem'] = sub_prox['aluno_id'].apply(lambda x: f"Veterano (participou de F{f_atual})" if x in atual_ids else f"Novato (entrou em F{f_prox})")
        
        for orig, grp in sub_prox.groupby('origem'):
            linhas.append({
                'Instrumento': nome_instrumento,
                'Fase de Referência': f"Fase {f_prox}",
                'Grupo': orig,
                'N': len(grp),
                'Média Pré': round(grp['total_pre'].mean(), 2),
                'Média Pós': round(grp['total_pos'].mean(), 2),
                'Ganho Médio': round((grp['total_pos'] - grp['total_pre']).mean(), 2),
            })
            
    return pd.DataFrame(linhas)

atrito_v = analisar_atrito_fases(df_pv, 'Vocabulário')
atrito_t = analisar_atrito_fases(df_pt, 'TDE II')

print("=== ANÁLISE DE ATRITO E COMPOSIÇÃO: VOCABULÁRIO ===")
display(atrito_v)

print()
print("=== ANÁLISE DE ATRITO E COMPOSIÇÃO: TDE II ===")
display(atrito_t)
""")

    nb.add_markdown("""
### Comparação Visual: Veteranos vs. Novatos (Impacto da Composição na Fase 3 e 4)
""")

    nb.add_code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (nome, df_atrito, ax) in enumerate([('Vocabulário', atrito_v, axes[0]), ('TDE II', atrito_t, axes[1])]):
    sub = df_atrito[df_atrito['Grupo'].str.contains('Veterano|Novato')].copy()
    sub['Rotulo'] = sub['Fase de Referência'] + " - " + sub['Grupo'].apply(lambda s: s.split(' (')[0])
    bars = ax.barh(sub['Rotulo'], sub['Média Pré'], color='teal', alpha=0.8, edgecolor='black', label='Média Pré')
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.2f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
    ax.set_title(f"Linha de Base (Pré-teste): Veteranos vs. Novatos — {nome}", fontsize=12, fontweight='bold')
    ax.set_xlabel("Média no Pré-teste")
    ax.set_xlim(0, max(sub['Média Pré']) * 1.25)
    ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Bloco 7: Consolidação e Exportação
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 6: Consolidação e Exportação dos Dados Analíticos Longitudinais
""")

    nb.add_code("""
# Consolidar painéis de ambos os instrumentos
painel_completo = pd.concat([painel_v, painel_t], ignore_index=True)

# Exportar painel individual de estudantes
painel_parquet = DATA_ANALYTICS / 'painel_longitudinal.parquet'
painel_csv = DATA_ANALYTICS / 'painel_longitudinal.csv'
painel_completo.to_parquet(painel_parquet, index=False)
painel_completo.to_csv(painel_csv, index=False, encoding='utf-8-sig')

# Exportar métricas agregadas de transição e atrito
metricas_long_parquet = DATA_ANALYTICS / 'metricas_longitudinais.parquet'
metricas_long_csv = DATA_ANALYTICS / 'metricas_longitudinais.csv'

metricas_consolidadas = pd.concat([
    tabela_retencao.assign(Tipo_Metrica='Transicao_Interfases'),
    atrito_v.assign(Tipo_Metrica='Atrito_Composicao'),
    atrito_t.assign(Tipo_Metrica='Atrito_Composicao'),
], ignore_index=True)

metricas_consolidadas.to_parquet(metricas_long_parquet, index=False)
metricas_consolidadas.to_csv(metricas_long_csv, index=False, encoding='utf-8-sig')

print("Arquivos analíticos longitudinais exportados com sucesso:")
print(f"  -> {painel_parquet} ({len(painel_completo):,} registros)")
print(f"  -> {painel_csv}")
print(f"  -> {metricas_long_parquet} ({len(metricas_consolidadas):,} registros)")
print(f"  -> {metricas_long_csv}")
""")

    # Salva notebook
    out_nb = Path(__file__).resolve().parents[1] / "notebooks" / "03_analise_longitudinal.ipynb"
    nb.save(out_nb)


if __name__ == "__main__":
    construir_notebook_03()
