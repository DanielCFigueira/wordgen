"""
Script gerador do Notebook 05: Indicadores Finais para o Dashboard (WordGen)
Executa todas as análises da Etapa 5 e salva notebooks/05_indicadores_dashboard.ipynb
"""

import sys
from pathlib import Path

# Adiciona scripts ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import NotebookBuilder


def construir_notebook_05():
    nb = NotebookBuilder(title="05_indicadores_dashboard")

    # -------------------------------------------------------------
    # Introdução
    # -------------------------------------------------------------
    nb.add_markdown("""
# Nível 5: Indicadores Finais e Datasets para o Dashboard

**Projeto WordGen — Mineração e Análise de Dados Educacionais**

---

### Objetivos da Etapa 5:
1. **Consolidação de KPIs Mestres:** definir os números-chave de topo do painel executivo (alunos atendidos, escolas, turmas, testes, taxa de conclusão e effect sizes globais).
2. **Cubo Analítico de Filtros:** pré-computar métricas descritivas e de impacto ($d_z$) fatiadas por Fase, Escola, Série, Sexo e Zona residencial para garantir resposta instantânea (< 100ms) no Streamlit.
3. **Ranqueamento e Benchmark de Escolas:** estruturar tabelas de comparação institucional para apoiar a tomada de decisão da gestão pedagógica.
4. **Matrizes de Distribuição de Mudança:** consolidar dados de transição (% de alunos com avanço, estabilidade ou queda) para alimentar gráficos de pizza e rosca no dashboard.
5. **Exportação Otimizada:** gerar arquivos Parquet compactos e CSVs em `data/analytics/` para integração direta com a interface Streamlit.
""")

    # -------------------------------------------------------------
    # Bloco 1: Carregamento
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

# Carregamento de bases tratadas
df_v_raw = pd.read_parquet(DATA_FINAL / 'dataset_vocabulario.parquet')
df_t_raw = pd.read_parquet(DATA_FINAL / 'dataset_tde.parquet')
df_pv = pd.read_parquet(DATA_FINAL / 'pareado_vocabulario.parquet')
df_pt = pd.read_parquet(DATA_FINAL / 'pareado_tde.parquet')
df_aluno = pd.read_parquet(DATA_FINAL / 'dim_aluno.parquet')

# Junção com perfil sociodemográfico
df_v_full = df_pv.merge(df_aluno, on='aluno_id', how='left')
df_t_full = df_pt.merge(df_aluno, on='aluno_id', how='left')

# Ganhos
df_v_full['ganho'] = df_v_full['total_pos'] - df_v_full['total_pre']
df_t_full['ganho'] = df_t_full['total_pos'] - df_t_full['total_pre']

print("Bases analíticas prontas para consolidação do Dashboard.")
""")

    # -------------------------------------------------------------
    # Bloco 2: KPIs Mestres do Projeto
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 2: Consolidação dos KPIs Mestres (Cards Executivos)

Calculamos os grandes totais do projeto WordGen e indicadores de eficácia geral.
""")

    nb.add_code("""
# 1. Total de alunos únicos no ecossistema (união de vocab, tde e dim_aluno)
todos_alunos = set(df_v_raw['aluno_id']).union(set(df_t_raw['aluno_id'])).union(set(df_aluno['aluno_id']))
n_alunos_total = len(todos_alunos)

# 2. Escolas ativas (excluindo NAO INFORMADA)
escolas_ativas = [e for e in df_v_raw['escola'].dropna().unique() if e != 'NAO INFORMADA']
n_escolas = len(escolas_ativas)

# 3. Turmas ativas
turmas_v = set(zip(df_v_raw['escola'], df_v_raw['serie'], df_v_raw['turma_letra']))
turmas_t = set(zip(df_t_raw['escola'], df_t_raw['serie'], df_t_raw['turma_letra']))
n_turmas = len(turmas_v.union(turmas_t))

# 4. Avaliações aplicadas
total_testes = len(df_v_raw) + len(df_t_raw)

# 5. Pares completos
comp_v = df_pv['completo_pre'] & df_pv['completo_pos']
comp_t = df_pt['completo_pre'] & df_pt['completo_pos']
n_pares_completos = int(comp_v.sum() + comp_t.sum())
pct_conclusao_pares = round(100 * n_pares_completos / (len(df_pv) + len(df_pt)), 1)

# Tabela de KPIs Mestres
kpis_mestres = pd.DataFrame([
    {'KPI': 'Total de Estudantes Únicos', 'Valor_Numerico': n_alunos_total, 'Valor_Formatado': f"{n_alunos_total:,}", 'Descricao': 'Estudantes com registro em pelo menos uma base do projeto'},
    {'KPI': 'Escolas Públicas Atendidas', 'Valor_Numerico': n_escolas, 'Valor_Formatado': f"{n_escolas}", 'Descricao': 'Escolas municipais participantes da intervenção WordGen'},
    {'KPI': 'Turmas Atendidas', 'Valor_Numerico': n_turmas, 'Valor_Formatado': f"{n_turmas}", 'Descricao': 'Turmas escolares do 6º ao 9º ano participantes'},
    {'KPI': 'Total de Testes Aplicados', 'Valor_Numerico': total_testes, 'Valor_Formatado': f"{total_testes:,}", 'Descricao': 'Soma de aplicações de testes de Vocabulário e TDE II'},
    {'KPI': 'Avaliações Pareadas Completas', 'Valor_Numerico': n_pares_completos, 'Valor_Formatado': f"{n_pares_completos:,}", 'Descricao': 'Estudantes com testes pré e pós válidos (>= 80% itens)'},
    {'KPI': 'Taxa de Completude Pareada', 'Valor_Numerico': pct_conclusao_pares, 'Valor_Formatado': f"{pct_conclusao_pares}%", 'Descricao': 'Percentual de pares válidos com completude técnica estrita'},
])

print("=== CARDS EXECUTIVOS: KPIS MESTRES DO WORDGEN ===")
display(kpis_mestres)
""")

    # -------------------------------------------------------------
    # Bloco 3: Cubo Analítico Fatiável
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 3: Construção do Cubo Analítico Fatiável para Filtros Interativos

Criamos uma tabela unificada onde cada corte dimensional relevante já possui as métricas agregadas calculadas:
- Instrumento (`Vocabulário` / `TDE II`)
- Fase (`Geral`, `Fase 2`, `Fase 3`, `Fase 4`)
- Dimensão de Corte (`Geral`, `Escola`, `Serie`, `Sexo`, `Zona`)
- Valor do Corte
""")

    nb.add_code("""
def gerar_linha_cubo(df_sub, instrumento, fase_label, dimensao, valor_corte):
    completos = df_sub[df_sub['completo_pre'] & df_sub['completo_pos']]
    n_total = len(df_sub)
    n_comp = len(completos)
    
    if n_comp == 0:
        return None
        
    pre = completos['total_pre']
    pos = completos['total_pos']
    ganho = pos - pre
    
    media_pre = pre.mean()
    media_pos = pos.mean()
    ganho_m = ganho.mean()
    s_d = ganho.std()
    mediana_g = ganho.median()
    
    dz = ganho_m / s_d if s_d > 0 else 0.0
    se_dz = np.sqrt(1.0/n_comp + (dz**2)/(2.0*n_comp))
    
    n_pos_ganho = (ganho > 0).sum()
    n_zero_ganho = (ganho == 0).sum()
    n_neg_ganho = (ganho < 0).sum()
    
    return {
        'Instrumento': instrumento,
        'Fase': fase_label,
        'Dimensao': dimensao,
        'Corte': str(valor_corte),
        'N_Total_Pares': n_total,
        'N_Completos': n_comp,
        'Media_Pre': round(media_pre, 2),
        'Media_Pos': round(media_pos, 2),
        'Ganho_Medio': round(ganho_m, 2),
        'Mediana_Ganho': round(mediana_g, 2),
        'Desvio_Padrao_Ganho': round(s_d, 2),
        'Effect_Size_dz': round(dz, 3),
        'IC_95_Inf': round(dz - 1.96 * se_dz, 3),
        'IC_95_Sup': round(dz + 1.96 * se_dz, 3),
        'Pct_Alunos_Ganho': round(100 * n_pos_ganho / n_comp, 1),
        'Pct_Alunos_Inalterado': round(100 * n_zero_ganho / n_comp, 1),
        'Pct_Alunos_Queda': round(100 * n_neg_ganho / n_comp, 1),
    }

def construir_cubo_analitico():
    linhas_cubo = []
    
    for nome_inst, df_full in [('Vocabulário', df_v_full), ('TDE II', df_t_full)]:
        fases = ['Geral'] + [f"Fase {f}" for f in sorted(df_full['fase'].unique())]
        
        for f_label in fases:
            sub_fase = df_full if f_label == 'Geral' else df_full[df_full['fase'] == int(f_label.split()[-1])]
            
            # 1. Corte Geral
            r = gerar_linha_cubo(sub_fase, nome_inst, f_label, 'Geral', 'Todos')
            if r: linhas_cubo.append(r)
            
            # 2. Corte por Escola
            for esc in sub_fase['escola'].dropna().unique():
                r = gerar_linha_cubo(sub_fase[sub_fase['escola'] == esc], nome_inst, f_label, 'Escola', esc)
                if r: linhas_cubo.append(r)
                
            # 3. Corte por Série
            for ser in sorted(sub_fase['serie'].dropna().unique()):
                r = gerar_linha_cubo(sub_fase[sub_fase['serie'] == ser], nome_inst, f_label, 'Serie', f"{int(ser)}º Ano")
                if r: linhas_cubo.append(r)
                
            # 4. Corte por Sexo
            for sx in ['M', 'F']:
                r = gerar_linha_cubo(sub_fase[sub_fase['sexo'] == sx], nome_inst, f_label, 'Sexo', 'Masculino' if sx == 'M' else 'Feminino')
                if r: linhas_cubo.append(r)
                
            # 5. Corte por Zona
            for zn in ['URBANA', 'RURAL']:
                r = gerar_linha_cubo(sub_fase[sub_fase['zona'] == zn], nome_inst, f_label, 'Zona', zn.capitalize())
                if r: linhas_cubo.append(r)
                
    return pd.DataFrame(linhas_cubo)

df_cubo = construir_cubo_analitico()
print(f"Cubo Analítico gerado com {len(df_cubo):,} fatias pré-computadas!")
display(df_cubo.head(10))
""")

    # -------------------------------------------------------------
    # Bloco 4: Ranqueamento e Benchmark de Escolas
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 4: Tabela de Ranqueamento e Comparativo de Escolas

Estruturamos um ranking para visualização rápida no painel, destacando o volume de pares completos, o ganho médio e o $d_z$ de Cohen.
""")

    nb.add_code("""
ranking_escolas = df_cubo[
    (df_cubo['Dimensao'] == 'Escola') & 
    (df_cubo['Fase'] != 'Geral') & 
    (df_cubo['Corte'] != 'NAO INFORMADA') &
    (df_cubo['N_Completos'] >= 15)
][['Instrumento', 'Fase', 'Corte', 'N_Completos', 'Media_Pre', 'Media_Pos', 'Ganho_Medio', 'Effect_Size_dz', 'Pct_Alunos_Ganho']]

ranking_escolas = ranking_escolas.rename(columns={'Corte': 'Escola'}).sort_values(by=['Instrumento', 'Fase', 'Effect_Size_dz'], ascending=[True, True, False])

print("=== TOP ESCOLAS POR EFFECT SIZE (EXEMPLO FASE 2) ===")
display(ranking_escolas[ranking_escolas['Fase'] == 'Fase 2'].head(10))
""")

    # -------------------------------------------------------------
    # Bloco 5: Visualizações Simuladas para o Dashboard
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 5: Simulação Visual dos Gráficos do Dashboard

Geramos representações visuais correspondentes às telas interativas do dashboard.
""")

    nb.add_code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Barras de Effect Size por Escola na Fase 2
ax1 = axes[0]
sub_rank = ranking_escolas[(ranking_escolas['Fase'] == 'Fase 2') & (ranking_escolas['Instrumento'] == 'Vocabulário')].sort_values(by='Effect_Size_dz')
nomes_esc_curtos = [e.replace('EMEB ', '').replace('EMEF ', '') for e in sub_rank['Escola']]
cores_bar = ['forestgreen' if dz >= 0.2 else ('goldenrod' if dz >= 0 else 'firebrick') for dz in sub_rank['Effect_Size_dz']]

bars1 = ax1.barh(nomes_esc_curtos, sub_rank['Effect_Size_dz'], color=cores_bar, edgecolor='black', alpha=0.85)
ax1.axvline(0, color='black', linewidth=1)
ax1.axvline(0.2, color='green', linestyle='--', alpha=0.7, label='Benchmark Pequeno (0.2)')

for bar in bars1:
    w = bar.get_width()
    ax1.annotate(f"{w:+.2f}", xy=(w, bar.get_y() + bar.get_height() / 2),
                 xytext=(5 if w >= 0 else -25, 0), textcoords="offset points",
                 ha='left', va='center', fontsize=9, fontweight='bold')
    
ax1.set_title("Vocabulário Fase 2: Effect Size ($d_z$) por Escola", fontsize=12, fontweight='bold')
ax1.set_xlabel("Effect Size ($d_z$)")
ax1.grid(alpha=0.3, axis='x')
ax1.legend(loc='lower right')

# Gráfico 2: Proporção de Alunos com Ganho vs. Queda por Fase
ax2 = axes[1]
sub_mud = df_cubo[(df_cubo['Dimensao'] == 'Geral') & (df_cubo['Fase'] != 'Geral') & (df_cubo['Instrumento'] == 'Vocabulário')]
x = np.arange(len(sub_mud))
w = 0.25

ax2.bar(x - w, sub_mud['Pct_Alunos_Ganho'], width=w, label='Ganho (+)', color='mediumseagreen', edgecolor='black', alpha=0.85)
ax2.bar(x, sub_mud['Pct_Alunos_Inalterado'], width=w, label='Inalterado (=)', color='khaki', edgecolor='black', alpha=0.85)
ax2.bar(x + w, sub_mud['Pct_Alunos_Queda'], width=w, label='Queda (-)', color='indianred', edgecolor='black', alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(sub_mud['Fase'], fontsize=11)
ax2.set_ylabel("% de Estudantes")
ax2.set_title("Vocabulário: Distribuição de Alunos por Categoria de Mudança", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Bloco 6: Exportação dos Dados do Dashboard
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 6: Exportação dos Datasets Estruturados para o Streamlit
""")

    nb.add_code("""
# 1. KPIs de Topo
kpis_mestres.to_parquet(DATA_ANALYTICS / 'dashboard_kpis.parquet', index=False)
kpis_mestres.to_csv(DATA_ANALYTICS / 'dashboard_kpis.csv', index=False, encoding='utf-8-sig')

# 2. Cubo de Filtros Principal
df_cubo.to_parquet(DATA_ANALYTICS / 'dashboard_cubo_filtros.parquet', index=False)
df_cubo.to_csv(DATA_ANALYTICS / 'dashboard_cubo_filtros.csv', index=False, encoding='utf-8-sig')

# 3. Ranking de Escolas
ranking_escolas.to_parquet(DATA_ANALYTICS / 'dashboard_escolas_ranking.parquet', index=False)
ranking_escolas.to_csv(DATA_ANALYTICS / 'dashboard_escolas_ranking.csv', index=False, encoding='utf-8-sig')

print("Datasets analíticos do Dashboard exportados com sucesso em data/analytics/:")
print(f"  -> dashboard_kpis.parquet")
print(f"  -> dashboard_cubo_filtros.parquet ({len(df_cubo):,} registros)")
print(f"  -> dashboard_escolas_ranking.parquet ({len(ranking_escolas):,} registros)")
""")

    # Salva notebook
    out_nb = Path(__file__).resolve().parents[1] / "notebooks" / "05_indicadores_dashboard.ipynb"
    nb.save(out_nb)


if __name__ == "__main__":
    construir_notebook_05()
