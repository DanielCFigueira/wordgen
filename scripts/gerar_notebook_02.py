"""
Script gerador do Notebook 02: Desempenho e Effect Size (WordGen)
Executa todas as análises da Etapa 2 e salva notebooks/02_desempenho_effect_size.ipynb
"""

import sys
from pathlib import Path

# Adiciona scripts ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import NotebookBuilder


def construir_notebook_02():
    nb = NotebookBuilder(title="02_desempenho_effect_size")

    # -------------------------------------------------------------
    # Introdução
    # -------------------------------------------------------------
    nb.add_markdown("""
# Nível 2: Desempenho e Effect Size ($d_z$)

**Projeto WordGen — Mineração e Análise de Dados Educacionais**

---

### Objetivos da Etapa 2:
1. **Estatísticas descritivas:** avaliar pontuações de Vocabulário (0 a 50) e TDE (0 a 80) no pré e pós-teste.
2. **Distribuição dos ganhos:** analisar a diferença individual $D = Pós - Pré$, identificando proporções de alunos com ganho, estabilidade ou queda.
3. **Effect Size pareado ($d_z$ de Cohen):** quantificar a magnitude padronizada do impacto da intervenção com intervalos de confiança (IC 95%) e testes de significância (t pareado e Wilcoxon).
4. **Desagregações:** investigar a heterogeneidade do efeito por escola, série (ano escolar) e turma.
5. **Exportação analítica:** salvar as métricas consolidadas em `data/analytics/` para alimentação do dashboard e do agente RAG.
""")

    # -------------------------------------------------------------
    # Configuração e Carregamento
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 1: Configuração do Ambiente e Carregamento dos Dados Pareados
""")

    nb.add_code("""
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Configuração de caminhos determinísticos
ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent

DATA_FINAL = ROOT / 'data' / 'final'
DATA_ANALYTICS = ROOT / 'data' / 'analytics'
DATA_ANALYTICS.mkdir(parents=True, exist_ok=True)

# Carregamento dos dados analíticos pareados
df_pv = pd.read_parquet(DATA_FINAL / 'pareado_vocabulario.parquet')
df_pt = pd.read_parquet(DATA_FINAL / 'pareado_tde.parquet')

print(f"Pares carregados - Vocabulário: {len(df_pv):,} registros | TDE: {len(df_pt):,} registros")
""")

    # -------------------------------------------------------------
    # Estatísticas Descritivas
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 2: Estatísticas Descritivas de Pontuação (Pré vs. Pós)

Calculamos média, desvio padrão, mediana, intervalo interquartil (IQR), mínimo e máximo para cada instrumento e fase.
""")

    nb.add_code("""
def gerar_estatisticas_descritivas(df, nome_instrumento, escala_max):
    registros = []
    
    # Amostra global e por fase
    fases = ['Todas as Fases'] + sorted(df['fase'].unique().tolist())
    
    for f in fases:
        sub = df if f == 'Todas as Fases' else df[df['fase'] == f]
        
        # Visão geral de pares válidos
        for momento, col in [('Pré-teste', 'total_pre'), ('Pós-teste', 'total_pos')]:
            vals = sub[col].dropna()
            registros.append({
                'Instrumento': nome_instrumento,
                'Fase': str(f),
                'Momento': momento,
                'N': int(len(vals)),
                'Média': round(vals.mean(), 2),
                'Desv. Padrão': round(vals.std(), 2),
                'Mediana': round(vals.median(), 2),
                'Q25': round(vals.quantile(0.25), 2),
                'Q75': round(vals.quantile(0.75), 2),
                'IQR': round(vals.quantile(0.75) - vals.quantile(0.25), 2),
                'Mínimo': round(vals.min(), 1),
                'Máximo': round(vals.max(), 1),
                '% Escala': round(100 * vals.mean() / escala_max, 1),
            })
            
    return pd.DataFrame(registros)

desc_v = gerar_estatisticas_descritivas(df_pv, 'Vocabulário', 50)
desc_t = gerar_estatisticas_descritivas(df_pt, 'TDE II', 80)

print("=== ESTATÍSTICAS DESCRITIVAS: VOCABULÁRIO ===")
display(desc_v)

print()
print("=== ESTATÍSTICAS DESCRITIVAS: TDE II ===")
display(desc_t)
""")

    # -------------------------------------------------------------
    # Análise das Diferenças e Mudanças
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 3: Análise dos Ganhos Individuais ($D = Pós - Pré$)

Avaliamos a magnitude das mudanças intra-aluno e categorizamos os estudantes em três grupos:
- **Ganho positivo ($D > 0$):** avanço na pontuação após a intervenção.
- **Estabilidade ($D = 0$):** pontuação idêntica.
- **Queda ($D < 0$):** desempenho inferior no pós-teste.
""")

    nb.add_code("""
for nome, df in [('Vocabulário', df_pv), ('TDE II', df_pt)]:
    df['ganho'] = df['total_pos'] - df['total_pre']
    df['pct_ganho'] = (df['ganho'] / df['total_pre'].replace(0, np.nan)) * 100
    
    # Classificação de mudança
    condicoes = [df['ganho'] > 0, df['ganho'] == 0, df['ganho'] < 0]
    escolhas = ['Ganho (+)', 'Inalterado (=)', 'Queda (-)']
    df['categoria_mudanca'] = np.select(condicoes, escolhas, default='Indefinido')

def resumir_mudancas(df, nome_instrumento):
    resumo = []
    for f in sorted(df['fase'].unique()):
        sub = df[df['fase'] == f]
        n_total = len(sub)
        n_pos = (sub['ganho'] > 0).sum()
        n_zero = (sub['ganho'] == 0).sum()
        n_neg = (sub['ganho'] < 0).sum()
        
        resumo.append({
            'Instrumento': nome_instrumento,
            'Fase': f"Fase {f}",
            'N Pares': n_total,
            'Ganho Médio (D)': round(sub['ganho'].mean(), 2),
            'Mediana Ganho': round(sub['ganho'].median(), 2),
            'DP Ganho (s_D)': round(sub['ganho'].std(), 2),
            'Alunos com Ganho': f"{n_pos} ({100*n_pos/n_total:.1f}%)",
            'Alunos Inalterados': f"{n_zero} ({100*n_zero/n_total:.1f}%)",
            'Alunos com Queda': f"{n_neg} ({100*n_neg/n_total:.1f}%)",
        })
    return pd.DataFrame(resumo)

tabela_mudancas = pd.concat([
    resumir_mudancas(df_pv, 'Vocabulário'),
    resumir_mudancas(df_pt, 'TDE II'),
], ignore_index=True)

display(tabela_mudancas)
""")

    # -------------------------------------------------------------
    # Visualizações de Distribuição de Ganhos
    # -------------------------------------------------------------
    nb.add_markdown("""
### Visualização das Mudanças e Densidades de Ganhos
""")

    nb.add_code("""
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
fig.suptitle('Distribuição das Diferenças Intra-aluno (Pós - Pré) por Fase', fontsize=15, y=1.02)

for i, (nome, df, cor) in enumerate([('Vocabulário', df_pv, 'steelblue'), ('TDE II', df_pt, 'darkorange')]):
    for j, fase in enumerate([2, 3, 4]):
        ax = axes[i, j]
        sub = df[df['fase'] == fase]['ganho'].dropna()
        
        media_g = sub.mean()
        mediana_g = sub.median()
        
        ax.hist(sub, bins=25, color=cor, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Sem mudança (0)')
        ax.axvline(media_g, color='darkgreen', linestyle='-', linewidth=2, label=f'Média ({media_g:+.2f})')
        ax.axvline(mediana_g, color='purple', linestyle=':', linewidth=2, label=f'Mediana ({mediana_g:+.2f})')
        
        ax.set_title(f"{nome} — Fase {fase} (N={len(sub):,})", fontsize=12)
        ax.set_xlabel('Ganho (Pós - Pré)')
        if j == 0:
            ax.set_ylabel('Densidade')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Cálculo Formal de Effect Size (d_z de Cohen)
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 4: Effect Size Pareado ($d_z$ de Cohen) e Testes de Significância

Para medidas repetidas (mesmo sujeito avaliado em dois momentos), o tamanho de efeito apropriado é o **$d_z$ de Cohen**:

$$d_z = \\frac{\\bar{D}}{s_D}$$

onde:
- $\\bar{D} = \\frac{1}{N} \\sum_{i=1}^N (Pós_i - Pré_i)$ é a média das diferenças intra-aluno;
- $s_D = \\sqrt{\\frac{1}{N-1} \\sum_{i=1}^N (D_i - \\bar{D})^2}$ é o desvio padrão das diferenças.

O erro padrão assintótico de $d_z$ e o intervalo de confiança de 95% são dados por:
$$SE(d_z) = \\sqrt{\\frac{1}{N} + \\frac{d_z^2}{2N}}$$
$$IC_{95\\%}(d_z) = [d_z - 1.96 \\cdot SE(d_z), \\; d_z + 1.96 \\cdot SE(d_z)]$$

Calculamos para duas populações:
1. **Amostra Completa:** avaliações com pelo menos 80% dos itens válidos (`completo_pre & completo_pos` — padrão de referência).
2. **Amostra Total:** todos os pares válidos (`n_validos > 0`).
""")

    nb.add_code("""
def calcular_metricas_effect_size(df, nome_instrumento, apenas_completos=True):
    linhas = []
    
    for fase in sorted(df['fase'].unique()):
        sub = df[df['fase'] == fase]
        if apenas_completos:
            sub = sub[sub['completo_pre'] & sub['completo_pos']]
            
        pre = sub['total_pre'].to_numpy()
        pos = sub['total_pos'].to_numpy()
        diferencas = pos - pre
        n = len(diferencas)
        
        if n == 0:
            continue
            
        media_pre = np.mean(pre)
        dp_pre = np.std(pre, ddof=1)
        media_pos = np.mean(pos)
        dp_pos = np.std(pos, ddof=1)
        
        d_medio = np.mean(diferencas)
        s_d = np.std(diferencas, ddof=1)
        
        # d_z de Cohen
        dz = d_medio / s_d if s_d > 0 else 0.0
        se_dz = np.sqrt((1.0 / n) + (dz**2 / (2.0 * n)))
        ci_inf = dz - 1.96 * se_dz
        ci_sup = dz + 1.96 * se_dz
        
        # Teste t pareado
        t_stat, p_val_t = stats.ttest_rel(pos, pre)
        
        # Teste não-paramétrico de Wilcoxon
        try:
            w_stat, p_val_w = stats.wilcoxon(pos, pre)
        except Exception:
            w_stat, p_val_w = np.nan, np.nan
            
        # Classificação qualitativa segundo convenção de Cohen
        if abs(dz) < 0.20:
            interpretacao = "Trivial / Insignificante"
        elif abs(dz) < 0.50:
            interpretacao = "Pequeno (Small)"
        elif abs(dz) < 0.80:
            interpretacao = "Médio (Medium)"
        else:
            interpretacao = "Grande (Large)"
            
        linhas.append({
            'Instrumento': nome_instrumento,
            'Fase': f"Fase {fase}",
            'Filtro': 'Completas (>=80%)' if apenas_completos else 'Todos os Pares Válidos',
            'N Pares': n,
            'Média Pré': round(media_pre, 2),
            'Média Pós': round(media_pos, 2),
            'Ganho Médio': round(d_medio, 2),
            's_D': round(s_d, 2),
            'd_z (Cohen)': round(dz, 3),
            'SE (d_z)': round(se_dz, 3),
            'IC 95% Inf': round(ci_inf, 3),
            'IC 95% Sup': round(ci_sup, 3),
            't (Student)': round(t_stat, 2),
            'p-valor (t)': "< 0.001" if p_val_t < 0.001 else f"{p_val_t:.4f}",
            'p-valor (Wilcoxon)': "< 0.001" if p_val_w < 0.001 else f"{p_val_w:.4f}",
            'Interpretação': interpretacao,
        })
        
    return pd.DataFrame(linhas)

# Cálculo para amostras completas
es_vocab_comp = calcular_metricas_effect_size(df_pv, 'Vocabulário', apenas_completos=True)
es_tde_comp = calcular_metricas_effect_size(df_pt, 'TDE II', apenas_completos=True)
tabela_es_completos = pd.concat([es_vocab_comp, es_tde_comp], ignore_index=True)

print("=== EFFECT SIZE PAREADO (AMOSTRA COMPLETA >= 80% ITENS) ===")
display(tabela_es_completos)

# Cálculo para amostras totais
es_vocab_tot = calcular_metricas_effect_size(df_pv, 'Vocabulário', apenas_completos=False)
es_tde_tot = calcular_metricas_effect_size(df_pt, 'TDE II', apenas_completos=False)
tabela_es_totais = pd.concat([es_vocab_tot, es_tde_tot], ignore_index=True)

print()
print("=== EFFECT SIZE PAREADO (TODOS OS PARES VÁLIDOS) ===")
display(tabela_es_totais)
""")

    # -------------------------------------------------------------
    # Visualização de Forest Plot do Effect Size
    # -------------------------------------------------------------
    nb.add_markdown("""
### Gráfico de Intervalos de Confiança (Forest Plot) do Effect Size
""")

    nb.add_code("""
plt.figure(figsize=(10, 5))

cores = {'Vocabulário': 'royalblue', 'TDE II': 'darkorange'}

y_ticks = []
y_labels = []

for idx, row in tabela_es_completos.iterrows():
    y = len(tabela_es_completos) - 1 - idx
    cor = cores.get(row['Instrumento'], 'black')
    
    # Linha do IC 95%
    plt.errorbar(
        x=row['d_z (Cohen)'],
        y=y,
        xerr=[[row['d_z (Cohen)'] - row['IC 95% Inf']], [row['IC 95% Sup'] - row['d_z (Cohen)']]],
        fmt='o',
        color=cor,
        ecolor=cor,
        elinewidth=2.5,
        capsize=6,
        markersize=7,
        label=row['Instrumento'] if row['Instrumento'] not in [l[0] for l in y_labels] else ""
    )
    y_ticks.append(y)
    y_labels.append((row['Instrumento'], f"{row['Instrumento']} {row['Fase']} (N={row['N Pares']:,})"))

plt.axvline(0, color='gray', linestyle='--', linewidth=1.2)
plt.axvline(0.2, color='green', linestyle=':', alpha=0.6, label='Pequeno (0.2)')
plt.axvline(-0.2, color='red', linestyle=':', alpha=0.6, label='Queda (-0.2)')

plt.yticks(y_ticks, [label[1] for label in y_labels], fontsize=11)
plt.xlabel("Effect Size Pareado ($d_z$ de Cohen) com IC 95%", fontsize=12)
plt.title("Impacto da Intervenção WordGen por Instrumento e Fase", fontsize=14)
plt.grid(alpha=0.3, axis='x')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Comparações por Escola e Série
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 5: Desagregações por Escola, Série e Turma

Investigamos como o impacto varia entre as diferentes escolas e séries atendidas, aplicando filtro de suficiência amostral ($N \\ge 15$).
""")

    nb.add_code("""
def analisar_efeito_por_grupo(df, grupo_col, min_n=15):
    linhas = []
    
    for (fase, grp), dados in df.groupby(['fase', grupo_col]):
        completos = dados[dados['completo_pre'] & dados['completo_pos']]
        n = len(completos)
        
        if n < min_n:
            continue
            
        pre = completos['total_pre']
        pos = completos['total_pos']
        diff = pos - pre
        
        d_medio = diff.mean()
        s_d = diff.std()
        dz = d_medio / s_d if s_d > 0 else 0.0
        
        linhas.append({
            'Fase': f"Fase {fase}",
            grupo_col.capitalize(): grp,
            'N Pares': n,
            'Média Pré': round(pre.mean(), 2),
            'Média Pós': round(pos.mean(), 2),
            'Ganho Médio': round(d_medio, 2),
            's_D': round(s_d, 2),
            'd_z': round(dz, 3),
        })
        
    return pd.DataFrame(linhas)

escolas_v = analisar_efeito_por_grupo(df_pv, 'escola')
escolas_t = analisar_efeito_por_grupo(df_pt, 'escola')

print("=== IMPACTO POR ESCOLA (VOCABULÁRIO) ===")
display(escolas_v.sort_values(by=['Fase', 'd_z'], ascending=[True, False]))

print()
print("=== IMPACTO POR ESCOLA (TDE II) ===")
display(escolas_t.sort_values(by=['Fase', 'd_z'], ascending=[True, False]))
""")

    nb.add_markdown("""
### Desagregação por Série (6º ao 9º Ano)
""")

    nb.add_code("""
series_v = analisar_efeito_por_grupo(df_pv, 'serie', min_n=15)
series_t = analisar_efeito_por_grupo(df_pt, 'serie', min_n=15)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for i, (nome, df_s, ax) in enumerate([('Vocabulário', series_v, axes[0]), ('TDE II', series_t, axes[1])]):
    pivot_dz = df_s.pivot(index='Serie', columns='Fase', values='d_z')
    pivot_dz.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', alpha=0.85)
    ax.set_title(f"Effect Size ($d_z$) por Série — {nome}", fontsize=13)
    ax.set_xlabel("Série (Ano Escolar)")
    ax.set_ylabel("Effect Size ($d_z$)")
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(title='Fase')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Consolidação e Exportação
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 6: Consolidação e Exportação das Métricas Analíticas
""")

    nb.add_code("""
# Consolidar tabela final de effect size
df_analytics_es = pd.concat([
    tabela_es_completos.assign(Estratificacao='Geral_Completo'),
    tabela_es_totais.assign(Estratificacao='Geral_Total'),
], ignore_index=True)

# Salvar em Parquet e CSV para uso no Dashboard e no RAG
out_parquet = DATA_ANALYTICS / 'metricas_effect_size.parquet'
out_csv = DATA_ANALYTICS / 'metricas_effect_size.csv'

df_analytics_es.to_parquet(out_parquet, index=False)
df_analytics_es.to_csv(out_csv, index=False, encoding='utf-8-sig')

# Exportar detalhamento por escola e série
escolas_v.assign(Instrumento='Vocabulário').to_parquet(DATA_ANALYTICS / 'effect_size_escolas_vocab.parquet', index=False)
escolas_t.assign(Instrumento='TDE II').to_parquet(DATA_ANALYTICS / 'effect_size_escolas_tde.parquet', index=False)

print(f"Métricas exportadas com sucesso:")
print(f"  -> {out_parquet}")
print(f"  -> {out_csv}")
""")

    # Salva notebook
    out_nb = Path(__file__).resolve().parents[1] / "notebooks" / "02_desempenho_effect_size.ipynb"
    nb.save(out_nb)


if __name__ == "__main__":
    construir_notebook_02()
