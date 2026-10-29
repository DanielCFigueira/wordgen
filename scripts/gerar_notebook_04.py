"""
Script gerador do Notebook 04: Perfil Socioeconômico e Equidade (WordGen)
Executa todas as análises da Etapa 4 e salva notebooks/04_perfil_socioeconomico.ipynb
"""

import sys
from pathlib import Path

# Adiciona scripts ao path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import NotebookBuilder


def construir_notebook_04():
    nb = NotebookBuilder(title="04_perfil_socioeconomico")

    # -------------------------------------------------------------
    # Introdução
    # -------------------------------------------------------------
    nb.add_markdown("""
# Nível 4: Perfil Socioeconômico, Desempenho e Equidade Educacional

**Projeto WordGen — Mineração e Análise de Dados Educacionais**

---

### Objetivos da Etapa 4:
1. **Perfil Demográfico:** caracterizar a população estudantil catalogada na dimensão `dim_aluno` (sexo, faixa etária, cor/raça, zona residencial e necessidades especiais).
2. **Cobertura Socioeconômica:** documentar com transparência a cobertura dos dados demográficos em relação aos testes aplicados (94.76% de correspondência; 140 estudantes mantidos nas análises de desempenho sob categoria não informada).
3. **Desempenho e Effect Size por Subgrupos:** avaliar a evolução das pontuações em recortes de gênero e faixa etária, reportando tanto amostras completas (>=80% itens) quanto o total de pares válidos.
4. **Equidade Territorial (Zona Rural vs. Urbana):** quantificar a evolução do gap de pontuação ($Gap = Urbana - Rural$; $\\Delta Gap = Gap_{pós} - Gap_{pré}$), investigando padrões de convergência observados.
5. **Inclusão Escolar:** analisar o acompanhamento dos 66 estudantes com necessidades educacionais especiais registradas, preservando o rigor conceitual da categoria `NAO_REGISTRADO` (ausência de registro formal, não confirmação de ausência de necessidade).
6. **Exportação Analítica:** gerar os arquivos consolidados `perfil_socioeconomico_desempenho.parquet` e `metricas_equidade.parquet` em `data/analytics/`.
""")

    # -------------------------------------------------------------
    # Bloco 1: Carregamento dos Dados
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

# Carregamento de dim_aluno e bases pareadas
df_aluno = pd.read_parquet(DATA_FINAL / 'dim_aluno.parquet')
df_pv = pd.read_parquet(DATA_FINAL / 'pareado_vocabulario.parquet')
df_pt = pd.read_parquet(DATA_FINAL / 'pareado_tde.parquet')

# Merge com dados demográficos (preservando estudantes sem correspondência)
df_v_socio = df_pv.merge(df_aluno, on='aluno_id', how='left')
df_t_socio = df_pt.merge(df_aluno, on='aluno_id', how='left')

# Preenchimento explícito de nulos para categorias socioeconômicas
cols_socio = ['sexo', 'faixa_etaria', 'cor_raca', 'zona', 'necessidade_especial']
for col in cols_socio:
    df_v_socio[col] = df_v_socio[col].fillna('NAO INFORMADA' if col != 'necessidade_especial' else 'NAO_REGISTRADO')
    df_t_socio[col] = df_t_socio[col].fillna('NAO INFORMADA' if col != 'necessidade_especial' else 'NAO_REGISTRADO')

# Cálculo dos ganhos individuais
df_v_socio['ganho'] = df_v_socio['total_pos'] - df_v_socio['total_pre']
df_t_socio['ganho'] = df_t_socio['total_pos'] - df_t_socio['total_pre']

print(f"Alunos na dim_aluno: {len(df_aluno):,}")
print(f"Pares Vocabulário com perfil demográfico: {(df_v_socio['sexo'] != 'NAO INFORMADA').sum():,} de {len(df_v_socio):,} ({100*(df_v_socio['sexo'] != 'NAO INFORMADA').mean():.1f}%)")
print(f"Pares TDE II com perfil demográfico:      {(df_t_socio['sexo'] != 'NAO INFORMADA').sum():,} de {len(df_t_socio):,} ({100*(df_t_socio['sexo'] != 'NAO INFORMADA').mean():.1f}%)")
""")

    # -------------------------------------------------------------
    # Bloco 2: Distribuição Demográfica Geral
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 2: Perfil Sociodemográfico da População Atendida (`dim_aluno`)

Apresentamos a distribuição univariada dos 2.530 estudantes catalogados na dimensão socioeconômica e a auditoria de cobertura dos 2.670 estudantes únicos presentes nos testes.
""")

    nb.add_code("""
def tabela_frequencia(df, coluna, rotulo):
    cont = df[coluna].value_counts(dropna=False)
    pct = (cont / len(df) * 100).round(1)
    df_res = pd.DataFrame({
        'Variável': rotulo,
        'Categoria': cont.index.astype(str),
        'N Alunos': cont.values,
        '% Alunos': pct.values
    })
    return df_res

variaveis_socio = [
    ('sexo', 'Sexo'),
    ('faixa_etaria', 'Faixa Etária'),
    ('cor_raca', 'Cor / Raça'),
    ('zona', 'Zona Residencial'),
    ('necessidade_especial', 'Necessidade Especial'),
]

tabela_perfil = pd.concat([tabela_frequencia(df_aluno, col, rot) for col, rot in variaveis_socio], ignore_index=True)
display(tabela_perfil)
""")

    nb.add_markdown("""
### Visualização das Dimensões Sociodemográficas
""")

    nb.add_code("""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sexo
ax1 = axes[0, 0]
sexo_counts = df_aluno['sexo'].value_counts()
ax1.pie(sexo_counts, labels=[f"Meninos (M): {sexo_counts['M']:,} ({sexo_counts['M']/len(df_aluno)*100:.1f}%)",
                             f"Meninas (F): {sexo_counts['F']:,} ({sexo_counts['F']/len(df_aluno)*100:.1f}%)"],
        autopct='%1.1f%%', colors=['cornflowerblue', 'lightcoral'], startangle=140, textprops={'fontsize': 10})
ax1.set_title("Distribuição por Sexo", fontsize=12, fontweight='bold')

# 2. Faixa Etária
ax2 = axes[0, 1]
fe_counts = df_aluno['faixa_etaria'].value_counts().sort_index()
bars2 = ax2.bar(fe_counts.index, fe_counts.values, color='mediumseagreen', edgecolor='black', alpha=0.85)
for bar in bars2:
    h = bar.get_height()
    ax2.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_title("Distribuição por Faixa Etária", fontsize=12, fontweight='bold')
ax2.set_ylabel("N Alunos")
ax2.grid(alpha=0.3, axis='y')

# 3. Cor/Raça
ax3 = axes[1, 0]
cr_counts = df_aluno['cor_raca'].value_counts().sort_values(ascending=True)
bars3 = ax3.barh(cr_counts.index, cr_counts.values, color='chocolate', edgecolor='black', alpha=0.85)
for bar in bars3:
    w = bar.get_width()
    ax3.annotate(f"{w:,}", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                 textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold')
ax3.set_title("Distribuição por Cor / Raça", fontsize=12, fontweight='bold')
ax3.set_xlabel("N Alunos")
ax3.set_xlim(0, max(cr_counts.values) * 1.18)
ax3.grid(alpha=0.3, axis='x')

# 4. Zona Residencial
ax4 = axes[1, 1]
zn_counts = df_aluno['zona'].value_counts()
bars4 = ax4.bar(zn_counts.index, zn_counts.values, color=['goldenrod', 'darkcyan', 'gray'], edgecolor='black', alpha=0.85)
for bar in bars4:
    h = bar.get_height()
    pct = 100 * h / len(df_aluno)
    ax4.annotate(f"{h:,} ({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                 textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_title("Distribuição por Zona Residencial", fontsize=12, fontweight='bold')
ax4.set_ylabel("N Alunos")
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Bloco 3: Desempenho por Sexo e Faixa Etária
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 3: Desempenho e Effect Size por Sexo e Faixa Etária

Avaliamos as pontuações e o effect size pareado ($d_z$) por subgrupo demográfico.
Documentamos formalmente os resultados para a amostra de avaliações completas (padrão de referência) e também para o total de pares válidos.
""")

    nb.add_code("""
def calcular_efeito_por_subgrupo(df, col_subgrupo, nome_instrumento, min_n=15):
    registros = []
    
    for (fase, grp), dados in df.groupby(['fase', col_subgrupo]):
        for apenas_comp in [True, False]:
            sub = dados[dados['completo_pre'] & dados['completo_pos']] if apenas_comp else dados
            n = len(sub)
            
            if n < min_n:
                continue
                
            pre = sub['total_pre']
            pos = sub['total_pos']
            diff = pos - pre
            
            d_medio = diff.mean()
            s_d = diff.std()
            dz = d_medio / s_d if s_d > 0 else 0.0
            se_dz = np.sqrt(1.0/n + (dz**2)/(2.0*n))
            
            t_stat, p_val = stats.ttest_rel(pos, pre)
            
            registros.append({
                'Instrumento': nome_instrumento,
                'Fase': f"Fase {fase}",
                'Dimensao': col_subgrupo,
                'Subgrupo': str(grp),
                'Criterio_Amostra': 'Completas (>=80%)' if apenas_comp else 'Todos os Pares Válidos',
                'N Pares': n,
                'Média Pré': round(pre.mean(), 2),
                'Média Pós': round(pos.mean(), 2),
                'Ganho Médio': round(d_medio, 2),
                's_D': round(s_d, 2),
                'd_z': round(dz, 3),
                'IC 95% Inf': round(dz - 1.96 * se_dz, 3),
                'IC 95% Sup': round(dz + 1.96 * se_dz, 3),
                'p-valor (t)': "< 0.001" if p_val < 0.001 else f"{p_val:.4f}",
            })
        
    return pd.DataFrame(registros)

efeito_sexo_v = calcular_efeito_por_subgrupo(df_v_socio, 'sexo', 'Vocabulário')
efeito_sexo_t = calcular_efeito_por_subgrupo(df_t_socio, 'sexo', 'TDE II')

tabela_sexo = pd.concat([efeito_sexo_v, efeito_sexo_t], ignore_index=True)
print("=== EFFECT SIZE POR SEXO (MENINOS VS. MENINAS) ===")
display(tabela_sexo[['Instrumento', 'Fase', 'Subgrupo', 'Criterio_Amostra', 'N Pares', 'Média Pré', 'Média Pós', 'Ganho Médio', 'd_z', 'p-valor (t)']])
""")

    nb.add_markdown("""
### Desempenho por Faixa Etária e Distorção Idade-Série
""")

    nb.add_code("""
efeito_idade_v = calcular_efeito_por_subgrupo(df_v_socio, 'faixa_etaria', 'Vocabulário')
efeito_idade_t = calcular_efeito_por_subgrupo(df_t_socio, 'faixa_etaria', 'TDE II')

tabela_idade = pd.concat([efeito_idade_v, efeito_idade_t], ignore_index=True)
print("=== EFFECT SIZE POR FAIXA ETÁRIA ===")
display(tabela_idade[tabela_idade['Criterio_Amostra'] == 'Completas (>=80%)'][['Instrumento', 'Fase', 'Subgrupo', 'N Pares', 'Média Pré', 'Média Pós', 'Ganho Médio', 'd_z']])
""")

    # -------------------------------------------------------------
    # Bloco 4: Equidade Geográfica (Rural vs Urbana) e Racial
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 4: Equidade Territorial (Zona Rural vs. Urbana) e Análise de Disparidades

Investigamos a diferença observada de pontuação entre estudantes de escolas rurais e urbanas:

$$Gap_{pré} = \\bar{Y}_{pré, \\, urbana} - \\bar{Y}_{pré, \\, rural}$$
$$Gap_{pós} = \\bar{Y}_{pós, \\, urbana} - \\bar{Y}_{pós, \\, rural}$$
$$\\Delta Gap = Gap_{pós} - Gap_{pré}$$

- $\\Delta Gap < 0$: indica redução da disparidade entre as zonas na amostra e fase analisadas.
- $\\Delta Gap > 0$: indica aumento da disparidade.

*Rigor interpretativo:* a redução observada da diferença de pontuação reflete que os estudantes da zona rural apresentaram ganhos médios superiores aos dos estudantes urbanos no período avaliado. Essa evidência descritiva não deve ser extrapolada como comprovação causal estrita de eficácia isolada do programa.
""")

    nb.add_code("""
def analisar_gap_territorial(df_socio, nome_instrumento):
    linhas = []
    
    for fase in sorted(df_socio['fase'].unique()):
        for apenas_comp in [True, False]:
            sub = df_socio[df_socio['fase'] == fase]
            if apenas_comp:
                sub = sub[sub['completo_pre'] & sub['completo_pos']]
                
            urb = sub[sub['zona'] == 'URBANA']
            rur = sub[sub['zona'] == 'RURAL']
            
            if len(urb) < 10 or len(rur) < 10:
                continue
                
            pre_urb, pos_urb = urb['total_pre'].mean(), urb['total_pos'].mean()
            pre_rur, pos_rur = rur['total_pre'].mean(), rur['total_pos'].mean()
            
            gap_pre = pre_urb - pre_rur
            gap_pos = pos_urb - pos_rur
            delta_gap = gap_pos - gap_pre
            
            ganho_urb = pos_urb - pre_urb
            ganho_rur = pos_rur - pre_rur
            
            diag = (
                'Redução da disparidade observada' if delta_gap < -0.1
                else ('Aumento da disparidade observada' if delta_gap > 0.1 else 'Disparidade estável')
            )
            
            linhas.append({
                'Instrumento': nome_instrumento,
                'Fase': f"Fase {fase}",
                'Critério_Amostra': 'Completas (>=80%)' if apenas_comp else 'Todos os Pares Válidos',
                'N Urbana': len(urb),
                'N Rural': len(rur),
                'Pré Urbana': round(pre_urb, 2),
                'Pré Rural': round(pre_rur, 2),
                'Gap Pré (U - R)': round(gap_pre, 2),
                'Pós Urbana': round(pos_urb, 2),
                'Pós Rural': round(pos_rur, 2),
                'Gap Pós (U - R)': round(gap_pos, 2),
                'Delta Gap': round(delta_gap, 2),
                'Ganho Médio Rural': round(ganho_rur, 2),
                'Ganho Médio Urbano': round(ganho_urb, 2),
                'Diagnóstico': diag,
            })
        
    return pd.DataFrame(linhas)

gap_v = analisar_gap_territorial(df_v_socio, 'Vocabulário')
gap_t = analisar_gap_territorial(df_t_socio, 'TDE II')

tabela_gap = pd.concat([gap_v, gap_t], ignore_index=True)
print("=== ANÁLISE DE EQUIDADE TERRITORIAL E FECHAMENTO DE GAPS ===")
display(tabela_gap)
""")

    nb.add_markdown("""
### Visualização da Trajetória Urbana vs. Rural
""")

    nb.add_code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (nome, df_socio, ax) in enumerate([('Vocabulário', df_v_socio, axes[0]), ('TDE II', df_t_socio, axes[1])]):
    medias_grafico = []
    for fase in [2, 4]:
        for z, cor, estilo in [('URBANA', 'royalblue', '-o'), ('RURAL', 'forestgreen', '-s')]:
            sub = df_socio[(df_socio['fase'] == fase) & (df_socio['zona'] == z) & (df_socio['completo_pre']) & (df_socio['completo_pos'])]
            ax.plot([f"F{fase} Pré", f"F{fase} Pós"], [sub['total_pre'].mean(), sub['total_pos'].mean()],
                    estilo, color=cor, linewidth=2.5, markersize=8, label=f"F{fase} {z.capitalize()}" if f"F{fase} {z.capitalize()}" not in [l.get_label() for l in ax.lines] else "")
            
    ax.set_title(f"Evolução Pré vs. Pós por Zona Residencial — {nome}", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pontuação Média")
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)

plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # Bloco 5: Cor/Raça e Necessidades Especiais
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 5: Análise por Cor / Raça e Inclusão de Alunos com Necessidades Especiais

*Atenção à categoria:* `NAO_REGISTRADO` indica que a informação não constava nos registros administrativos consultados e não deve ser confundida com ausência confirmada de necessidade especial.
""")

    nb.add_code("""
efeito_raca_v = calcular_efeito_por_subgrupo(df_v_socio, 'cor_raca', 'Vocabulário', min_n=15)
efeito_raca_t = calcular_efeito_por_subgrupo(df_t_socio, 'cor_raca', 'TDE II', min_n=15)

tabela_raca = pd.concat([efeito_raca_v, efeito_raca_t], ignore_index=True)
print("=== EFFECT SIZE POR COR / RAÇA (N >= 15, AMOSTRAS COMPLETAS) ===")
display(tabela_raca[tabela_raca['Criterio_Amostra'] == 'Completas (>=80%)'][['Instrumento', 'Fase', 'Subgrupo', 'N Pares', 'Média Pré', 'Média Pós', 'Ganho Médio', 'd_z']])
""")

    nb.add_markdown("""
### Desempenho dos Estudantes com Necessidades Educacionais Especiais
""")

    nb.add_code("""
def analisar_necessidades_especiais(df_socio, nome_instrumento):
    linhas = []
    for fase in sorted(df_socio['fase'].unique()):
        sub = df_socio[df_socio['fase'] == fase]
        for ne, grp in sub.groupby('necessidade_especial'):
            val_pre = grp['total_pre'].dropna()
            val_pos = grp['total_pos'].dropna()
            
            pares = grp.dropna(subset=['total_pre', 'total_pos'])
            ganho_m = (pares['total_pos'] - pares['total_pre']).mean() if len(pares) > 0 else np.nan
            
            linhas.append({
                'Instrumento': nome_instrumento,
                'Fase': f"Fase {fase}",
                'Necessidade Especial': ne,
                'N Total Registros': len(grp),
                'N Pares Válidos': len(pares),
                'Média Pré': round(val_pre.mean(), 2) if len(val_pre) > 0 else np.nan,
                'Média Pós': round(val_pos.mean(), 2) if len(val_pos) > 0 else np.nan,
                'Ganho Médio': round(ganho_m, 2) if pd.notna(ganho_m) else np.nan,
            })
    return pd.DataFrame(linhas)

ne_v = analisar_necessidades_especiais(df_v_socio, 'Vocabulário')
ne_t = analisar_necessidades_especiais(df_t_socio, 'TDE II')

tabela_ne = pd.concat([ne_v, ne_t], ignore_index=True)
print("=== ACOMPANHAMENTO: NECESSIDADES ESPECIAIS ===")
display(tabela_ne)
""")

    # -------------------------------------------------------------
    # Bloco 6: Consolidação e Exportação
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 6: Consolidação e Exportação dos Dados Analíticos Sociodemográficos
""")

    nb.add_code("""
# Consolidar base individual com demografia e ganhos
df_analise_socio = pd.concat([
    df_v_socio.assign(instrumento='Vocabulário'),
    df_t_socio.assign(instrumento='TDE II'),
], ignore_index=True)

# Exportar base enriquecida
out_socio_parquet = DATA_ANALYTICS / 'perfil_socioeconomico_desempenho.parquet'
out_socio_csv = DATA_ANALYTICS / 'perfil_socioeconomico_desempenho.csv'
df_analise_socio.to_parquet(out_socio_parquet, index=False)
df_analise_socio.to_csv(out_socio_csv, index=False, encoding='utf-8-sig')

# Consolidar métricas de equidade agregadas
metricas_equidade_consolidadas = pd.concat([
    tabela_sexo.assign(Tipo='Sexo'),
    tabela_idade.assign(Tipo='Faixa_Etaria'),
    tabela_gap.assign(Tipo='Gap_Territorial_Rural_Urbano'),
    tabela_raca.assign(Tipo='Cor_Raca'),
], ignore_index=True)

out_eq_parquet = DATA_ANALYTICS / 'metricas_equidade.parquet'
out_eq_csv = DATA_ANALYTICS / 'metricas_equidade.csv'
metricas_equidade_consolidadas.to_parquet(out_eq_parquet, index=False)
metricas_equidade_consolidadas.to_csv(out_eq_csv, index=False, encoding='utf-8-sig')

print("Métricas sociodemográficas exportadas com sucesso:")
print(f"  -> {out_socio_parquet} ({len(df_analise_socio):,} registros)")
print(f"  -> {out_socio_csv}")
print(f"  -> {out_eq_parquet} ({len(metricas_equidade_consolidadas):,} métricas)")
print(f"  -> {out_eq_csv}")
""")

    # Salva notebook
    out_nb = Path(__file__).resolve().parents[1] / "notebooks" / "04_perfil_socioeconomico.ipynb"
    nb.save(out_nb)


if __name__ == "__main__":
    construir_notebook_04()
