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
1. **Consolidação de KPIs Mestres:** definir os números-chave executivos do projeto, distinguindo com precisão estudantes únicos (pessoas), registros de testes (aplicações), pares válidos e pares completos.
2. **Auditoria Rigorosa do KPI de Turmas:** normalizar nomes de escolas, séries e letras de turma, excluindo registros incompletos e documentando o total de turmas válidas em escolas municipais identificadas (84 turmas) e o total censitário incluindo registros com escola não informada (92 turmas).
3. **Cubo Analítico Multidimensional:** pré-computar métricas descritivas e effect sizes ($d_z$) fatiados por todas as dimensões de interesse do Streamlit (Fase, Escola, Turma, Série, Sexo, Zona, Faixa Etária, Cor/Raça e Necessidade Especial).
4. **Ranqueamento e Benchmark de Escolas:** estruturar tabelas prontas para comparação institucional com ordenação por effect size e ganho médio.
5. **Exportação Otimizada:** salvar datasets em formato Parquet e CSV na pasta `data/analytics/`.
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

# Junção com perfil sociodemográfico preservando alunos sem correspondência
df_v_full = df_pv.merge(df_aluno, on='aluno_id', how='left')
df_t_full = df_pt.merge(df_aluno, on='aluno_id', how='left')

# Preenchimento de categorias nulas para filtros
cols_socio = ['sexo', 'faixa_etaria', 'cor_raca', 'zona', 'necessidade_especial']
for col in cols_socio:
    df_v_full[col] = df_v_full[col].fillna('NAO INFORMADA' if col != 'necessidade_especial' else 'NAO_REGISTRADO')
    df_t_full[col] = df_t_full[col].fillna('NAO INFORMADA' if col != 'necessidade_especial' else 'NAO_REGISTRADO')

# Ganhos individuais
df_v_full['ganho'] = df_v_full['total_pos'] - df_v_full['total_pre']
df_t_full['ganho'] = df_t_full['total_pos'] - df_t_full['total_pre']

print("Bases carregadas com integridade garantida.")
""")

    # -------------------------------------------------------------
    # Bloco 2: Auditoria e Cálculo do KPI de Turmas
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 2: Auditoria e Cálculo Rigoroso do KPI de Turmas

Para evitar a contagem espúria de valores nulos (onde `NaN != NaN` gerava duplicatas artificiais no código anterior) ou de registros sem identificação escolar completa:
1. Normalizamos escola (maiúsculas, sem espaços laterais), série (inteiro de 6 a 9) e letra da turma (A, B, C, etc.).
2. Excluímos linhas com componentes ausentes e auditamos exatamente quantas linhas foram descartadas e por qual motivo.
3. Calculamos turmas válidas em escolas municipais identificadas e turmas totais incluindo registros com escola não informada.
""")

    nb.add_code("""
def auditar_e_extrair_turmas(df, nome_instrumento):
    df_c = df.copy()
    
    sem_esc = df_c['escola'].isna() | (df_c['escola'] == 'NAO INFORMADA')
    sem_ser = df_c['serie'].isna()
    sem_tur = df_c['turma_letra'].isna() | (df_c['turma_letra'] == 'NAO INFORMADA')
    
    incompletos = sem_esc | sem_ser | sem_tur
    
    # Turmas estritamente válidas em escolas identificadas
    validos = df_c[~incompletos].copy()
    validos['escola_norm'] = validos['escola'].astype(str).str.strip().str.upper()
    validos['serie_norm'] = validos['serie'].astype(int)
    validos['letra_norm'] = validos['turma_letra'].astype(str).str.strip().str.upper()
    
    turmas_identificadas = set(zip(validos['escola_norm'], validos['serie_norm'], validos['letra_norm']))
    
    # Turmas incluindo escola não informada (mas com série e letra válidas)
    validos_com_na = df_c[df_c['serie'].notna() & df_c['turma_letra'].notna() & (df_c['turma_letra'] != 'NAO INFORMADA')].copy()
    validos_com_na['escola_norm'] = validos_com_na['escola'].astype(str).str.strip().str.upper()
    validos_com_na['serie_norm'] = validos_com_na['serie'].astype(int)
    validos_com_na['letra_norm'] = validos_com_na['turma_letra'].astype(str).str.strip().str.upper()
    turmas_totais = set(zip(validos_com_na['escola_norm'], validos_com_na['serie_norm'], validos_com_na['letra_norm']))
    
    resumo_auditoria = {
        'Instrumento': nome_instrumento,
        'Total Linhas Avaliações': len(df_c),
        'Linhas sem Escola Válida': int(sem_esc.sum()),
        'Linhas sem Série Válida': int(sem_ser.sum()),
        'Linhas sem Turma Válida': int(sem_tur.sum()),
        'Linhas Incompletas Excluídas': int(incompletos.sum()),
        'Turmas em Escolas Identificadas': len(turmas_identificadas),
        'Turmas Totais (incl. Não Informada)': len(turmas_totais),
    }
    
    return turmas_identificadas, turmas_totais, pd.Series(resumo_auditoria)

turmas_v_id, turmas_v_tot, aud_v = auditar_e_extrair_turmas(df_v_raw, 'Vocabulário')
turmas_t_id, turmas_t_tot, aud_t = auditar_e_extrair_turmas(df_t_raw, 'TDE II')

tab_auditoria_turmas = pd.DataFrame([aud_v, aud_t])
print("=== AUDITORIA DE LINHAS E TURMAS POR INSTRUMENTO ===")
display(tab_auditoria_turmas)

# União dos instrumentos
uniao_turmas_identificadas = turmas_v_id.union(turmas_t_id)
uniao_turmas_totais = turmas_v_tot.union(turmas_t_tot)

n_turmas_identificadas = len(uniao_turmas_identificadas)
n_turmas_totais = len(uniao_turmas_totais)

print()
print("=== CONSOLIDAÇÃO FINAL DE TURMAS ===")
print(f"Turmas válidas nas 6 escolas municipais identificadas: {n_turmas_identificadas} turmas (Vocab: {len(turmas_v_id)}, TDE: {len(turmas_t_id)}, União: {n_turmas_identificadas})")
print(f"Turmas totais registradas (incluindo escola não informada): {n_turmas_totais} turmas (Vocab: {len(turmas_v_tot)}, TDE: {len(turmas_t_tot)}, União: {n_turmas_totais})")
""")

    # -------------------------------------------------------------
    # Bloco 3: KPIs Mestres do Projeto
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 3: Consolidação dos KPIs Mestres (Cards Executivos)

Apresentamos os indicadores centrais do projeto com unidades de observação rigorosamente documentadas (pessoas vs. linhas de testes).
""")

    nb.add_code("""
# 1. Estudantes únicos
ids_v = set(df_v_raw['aluno_id'])
ids_t = set(df_t_raw['aluno_id'])
ids_uniao = ids_v.union(ids_t)
n_alunos_uniao = len(ids_uniao)
n_alunos_v = len(ids_v)
n_alunos_t = len(ids_t)

ids_socio = set(df_aluno['aluno_id'])
n_alunos_socio = len(ids_socio)
n_alunos_sem_socio = len(ids_uniao - ids_socio)
pct_cobertura_socio = round(100 * len(ids_uniao.intersection(ids_socio)) / n_alunos_uniao, 2)

# 2. Escolas
escolas_ativas = [e for e in df_v_raw['escola'].dropna().unique() if e != 'NAO INFORMADA']
n_escolas = len(escolas_ativas)

# 3. Testes aplicados
total_testes = len(df_v_raw) + len(df_t_raw)

# 4. Linhas de pares válidos e completos
n_linhas_pares_validos = len(df_pv) + len(df_pt)
comp_v = df_pv['completo_pre'] & df_pv['completo_pos']
comp_t = df_pt['completo_pre'] & df_pt['completo_pos']
n_linhas_pares_completos = int(comp_v.sum() + comp_t.sum())
pct_conclusao_pares = round(100 * n_linhas_pares_completos / n_linhas_pares_validos, 1)

# 5. Pessoas com par completo
alunos_comp_v = set(df_pv[comp_v]['aluno_id'])
alunos_comp_t = set(df_pt[comp_t]['aluno_id'])
n_pessoas_par_completo = len(alunos_comp_v.union(alunos_comp_t))

# Tabela consolidada de KPIs
kpis_mestres = pd.DataFrame([
    {'KPI': 'Estudantes Únicos (União Vocabulário e TDE)', 'Valor_Numerico': n_alunos_uniao, 'Valor_Formatado': f"{n_alunos_uniao:,}", 'Unidade': 'Pessoas', 'Descricao': 'Estudantes com registro em pelo menos um dos dois instrumentos'},
    {'KPI': 'Estudantes Únicos em Vocabulário', 'Valor_Numerico': n_alunos_v, 'Valor_Formatado': f"{n_alunos_v:,}", 'Unidade': 'Pessoas', 'Descricao': 'Estudantes únicos com registro na base de Vocabulário'},
    {'KPI': 'Estudantes Únicos em TDE II', 'Valor_Numerico': n_alunos_t, 'Valor_Formatado': f"{n_alunos_t:,}", 'Unidade': 'Pessoas', 'Descricao': 'Estudantes únicos com registro na base de TDE II'},
    {'KPI': 'Estudantes com Cadastro Socioeconômico', 'Valor_Numerico': n_alunos_socio, 'Valor_Formatado': f"{n_alunos_socio:,}", 'Unidade': 'Pessoas', 'Descricao': 'Estudantes únicos cadastrados na dimensão dim_aluno'},
    {'KPI': 'Estudantes sem Cadastro Socioeconômico', 'Valor_Numerico': n_alunos_sem_socio, 'Valor_Formatado': f"{n_alunos_sem_socio:,}", 'Unidade': 'Pessoas', 'Descricao': 'Estudantes nos instrumentos sem correspondência socioeconômica'},
    {'KPI': 'Cobertura Socioeconômica', 'Valor_Numerico': pct_cobertura_socio, 'Valor_Formatado': f"{pct_cobertura_socio}%", 'Unidade': 'Percentual', 'Descricao': 'Percentual de estudantes avaliados com cadastro socioeconômico'},
    {'KPI': 'Escolas Públicas Atendidas', 'Valor_Numerico': n_escolas, 'Valor_Formatado': f"{n_escolas}", 'Unidade': 'Instituições', 'Descricao': 'Escolas municipais participantes da intervenção WordGen'},
    {'KPI': 'Turmas em Escolas Identificadas', 'Valor_Numerico': n_turmas_identificadas, 'Valor_Formatado': f"{n_turmas_identificadas}", 'Unidade': 'Turmas', 'Descricao': 'Turmas válidas (escola + série + turma) nas 6 escolas municipais'},
    {'KPI': 'Turmas Totais Registradas (incl. Não Informada)', 'Valor_Numerico': n_turmas_totais, 'Valor_Formatado': f"{n_turmas_totais}", 'Unidade': 'Turmas', 'Descricao': 'Total de turmas com série e letra válidas no censo geral'},
    {'KPI': 'Total de Registros de Testes Aplicados', 'Valor_Numerico': total_testes, 'Valor_Formatado': f"{total_testes:,}", 'Unidade': 'Aplicações', 'Descricao': 'Soma de aplicações de testes de Vocabulário e TDE II'},
    {'KPI': 'Linhas de Avaliações Pareadas Válidas', 'Valor_Numerico': n_linhas_pares_validos, 'Valor_Formatado': f"{n_linhas_pares_validos:,}", 'Unidade': 'Linhas de Pares', 'Descricao': 'Total de pares pré/pós válidos em ambos os instrumentos'},
    {'KPI': 'Linhas de Avaliações Pareadas Completas', 'Valor_Numerico': n_linhas_pares_completos, 'Valor_Formatado': f"{n_linhas_pares_completos:,}", 'Unidade': 'Linhas de Pares', 'Descricao': 'Pares com pré e pós completos (>= 80% de itens válidos)'},
    {'KPI': 'Estudantes Únicos com Par Completo', 'Valor_Numerico': n_pessoas_par_completo, 'Valor_Formatado': f"{n_pessoas_par_completo:,}", 'Unidade': 'Pessoas', 'Descricao': 'Estudantes únicos com pelo menos um par pré/pós completo'},
    {'KPI': 'Taxa de Completude Pareada', 'Valor_Numerico': pct_conclusao_pares, 'Valor_Formatado': f"{pct_conclusao_pares}%", 'Unidade': 'Percentual', 'Descricao': 'Proporção de pares completos em relação às linhas pareadas válidas'},
])

print("=== CARDS EXECUTIVOS: KPIS MESTRES DO WORDGEN ===")
display(kpis_mestres[['KPI', 'Valor_Formatado', 'Unidade', 'Descricao']])
""")

    # -------------------------------------------------------------
    # Bloco 4: Cubo Analítico Multidimensional
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 4: Construção do Cubo Analítico Fatiável para o Dashboard

Pré-computamos métricas descritivas e effect sizes ($d_z$) fatiados por:
- **Geral**
- **Escola**
- **Turma** (combinação normalizada de escola + série + letra)
- **Série** (6º, 7º, 8º, 9º Ano)
- **Sexo** (Masculino, Feminino, Não Informada)
- **Zona** (Urbana, Rural, Não Informada)
- **Faixa Etária** (<=10, 11-12, 13-14, 15+, Não Informada)
- **Cor / Raça** (Parda, Branca, Indígena, Preta, Amarela, Não Informado)
- **Necessidade Especial** (NAO_REGISTRADO, SIM)
""")

    nb.add_code("""
# Criação do identificador de turma normalizado nas bases pareadas
def atribuir_turma_normalizada(df):
    esc = df['escola'].fillna('NAO INFORMADA').astype(str).str.strip().str.upper()
    ser = df['serie'].fillna(0).astype(int).astype(str) + 'º'
    tur = df['turma'].fillna('U').astype(str).str.strip().str.upper()
    df['turma_id'] = esc + ' - ' + ser + ' ' + tur
    return df

df_v_full = atribuir_turma_normalizada(df_v_full)
df_t_full = atribuir_turma_normalizada(df_t_full)

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
    
    dz = ganho_m / s_d if (s_d is not None and s_d > 0 and not np.isnan(s_d)) else 0.0
    se_dz = np.sqrt(1.0/n_comp + (dz**2)/(2.0*n_comp)) if n_comp > 0 else 0.0
    
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
        'Desvio_Padrao_Ganho': round(s_d, 2) if pd.notna(s_d) else 0.0,
        'Effect_Size_dz': round(dz, 3),
        'IC_95_Inf': round(dz - 1.96 * se_dz, 3),
        'IC_95_Sup': round(dz + 1.96 * se_dz, 3),
        'Pct_Alunos_Ganho': round(100 * n_pos_ganho / n_comp, 1),
        'Pct_Alunos_Inalterado': round(100 * n_zero_ganho / n_comp, 1),
        'Pct_Alunos_Queda': round(100 * n_neg_ganho / n_comp, 1),
    }

def construir_cubo_analitico():
    linhas_cubo = []
    
    dimensoes = [
        ('Escola', 'escola'),
        ('Turma', 'turma_id'),
        ('Serie', 'serie'),
        ('Sexo', 'sexo'),
        ('Zona', 'zona'),
        ('Faixa_Etaria', 'faixa_etaria'),
        ('Cor_Raca', 'cor_raca'),
        ('Necessidade_Especial', 'necessidade_especial'),
    ]
    
    for nome_inst, df_full in [('Vocabulário', df_v_full), ('TDE II', df_t_full)]:
        fases = ['Geral'] + [f"Fase {f}" for f in sorted(df_full['fase'].unique())]
        
        for f_label in fases:
            sub_fase = df_full if f_label == 'Geral' else df_full[df_full['fase'] == int(f_label.split()[-1])]
            
            # 1. Corte Geral
            r = gerar_linha_cubo(sub_fase, nome_inst, f_label, 'Geral', 'Todos')
            if r: linhas_cubo.append(r)
            
            # 2. Demais Dimensões
            for nome_dim, col in dimensoes:
                valores = sub_fase[col].dropna().unique()
                for val in valores:
                    # Rótulos legíveis
                    val_str = str(val)
                    if nome_dim == 'Serie' and pd.notna(val):
                        val_str = f"{int(float(val))}º Ano"
                    elif nome_dim == 'Sexo':
                        val_str = 'Masculino' if val == 'M' else ('Feminino' if val == 'F' else str(val))
                    elif nome_dim == 'Zona':
                        val_str = str(val).capitalize()
                        
                    r = gerar_linha_cubo(sub_fase[sub_fase[col] == val], nome_inst, f_label, nome_dim, val_str)
                    if r: linhas_cubo.append(r)
                
    return pd.DataFrame(linhas_cubo)

df_cubo = construir_cubo_analitico()
print(f"Cubo Analítico gerado com {len(df_cubo):,} fatias pré-computadas!")
print("Dimensões indexadas:", df_cubo['Dimensao'].unique().tolist())
display(df_cubo.head(10))
""")

    # -------------------------------------------------------------
    # Bloco 5: Ranqueamento de Escolas
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 5: Tabela de Ranqueamento e Comparativo de Escolas

Estruturamos a comparação institucional ordenada por Effect Size ($d_z$) e volume amostral ($N \\ge 15$).
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
    # Bloco 6: Exportação dos Dados do Dashboard
    # -------------------------------------------------------------
    nb.add_markdown("""
## Bloco 6: Exportação dos Datasets Estruturados para o Streamlit
""")

    nb.add_code("""
# 1. KPIs de Topo
kpis_mestres.to_parquet(DATA_ANALYTICS / 'dashboard_kpis.parquet', index=False)
kpis_mestres.to_csv(DATA_ANALYTICS / 'dashboard_kpis.csv', index=False, encoding='utf-8-sig')

# 2. Cubo de Filtros Multidimensional
df_cubo.to_parquet(DATA_ANALYTICS / 'dashboard_cubo_filtros.parquet', index=False)
df_cubo.to_csv(DATA_ANALYTICS / 'dashboard_cubo_filtros.csv', index=False, encoding='utf-8-sig')

# 3. Ranking de Escolas
ranking_escolas.to_parquet(DATA_ANALYTICS / 'dashboard_escolas_ranking.parquet', index=False)
ranking_escolas.to_csv(DATA_ANALYTICS / 'dashboard_escolas_ranking.csv', index=False, encoding='utf-8-sig')

print("Datasets analíticos do Dashboard exportados com sucesso em data/analytics/:")
print(f"  -> dashboard_kpis.parquet ({len(kpis_mestres):,} KPIs)")
print(f"  -> dashboard_cubo_filtros.parquet ({len(df_cubo):,} registros de filtros)")
print(f"  -> dashboard_escolas_ranking.parquet ({len(ranking_escolas):,} registros)")
""")

    # Salva notebook
    out_nb = Path(__file__).resolve().parents[1] / "notebooks" / "05_indicadores_dashboard.ipynb"
    nb.save(out_nb)


if __name__ == "__main__":
    construir_notebook_05()
