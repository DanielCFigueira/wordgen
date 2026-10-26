import pandas as pd
import numpy as np

print('='*60)
print('  AUDITORIA DOS DATASETS GERADOS')
print('='*60)

# VOCABULARIO
df_v = pd.read_csv('data/processed/dataset_vocabulario.csv', low_memory=False)
print('\n[VOCABULARIO]')
print(f'  Shape: {df_v.shape}')
print(f'  Alunos unicos: {df_v["aluno_id"].nunique()}')
print(f'  Escolas: {df_v["escola"].nunique()}')
print(f'  Turmas: {df_v["turma"].nunique()}')
print('\n  Distribuicao fase x periodo:')
print(df_v.groupby(['fase','periodo']).size().unstack(fill_value=0))
print('\n  Dados socioeconomicos preenchidos:')
for col in ['sexo','cor_raca','zona','data_nascimento','idade']:
    if col in df_v.columns:
        pct = df_v[col].notna().mean()*100
        print(f'    {col}: {pct:.1f}%')

q_cols = [c for c in df_v.columns if c.startswith('Q')]
print(f'\n  Colunas de questao: {len(q_cols)} (Q01..Q50)')
print(f'  Total vocabulario - media: {df_v["total_vocabulario"].mean():.1f}, mediana: {df_v["total_vocabulario"].median():.1f}')
print(f'  Total vocabulario - NaN: {df_v["total_vocabulario"].isna().sum()}')

# TDE
df_t = pd.read_csv('data/processed/dataset_tde.csv', low_memory=False)
print('\n' + '-'*60)
print('[TDE]')
print(f'  Shape: {df_t.shape}')
print(f'  Alunos unicos: {df_t["aluno_id"].nunique()}')
print(f'  Escolas: {df_t["escola"].nunique()}')
print(f'  Turmas: {df_t["turma"].nunique()}')
print('\n  Distribuicao fase x periodo:')
print(df_t.groupby(['fase','periodo']).size().unstack(fill_value=0))
print('\n  Dados socioeconomicos preenchidos:')
for col in ['sexo','cor_raca','zona','data_nascimento','idade']:
    if col in df_t.columns:
        pct = df_t[col].notna().mean()*100
        print(f'    {col}: {pct:.1f}%')

p_cols = [c for c in df_t.columns if c.startswith('P')]
print(f'\n  Colunas de questao: {len(p_cols)} (P01..P40)')
print(f'  Total TDE - media: {df_t["total_tde"].mean():.1f}, mediana: {df_t["total_tde"].median():.1f}')
print(f'  Total TDE - NaN: {df_t["total_tde"].isna().sum()}')

# SOCIOECONOMICO
df_s = pd.read_csv('data/processed/dataset_socioeconomico.csv', low_memory=False)
print('\n' + '-'*60)
print('[SOCIOECONOMICO]')
print(f'  Shape: {df_s.shape}')
print(f'  Alunos unicos: {df_s["aluno_id"].nunique()}')
for col in df_s.columns:
    if col not in ['aluno_id','escola','turma','fase','ano_semestre']:
        pct = df_s[col].notna().mean()*100
        print(f'    {col}: {pct:.1f}%')

# ANALISE LONGITUDINAL
print('\n' + '-'*60)
print('[ANALISE LONGITUDINAL - Viabilidade]')
for ds_name, ds in [('Vocabulario', df_v), ('TDE', df_t)]:
    for fase in [2,3,4]:
        pre = set(ds[(ds['fase']==fase)&(ds['periodo']=='pre')]['aluno_id'])
        pos = set(ds[(ds['fase']==fase)&(ds['periodo']=='pos')]['aluno_id'])
        ambos = pre & pos
        print(f'  {ds_name} Fase {fase}: {len(pre)} pre, {len(pos)} pos, {len(ambos)} com ambos (pre+pos)')

print()
for ds_name, ds in [('Vocabulario', df_v), ('TDE', df_t)]:
    fases_por_aluno = ds.groupby('aluno_id')['fase'].nunique()
    print(f'  {ds_name} - alunos em 1 fase: {(fases_por_aluno==1).sum()}, 2 fases: {(fases_por_aluno==2).sum()}, 3 fases: {(fases_por_aluno==3).sum()}')

# ESCOLAS
print('\n' + '-'*60)
print('[ESCOLAS]')
print(df_v['escola'].value_counts().to_string())
