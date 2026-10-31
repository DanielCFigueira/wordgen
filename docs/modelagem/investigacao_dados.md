# Investigação e Seleção dos Dados — WordGen

## 1. Identificação das Bases Disponíveis

A auditoria dos dados para a etapa de modelagem preditiva examinou as bases consolidadas produzidas nas fases anteriores de Engenharia de Dados:

| Base de Dados | Formato | Dimensões Atuais | Finalidade no Projeto |
| :--- | :--- | :--- | :--- |
| `pareado_vocabulario` | Parquet / CSV | 3.418 × 11 | Avaliações pré e pós-teste pareadas de Vocabulário |
| `pareado_tde` | Parquet / CSV | 3.948 × 11 | Avaliações pré e pós-teste pareadas de TDE II (Escrita) |
| `dim_aluno` | Parquet / CSV | 2.530 × 7 | Dimensão socioeconômica e demográfica consolidada |
| `dataset_vocabulario` | Parquet / CSV | 8.826 × 76 | Base analítica unificada de Vocabulário por aplicação |
| `dataset_tde` | Parquet / CSV | 9.317 × 66 | Base analítica unificada de TDE II por aplicação |

---

## 2. Investigação da Estrutura dos Dados

A análise das variáveis primárias confirmou as seguintes estruturas:
- **Vocabulário**: escala de 0 a 50 pontos (50 itens binários Q01 a Q50). Nos dados reais pareados, as pontuações observadas variam de 0,0 a 48,0 no pré-teste e de 0,0 a 47,0 no pós-teste, respeitando integralmente os limites teóricos.
- **TDE II**: escala de 0 a 80 pontos (40 itens politômicos P01 a P40 pontuados em 0, 1 ou 2). Nos dados reais pareados, as pontuações observadas variam de 0,0 a 78,0 no pré-teste e de 0,0 a 80,0 no pós-teste, também respeitando rigorosamente os limites teóricos.

---

## 3. Investigação dos Registros Pareados e Completude

Uma distinção metodológica crucial verificada na auditoria é que **um registro pareado não equivale automaticamente a uma avaliação completa**:

### 3.1. Vocabulário Pareado
- **Registros pareados totais**: 3.418 linhas (2.028 estudantes únicos).
- **Pré-teste completo (`completo_pre == True`)**: 3.364 registros.
- **Pós-teste completo (`completo_pos == True`)**: 3.367 registros.
- **Pares completos simultâneos (`completo_pre & completo_pos`)**: **3.320 registros** (1.989 estudantes únicos).
- **Registros com preenchimento parcial (< 80% itens)**: 98 linhas descartadas para modelagem estrita.

Distribuição dos pares completos por fase:
- Fase 2: 1.352 pares
- Fase 3: 872 pares
- Fase 4: 1.096 pares

### 3.2. TDE II Pareado
- **Registros pareados totais**: 3.948 linhas (2.140 estudantes únicos).
- **Pré-teste completo (`completo_pre == True`)**: 3.826 registros.
- **Pós-teste completo (`completo_pos == True`)**: 3.853 registros.
- **Pares completos simultâneos (`completo_pre & completo_pos`)**: **3.796 registros** (2.069 estudantes únicos).
- **Registros com preenchimento parcial (< 80% itens)**: 152 linhas descartadas para modelagem estrita.

Distribuição dos pares completos por fase:
- Fase 2: 1.245 pares
- Fase 3: 1.545 pares
- Fase 4: 1.006 pares

---

## 4. Verificação de Duplicidades e Integridade de Chaves

1. **Bases Brutas de Aplicação**:
   - `dataset_vocabulario`: 0 duplicidades na chave `aluno_id + fase + periodo`.
   - `dataset_tde`: 0 duplicidades na chave `aluno_id + fase + periodo`.
2. **Bases Pareadas**:
   - `pareado_vocabulario`: 0 duplicidades na chave `aluno_id + fase`.
   - `pareado_tde`: 0 duplicidades na chave `aluno_id + fase`.
3. **Valores Ausentes nos Totais**:
   - `total_pre` e `total_pos`: zero valores ausentes em ambas as bases pareadas.

---

## 5. Análise de Cobertura e Significado de `dim_aluno`

A dimensão `dim_aluno` possui 2.530 registros com as seguintes características:
- **Cobertura nos Pares Completos de Vocabulário**: 3.289 de 3.320 linhas (99,07% de cobertura). Apenas 30 estudantes únicos não constam no cadastro demográfico.
- **Cobertura nos Pares Completos de TDE II**: 3.774 de 3.796 linhas (99,42% de cobertura). Apenas 22 estudantes únicos não constam no cadastro demográfico.

### Classificação e Natureza das Variáveis
- `sexo`: Demográfico (1.278 Masculino, 1.252 Feminino). 100% preenchido.
- `faixa_etaria`: Demográfico (13-14: 1.062; 11-12: 870; 15+: 475; <=10: 100; NAO INFORMADA: 23).
- `cor_raca`: Demográfico (PARDA: 1.803; NAO INFORMADO: 403; BRANCA: 244; INDÍGENA: 66; PRETA: 10; AMARELA: 4).
- `zona`: Territorial/Contextual (URBANA: 1.286; RURAL: 933; NAO INFORMADA: 311).
- `necessidade_especial`: Educacional/Inclusão (NAO_REGISTRADO: 2.464; SIM: 66).
  > **Nota Técnica:** `NAO_REGISTRADO` reflete ausência de laudo cadastral nos registros municipais, não confirmando ausência médica.
- `fonte_socio`: Metadado técnico de engenharia de dados (indica a pasta/arquivo de origem). **Excluída das features de modelagem**.
- `responsavel_legal`: Dado pessoal identificável (PII). **Estritamente excluído por privacidade e ética**.

---

## 6. Seleção dos Dados para Modelagem

### 6.1. Registros Elegíveis
Apenas os pares com pré e pós-teste completos e pontuações válidas:
- **Vocabulário**: 3.320 observações.
- **TDE II**: 3.796 observações.

### 6.2. Variável-Alvo (Target)
$$\text{ganho} = \text{total\_pos} - \text{total\_pre}$$
- **Vocabulário**: média = $+0{,}63$ pontos, desvio padrão = $5{,}98$, mín = $-29{,}0$, máx = $+29{,}0$.
- **TDE II**: média = $+0{,}04$ pontos, desvio padrão = $8{,}77$, mín = $-53{,}0$, máx = $+63{,}0$.

---

## 7. Conclusão da Investigação

| Decisão | Justificativa Metodológica |
| :--- | :--- |
| **Bases Primárias** | `pareado_vocabulario`, `pareado_tde` e `dim_aluno` |
| **Filtro de Amostra** | Apenas pares completos (`completo_pre & completo_pos`) para evitar ruído de testes parciais |
| **Tratamento de Ausências Demográficas** | Categoria explícita `NAO_INFORMADO` para não descartar estudantes sem cadastro demográfico |
| **Prevenção de Vazamento** | Exclusão de qualquer informação futura ou derivada do pós-teste |
