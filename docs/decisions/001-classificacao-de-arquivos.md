# 001 — Classificação de arquivos: o conteúdo define, o caminho confirma

## Contexto
Os CSVs brutos estão em pastas com nomes inconsistentes (maiúsculas, acentos,
variações de escrita). Ao conferir os cabeçalhos, encontramos um arquivo de TDE
guardado na pasta de vocabulário e seis arquivos nessa pasta sem colunas de item.
Classificar só pelo nome da pasta superestimaria o vocabulário (98 contra 91).

## Decisão
O instrumento é definido pelas colunas de item do cabeçalho: `Q1..` é
vocabulário, `P1..` é TDE. O caminho só confirma. Quando os dois discordam, ou
quando o cabeçalho mistura Q e P, o arquivo fica como `outro` com status
`divergente` e vai para revisão. A linha do cabeçalho é detectada (não é sempre
a primeira).

## Consequências
- Nenhum arquivo é descartado: todos entram no catálogo com status explícito.
- O momento da Fase 4 continua `desconhecido` até haver evidência nos controles.
- Arquivos com colunas de item repetidas são sinalizados para tratamento no M2.
