"""Re-treinamento dos modelos MLP com early_stopping=False (correção metodológica).

Contexto:
    A configuração anterior usava early_stopping=True no MLPRegressor, que cria
    internamente uma divisão aleatória do conjunto de treino sem respeitar o
    agrupamento por aluno_id. Isso conflita com a estratégia de validação agrupada
    do projeto (GroupShuffleSplit + GroupKFold). A correção define early_stopping=False
    e aumenta max_iter para 1000, mantendo a regularização por alpha (L2).

Modelos afetados:
    vocabulario_exp_a_mlp, vocabulario_exp_b_mlp, tde_exp_a_mlp, tde_exp_b_mlp

Modelos preservados (não afetados):
    vocabulario_exp_a_rf, vocabulario_exp_b_rf, tde_exp_a_rf, tde_exp_b_rf
"""

import os
from pathlib import Path

import matplotlib

os.environ["MPLCONFIGDIR"] = str(Path("models/modelagem/.mplcache").resolve())
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wordgen.modelagem.avaliacao import avaliar_modelo
from wordgen.modelagem.interpretacao import calcular_permutation_importance
from wordgen.modelagem.persistencia import carregar_artefato_modelo, salvar_artefato_modelo
from wordgen.modelagem.preparacao import carregar_e_preparar_dados
from wordgen.modelagem.treinamento import treinar_e_otimizar_mlp
from wordgen.modelagem.validacao import dividir_treino_teste_agrupado

ROOT = Path(".")
MODELS_DIR = ROOT / "models" / "modelagem"
GRAF_DIR = ROOT / "reports" / "modelagem" / "graficos"
GRAF_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("RE-TREINAMENTO MLP — Correção: early_stopping=False, max_iter=1000")
print("=" * 70)

# 1. Carregar datasets
print("\n[1] Carregando bases analíticas...")
datasets = carregar_e_preparar_dados("data")

cenarios_mlp = [
    ("Vocabulário Exp A", "vocabulario_exp_a", "Vocabulário", "A"),
    ("Vocabulário Exp B", "vocabulario_exp_b", "Vocabulário", "B"),
    ("TDE II Exp A", "tde_exp_a", "TDE II", "A"),
    ("TDE II Exp B", "tde_exp_b", "TDE II", "B"),
]

resultados_resumo = []
modelos_mlp = {}
splits_mlp = {}

for nome_cenario, chave, instrumento, exp in cenarios_mlp:
    print(f"\n[MLP] {nome_cenario}")
    df = datasets[chave]

    # Divisão agrupada
    X_train, X_test, y_train, y_test, g_train, _g_test = dividir_treino_teste_agrupado(df)
    splits_mlp[chave] = {"X_test": X_test, "y_test": y_test}

    # Features por tipo
    cols_numericas = [
        c for c in X_train.columns if X_train[c].dtype in [np.float64, np.int64, float, int]
    ]
    cols_categoricas = [c for c in X_train.columns if c not in cols_numericas]

    # Treinar MLP com early_stopping=False (via criar_pipeline_mlp corrigido)
    mlp_pipeline, info_treino = treinar_e_otimizar_mlp(
        X_train=X_train,
        y_train=y_train,
        groups_train=g_train,
        colunas_numericas=cols_numericas,
        colunas_categoricas=cols_categoricas,
        cv_splits=5,
        random_state=42,
    )

    # Verificar que early_stopping está False no melhor estimador
    mlp_model = mlp_pipeline.named_steps["regressor"]
    assert mlp_model.early_stopping is False, "ERRO: early_stopping ainda é True!"
    n_iter = int(mlp_model.n_iter_)
    print("  ✓ early_stopping=False confirmado")
    print(f"  Melhores hiperparâmetros: {info_treino['melhores_parametros']}")
    print(f"  MAE CV: {info_treino['cv_mae_medio']:.4f}")

    # Avaliar no conjunto de teste usando avaliar_modelo(modelo, X_test, y_test, ...)
    diag = avaliar_modelo(
        mlp_pipeline, X_test, y_test, nome_modelo="Rede Neural (MLP)", cenario=nome_cenario
    )
    resultados_resumo.append(
        {
            "Cenario": nome_cenario,
            "Modelo": "Rede Neural (MLP)",
            "MAE": diag["MAE"],
            "RMSE": diag["RMSE"],
            "R2": diag["R2"],
            "media_real": diag["media_real"],
            "media_pred": diag["media_pred"],
            "chave_dataset": chave,
        }
    )

    # Salvar artefato MLP atualizado
    nome_artefato = f"{chave}_mlp"
    metadados = {
        "instrumento": instrumento,
        "experimento": exp,
        "algoritmo": "MLPRegressor",
        "features_numericas": cols_numericas,
        "features_categoricas": cols_categoricas,
        "melhores_parametros": info_treino["melhores_parametros"],
        "cv_mae_medio": float(info_treino["cv_mae_medio"]),
        "n_treino": int(len(X_train)),
        "n_teste": int(len(X_test)),
        "MAE_teste": float(diag["MAE"]),
        "RMSE_teste": float(diag["RMSE"]),
        "R2_teste": float(diag["R2"]),
        "n_iter": n_iter,
        "max_iter": int(mlp_model.max_iter),
        "atingiu_limite_iteracoes": n_iter >= int(mlp_model.max_iter),
        "nota_metodologica": (
            "early_stopping=False: divisão interna do MLPRegressor desabilitada para "
            "manter consistência com validação agrupada por aluno_id (GroupKFold externo). "
            "max_iter=1000 garante convergência. Regularização via alpha (L2)."
        ),
    }
    salvar_artefato_modelo(mlp_pipeline, nome_artefato, metadados, models_dir=MODELS_DIR)
    modelos_mlp[chave] = mlp_pipeline
    print(f"  MAE={diag['MAE']:.4f} | RMSE={diag['RMSE']:.4f} | R²={diag['R2']:.4f}")

# Atualiza somente as linhas MLP dos relatórios existentes; baselines e RF ficam preservados.
metricas_dir = ROOT / "reports" / "modelagem" / "metricas"
metricas_path = metricas_dir / "metricas_comparativas_gerais.csv"
metricas = pd.read_csv(metricas_path)
metricas["Cenário"] = metricas["Cenário"].str.replace(
    "Acadêmico + Demográfico)", "Acadêmico + Demográfico/Contextual)", regex=False
)
rotulos_cenario = {
    "vocabulario_exp_a": "Vocabulário — Experimento A (Acadêmico)",
    "vocabulario_exp_b": "Vocabulário — Experimento B (Acadêmico + Demográfico/Contextual)",
    "tde_exp_a": "TDE II — Experimento A (Acadêmico)",
    "tde_exp_b": "TDE II — Experimento B (Acadêmico + Demográfico/Contextual)",
}
for resultado in resultados_resumo:
    mascara = (metricas["Cenário"] == rotulos_cenario[resultado["chave_dataset"]]) & (
        metricas["Modelo"] == "Rede Neural (MLP)"
    )
    if mascara.sum() != 1:
        raise ValueError(f"Linha MLP não encontrada de forma única: {resultado['Cenario']}")
    metricas.loc[mascara, ["MAE", "RMSE", "R²", "Média Real", "Média Prevista"]] = [
        resultado["MAE"],
        resultado["RMSE"],
        resultado["R2"],
        resultado["media_real"],
        resultado["media_pred"],
    ]
metricas.to_csv(metricas_path, index=False)
metricas[metricas["Cenário"].str.contains("Vocabulário")].to_csv(
    metricas_dir / "metricas_vocabulario.csv", index=False
)
metricas[metricas["Cenário"].str.contains("TDE II")].to_csv(
    metricas_dir / "metricas_tde.csv", index=False
)

# Atualiza a comparação de métricas, pois os valores de teste do MLP mudaram.
metricas_modelos = metricas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
modelos_unicos = metricas_modelos["Modelo"].unique()
cenarios_unicos = metricas_modelos["Cenário"].unique()
x = np.arange(len(modelos_unicos))
largura = 0.2
for i, cenario in enumerate(cenarios_unicos):
    subset = metricas_modelos[metricas_modelos["Cenário"] == cenario]
    for ax, coluna, titulo, ylabel in [
        (axes[0], "MAE", "Comparativo de MAE por Modelo e Cenário", "MAE"),
        (axes[1], "RMSE", "Comparativo de RMSE por Modelo e Cenário", "RMSE"),
    ]:
        valores = [
            subset.loc[subset["Modelo"] == modelo, coluna].iloc[0]
            if (subset["Modelo"] == modelo).any()
            else 0
            for modelo in modelos_unicos
        ]
        ax.bar(x + (i - 1.5) * largura, valores, largura, label=cenario)
        ax.set_title(titulo, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(modelos_unicos, rotation=20, ha="right")
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(GRAF_DIR / "comparacao_metricas_mae_rmse.png", dpi=120)
plt.close()

# 2. Carregar modelos RF (preservados)
print("\n[2] Carregando modelos RF (preservados, sem re-treino)...")
modelos_rf = {}
splits_rf = {}
for _nome, chave, _inst, _exp in cenarios_mlp:
    rf_pipeline, _ = carregar_artefato_modelo(f"{chave}_rf", models_dir=MODELS_DIR)
    modelos_rf[chave] = rf_pipeline
    df = datasets[chave]
    X_train, X_test, y_train, y_test, _g_tr, _g_te = dividir_treino_teste_agrupado(df)
    splits_rf[chave] = {"X_test": X_test, "y_test": y_test}
    print(f"  Carregado: {chave}_rf")

# 3. Gerar gráficos atualizados
print("\n[3] Gerando gráficos atualizados...")
DPI = 120

# Real vs Previsto — Vocabulário Exp B
y_test_v = splits_mlp["vocabulario_exp_b"]["y_test"]
X_test_v = splits_mlp["vocabulario_exp_b"]["X_test"]
pred_rf_v = modelos_rf["vocabulario_exp_b"].predict(splits_rf["vocabulario_exp_b"]["X_test"])
pred_mlp_v = modelos_mlp["vocabulario_exp_b"].predict(X_test_v)
y_ref_v = splits_rf["vocabulario_exp_b"]["y_test"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_ref_v, pred_rf_v, alpha=0.4, color="#1f77b4")
axes[0].plot([y_test_v.min(), y_test_v.max()], [y_test_v.min(), y_test_v.max()], "r--", lw=2)
axes[0].set_title("Vocabulário Exp B — Random Forest (Real vs Previsto)", fontweight="bold")
axes[0].set_xlabel("Ganho Real")
axes[0].set_ylabel("Ganho Previsto")
axes[1].scatter(y_test_v, pred_mlp_v, alpha=0.4, color="#ff7f0e")
axes[1].plot([y_test_v.min(), y_test_v.max()], [y_test_v.min(), y_test_v.max()], "r--", lw=2)
axes[1].set_title("Vocabulário Exp B — Rede Neural MLP (Real vs Previsto)", fontweight="bold")
axes[1].set_xlabel("Ganho Real")
axes[1].set_ylabel("Ganho Previsto")
plt.tight_layout()
plt.savefig(GRAF_DIR / "real_vs_previsto_vocabulario.png", dpi=DPI)
plt.close()
print("  ✓ real_vs_previsto_vocabulario.png")

# Real vs Previsto — TDE Exp B
y_test_t = splits_mlp["tde_exp_b"]["y_test"]
X_test_t = splits_mlp["tde_exp_b"]["X_test"]
pred_rf_t = modelos_rf["tde_exp_b"].predict(splits_rf["tde_exp_b"]["X_test"])
pred_mlp_t = modelos_mlp["tde_exp_b"].predict(X_test_t)
y_ref_t = splits_rf["tde_exp_b"]["y_test"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y_ref_t, pred_rf_t, alpha=0.4, color="#2ca02c")
axes[0].plot([y_test_t.min(), y_test_t.max()], [y_test_t.min(), y_test_t.max()], "r--", lw=2)
axes[0].set_title("TDE II Exp B — Random Forest (Real vs Previsto)", fontweight="bold")
axes[0].set_xlabel("Ganho Real")
axes[0].set_ylabel("Ganho Previsto")
axes[1].scatter(y_test_t, pred_mlp_t, alpha=0.4, color="#d62728")
axes[1].plot([y_test_t.min(), y_test_t.max()], [y_test_t.min(), y_test_t.max()], "r--", lw=2)
axes[1].set_title("TDE II Exp B — Rede Neural MLP (Real vs Previsto)", fontweight="bold")
axes[1].set_xlabel("Ganho Real")
axes[1].set_ylabel("Ganho Previsto")
plt.tight_layout()
plt.savefig(GRAF_DIR / "real_vs_previsto_tde.png", dpi=DPI)
plt.close()
print("  ✓ real_vs_previsto_tde.png")

# Resíduos — Vocabulário Exp B
res_mlp_v = y_test_v - pred_mlp_v
res_rf_v = y_ref_v - pred_rf_v
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(res_rf_v, bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Resíduos — Random Forest (Vocabulário Exp B)", fontweight="bold")
axes[0].set_xlabel("Resíduo (Real - Previsto)")
axes[1].hist(res_mlp_v, bins=30, color="#ff7f0e", edgecolor="black", alpha=0.7)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Resíduos — Rede Neural MLP (Vocabulário Exp B)", fontweight="bold")
axes[1].set_xlabel("Resíduo (Real - Previsto)")
plt.tight_layout()
plt.savefig(GRAF_DIR / "residuos_vocabulario.png", dpi=DPI)
plt.close()
print("  ✓ residuos_vocabulario.png")

# Permutation Importance (RF)
print("  Calculando permutation importance RF...")
perm_v = calcular_permutation_importance(
    modelos_rf["vocabulario_exp_b"],
    splits_rf["vocabulario_exp_b"]["X_test"],
    splits_rf["vocabulario_exp_b"]["y_test"],
    n_repeats=10,
    random_state=42,
)
perm_t = calcular_permutation_importance(
    modelos_rf["tde_exp_b"],
    splits_rf["tde_exp_b"]["X_test"],
    splits_rf["tde_exp_b"]["y_test"],
    n_repeats=10,
    random_state=42,
)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(
    perm_v["Feature"],
    perm_v["Permutation_Importance_Mean"],
    color="#1f77b4",
    edgecolor="black",
    alpha=0.8,
)
axes[0].invert_yaxis()
axes[0].set_title("Permutation Importance — Vocabulário Exp B (Teste)", fontweight="bold")
axes[0].set_xlabel("Aumento Médio no MAE ao embaralhar feature")
axes[1].barh(
    perm_t["Feature"],
    perm_t["Permutation_Importance_Mean"],
    color="#2ca02c",
    edgecolor="black",
    alpha=0.8,
)
axes[1].invert_yaxis()
axes[1].set_title("Permutation Importance — TDE II Exp B (Teste)", fontweight="bold")
axes[1].set_xlabel("Aumento Médio no MAE ao embaralhar feature")
plt.tight_layout()
plt.savefig(GRAF_DIR / "importancia_features_permutation.png", dpi=DPI)
plt.close()
print("  ✓ importancia_features_permutation.png")

# 4. Resumo final
print("\n" + "=" * 70)
print("RESUMO — MÉTRICAS MLP APÓS CORREÇÃO (early_stopping=False)")
print("=" * 70)
for r in resultados_resumo:
    print(f"  {r['Cenario']:40s} | MAE={r['MAE']:.4f} | RMSE={r['RMSE']:.4f} | R²={r['R2']:.4f}")

print("\n✓ Re-treinamento concluído. Modelos RF preservados sem alteração.")
print("✓ Artefatos MLP atualizados em models/modelagem/")
print("✓ Gráficos atualizados em reports/modelagem/graficos/")
