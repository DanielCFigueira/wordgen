"""Script orquestrador para executar o pipeline completo de modelagem preditiva."""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Configura diretório de cache do matplotlib para evitar warnings no Windows
os.environ["MPLCONFIGDIR"] = str(Path("models/modelagem/.mplcache").resolve())
matplotlib.use("Agg")

from wordgen.modelagem.avaliacao import (
    avaliar_modelo,
    consolidar_tabela_metricas,
)
from wordgen.modelagem.baselines import BaselineGanhoZero, BaselineMediaTreino
from wordgen.modelagem.interpretacao import (
    calcular_permutation_importance,
)
from wordgen.modelagem.persistencia import salvar_artefato_modelo
from wordgen.modelagem.preparacao import carregar_e_preparar_dados
from wordgen.modelagem.treinamento import (
    treinar_e_otimizar_mlp,
    treinar_e_otimizar_rf,
)
from wordgen.modelagem.validacao import dividir_treino_teste_agrupado


def executar_pipeline():
    print("=" * 70)
    print("INICIANDO EXECUÇÃO COMPLETA DA MODELAGEM DE MACHINE LEARNING — WORDGEN")
    print("=" * 70)

    # 1. Preparação dos datasets
    print("\n[PASSO 1] Carregando e preparando bases analíticas...")
    datasets = carregar_e_preparar_dados("data")

    cenarios = [
        ("Vocabulário — Experimento A (Acadêmico)", "vocabulario_exp_a", "Vocabulário", "A"),
        (
            "Vocabulário — Experimento B (Acadêmico + Demográfico/Contextual)",
            "vocabulario_exp_b",
            "Vocabulário",
            "B",
        ),
        ("TDE II — Experimento A (Acadêmico)", "tde_exp_a", "TDE II", "A"),
        ("TDE II — Experimento B (Acadêmico + Demográfico/Contextual)", "tde_exp_b", "TDE II", "B"),
    ]

    todos_resultados = []
    detalhes_execucao = {}
    modelos_finais = {}
    dados_teste = {}

    for nome_cenario, chave_dataset, instrumento, exp in cenarios:
        print("\n" + "-" * 60)
        print(f"PROCESSANDO CENÁRIO: {nome_cenario}")
        print("-" * 60)

        df = datasets[chave_dataset]
        print(f"Total de pares completos elegíveis: {len(df)}")
        print(f"Total de estudantes únicos: {df['aluno_id'].nunique()}")

        # Separação por aluno_id
        X_train, X_test, y_train, y_test, g_train, g_test = dividir_treino_teste_agrupado(
            df, target_col="ganho", grupo_col="aluno_id", test_size=0.20, random_state=42
        )

        n_alunos_treino = len(set(g_train))
        n_alunos_teste = len(set(g_test))
        print(f"Treino: {len(X_train)} amostras ({n_alunos_treino} estudantes únicos)")
        print(f"Teste:  {len(X_test)} amostras ({n_alunos_teste} estudantes únicos)")

        colunas_numericas = ["total_pre"]
        colunas_categoricas = [c for c in X_train.columns if c not in colunas_numericas]
        print(f"Features numéricas: {colunas_numericas}")
        print(f"Features categóricas: {colunas_categoricas}")

        # Guardar dados de teste para visualizações
        dados_teste[chave_dataset] = {
            "X_test": X_test,
            "y_test": y_test,
            "g_test": g_test,
        }

        # -------------------------------------------------------------
        # 1. Baseline 1: Ganho Zero
        # -------------------------------------------------------------
        b_zero = BaselineGanhoZero()
        b_zero.fit(X_train, y_train)
        diag_zero = avaliar_modelo(b_zero, X_test, y_test, "Baseline 1 (Ganho Zero)", nome_cenario)
        todos_resultados.append(diag_zero)
        print(
            f"-> Baseline 1 (Zero):  MAE={diag_zero['MAE']:.4f}, RMSE={diag_zero['RMSE']:.4f}, R²={diag_zero['R2']:.4f}"
        )

        # -------------------------------------------------------------
        # 2. Baseline 2: Média do Treino
        # -------------------------------------------------------------
        b_media = BaselineMediaTreino()
        b_media.fit(X_train, y_train)
        diag_media = avaliar_modelo(
            b_media, X_test, y_test, "Baseline 2 (Média Treino)", nome_cenario
        )
        todos_resultados.append(diag_media)
        print(
            f"-> Baseline 2 (Média): MAE={diag_media['MAE']:.4f}, RMSE={diag_media['RMSE']:.4f}, R²={diag_media['R2']:.4f}"
        )

        # -------------------------------------------------------------
        # 3. Random Forest Regressor
        # -------------------------------------------------------------
        print("Treinando e otimizando Random Forest Regressor com GroupKFold...")
        rf_pipe, info_rf = treinar_e_otimizar_rf(
            X_train,
            y_train,
            g_train,
            colunas_numericas,
            colunas_categoricas,
            cv_splits=5,
            random_state=42,
        )
        diag_rf = avaliar_modelo(rf_pipe, X_test, y_test, "Random Forest", nome_cenario)
        todos_resultados.append(diag_rf)
        modelos_finais[f"{chave_dataset}_rf"] = rf_pipe
        print(
            f"-> Random Forest:      MAE={diag_rf['MAE']:.4f}, RMSE={diag_rf['RMSE']:.4f}, R²={diag_rf['R2']:.4f}"
        )
        print(f"   Melhores parâmetros: {info_rf['melhores_parametros']}")

        # -------------------------------------------------------------
        # 4. Rede Neural Artificial (MLPRegressor)
        # -------------------------------------------------------------
        print("Treinando e otimizando Rede Neural Artificial (MLPRegressor) com GroupKFold...")
        mlp_pipe, info_mlp = treinar_e_otimizar_mlp(
            X_train,
            y_train,
            g_train,
            colunas_numericas,
            colunas_categoricas,
            cv_splits=5,
            random_state=42,
        )
        diag_mlp = avaliar_modelo(mlp_pipe, X_test, y_test, "Rede Neural (MLP)", nome_cenario)
        todos_resultados.append(diag_mlp)
        modelos_finais[f"{chave_dataset}_mlp"] = mlp_pipe
        print(
            f"-> Rede Neural (MLP):  MAE={diag_mlp['MAE']:.4f}, RMSE={diag_mlp['RMSE']:.4f}, R²={diag_mlp['R2']:.4f}"
        )
        print(f"   Melhores parâmetros: {info_mlp['melhores_parametros']}")

        # -------------------------------------------------------------
        # Persistência de Artefatos
        # -------------------------------------------------------------
        meta_rf = {
            "instrumento": instrumento,
            "experimento": exp,
            "cenario": nome_cenario,
            "modelo": "RandomForestRegressor",
            "parametros": info_rf["melhores_parametros"],
            "cv_mae_medio": info_rf["cv_mae_medio"],
            "metricas_teste": {"MAE": diag_rf["MAE"], "RMSE": diag_rf["RMSE"], "R2": diag_rf["R2"]},
            "features_numericas": colunas_numericas,
            "features_categoricas": colunas_categoricas,
            "amostras_treino": len(X_train),
            "alunos_treino": n_alunos_treino,
            "amostras_teste": len(X_test),
            "alunos_teste": n_alunos_teste,
        }
        salvar_artefato_modelo(rf_pipe, f"{chave_dataset}_rf", meta_rf, "models/modelagem")

        meta_mlp = {
            "instrumento": instrumento,
            "experimento": exp,
            "cenario": nome_cenario,
            "modelo": "MLPRegressor",
            "parametros": info_mlp["melhores_parametros"],
            "cv_mae_medio": info_mlp["cv_mae_medio"],
            "metricas_teste": {
                "MAE": diag_mlp["MAE"],
                "RMSE": diag_mlp["RMSE"],
                "R2": diag_mlp["R2"],
            },
            "features_numericas": colunas_numericas,
            "features_categoricas": colunas_categoricas,
            "amostras_treino": len(X_train),
            "alunos_treino": n_alunos_treino,
            "amostras_teste": len(X_test),
            "alunos_teste": n_alunos_teste,
        }
        salvar_artefato_modelo(mlp_pipe, f"{chave_dataset}_mlp", meta_mlp, "models/modelagem")

        detalhes_execucao[chave_dataset] = {
            "rf": {"diag": diag_rf, "info": info_rf, "meta": meta_rf},
            "mlp": {"diag": diag_mlp, "info": info_mlp, "meta": meta_mlp},
            "b_zero": diag_zero,
            "b_media": diag_media,
        }

    # -------------------------------------------------------------
    # Consolidação das Métricas em Tabela
    # -------------------------------------------------------------
    df_metricas = consolidar_tabela_metricas(todos_resultados)
    reports_met_dir = Path("reports/modelagem/metricas")
    reports_met_dir.mkdir(parents=True, exist_ok=True)
    df_metricas.to_csv(reports_met_dir / "metricas_comparativas_gerais.csv", index=False)

    # Salva também tabelas específicas por instrumento
    df_vocab = df_metricas[df_metricas["Cenário"].str.contains("Vocabulário")].reset_index(
        drop=True
    )
    df_tde = df_metricas[df_metricas["Cenário"].str.contains("TDE II")].reset_index(drop=True)
    df_vocab.to_csv(reports_met_dir / "metricas_vocabulario.csv", index=False)
    df_tde.to_csv(reports_met_dir / "metricas_tde.csv", index=False)

    print("\n" + "=" * 70)
    print("TABELA COMPARATIVA GERAL DE DESEMPENHO")
    print("=" * 70)
    print(df_metricas.to_string(index=False))

    # -------------------------------------------------------------
    # Geração de Visualizações Gráficas com Matplotlib Puro
    # -------------------------------------------------------------
    graf_dir = Path("reports/modelagem/graficos")
    graf_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use(
        "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
    )

    # Gráfico 1: Comparação de MAE e RMSE entre Modelos
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # MAE
    modelos_unicos = df_metricas["Modelo"].unique()
    cenarios_unicos = df_metricas["Cenário"].unique()
    x = np.arange(len(modelos_unicos))
    largura = 0.2

    for i, cen in enumerate(cenarios_unicos):
        sub_df = df_metricas[df_metricas["Cenário"] == cen]
        val_mae = [
            sub_df[sub_df["Modelo"] == m]["MAE"].values[0]
            if len(sub_df[sub_df["Modelo"] == m]) > 0
            else 0
            for m in modelos_unicos
        ]
        axes[0].bar(x + (i - 1.5) * largura, val_mae, largura, label=cen)

    axes[0].set_title("Comparativo de MAE por Modelo e Cenário", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("MAE (Erro Médio Absoluto)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(modelos_unicos, rotation=20, ha="right")
    axes[0].legend(fontsize=8)

    # RMSE
    for i, cen in enumerate(cenarios_unicos):
        sub_df = df_metricas[df_metricas["Cenário"] == cen]
        val_rmse = [
            sub_df[sub_df["Modelo"] == m]["RMSE"].values[0]
            if len(sub_df[sub_df["Modelo"] == m]) > 0
            else 0
            for m in modelos_unicos
        ]
        axes[1].bar(x + (i - 1.5) * largura, val_rmse, largura, label=cen)

    axes[1].set_title("Comparativo de RMSE por Modelo e Cenário", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("RMSE (Raiz do Erro Quadrático Médio)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modelos_unicos, rotation=20, ha="right")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(graf_dir / "comparacao_metricas_mae_rmse.png", dpi=120)
    plt.close()

    # Gráfico 2: Distribuição dos Ganhos Reais
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        datasets["vocabulario_exp_a"]["ganho"],
        bins=30,
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0].set_title("Distribuição do Ganho — Vocabulário (Pares Completos)", fontweight="bold")
    axes[0].set_xlabel("Ganho (Pós - Pré)")
    axes[0].set_ylabel("Frequência")

    axes[1].hist(
        datasets["tde_exp_a"]["ganho"], bins=30, color="#2b8f5c", edgecolor="black", alpha=0.7
    )
    axes[1].set_title("Distribuição do Ganho — TDE II (Pares Completos)", fontweight="bold")
    axes[1].set_xlabel("Ganho (Pós - Pré)")
    axes[1].set_ylabel("Frequência")

    plt.tight_layout()
    plt.savefig(graf_dir / "distribuicao_ganho_instrumentos.png", dpi=120)
    plt.close()

    # Gráfico 3: Real vs Previsto (Random Forest vs MLP) no Vocabulário Exp B
    y_test_v = dados_teste["vocabulario_exp_b"]["y_test"]
    pred_rf_v = modelos_finais["vocabulario_exp_b_rf"].predict(
        dados_teste["vocabulario_exp_b"]["X_test"]
    )
    pred_mlp_v = modelos_finais["vocabulario_exp_b_mlp"].predict(
        dados_teste["vocabulario_exp_b"]["X_test"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_test_v, pred_rf_v, alpha=0.4, color="#1f77b4")
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
    plt.savefig(graf_dir / "real_vs_previsto_vocabulario.png", dpi=120)
    plt.close()

    # Gráfico 4: Real vs Previsto (TDE II Exp B)
    y_test_t = dados_teste["tde_exp_b"]["y_test"]
    pred_rf_t = modelos_finais["tde_exp_b_rf"].predict(dados_teste["tde_exp_b"]["X_test"])
    pred_mlp_t = modelos_finais["tde_exp_b_mlp"].predict(dados_teste["tde_exp_b"]["X_test"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_test_t, pred_rf_t, alpha=0.4, color="#2ca02c")
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
    plt.savefig(graf_dir / "real_vs_previsto_tde.png", dpi=120)
    plt.close()

    # Gráfico 5: Distribuição dos Resíduos
    res_rf_v = y_test_v - pred_rf_v
    res_mlp_v = y_test_v - pred_mlp_v
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
    plt.savefig(graf_dir / "residuos_vocabulario.png", dpi=120)
    plt.close()

    # Gráfico 6: Importância das Variáveis (Permutation Importance)
    print("Calculando Permutation Importance no conjunto de teste...")
    perm_v = calcular_permutation_importance(
        modelos_finais["vocabulario_exp_b_rf"],
        dados_teste["vocabulario_exp_b"]["X_test"],
        dados_teste["vocabulario_exp_b"]["y_test"],
        n_repeats=10,
        random_state=42,
    )
    perm_t = calcular_permutation_importance(
        modelos_finais["tde_exp_b_rf"],
        dados_teste["tde_exp_b"]["X_test"],
        dados_teste["tde_exp_b"]["y_test"],
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
    plt.savefig(graf_dir / "importancia_features_permutation.png", dpi=120)
    plt.close()

    # Gráfico 7: Efeito de Regressão à Média (total_pre vs ganho)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_v = datasets["vocabulario_exp_a"]
    axes[0].scatter(df_v["total_pre"], df_v["ganho"], alpha=0.2, color="#1f77b4")
    m_v, b_v = np.polyfit(df_v["total_pre"], df_v["ganho"], 1)
    axes[0].plot(df_v["total_pre"], m_v * df_v["total_pre"] + b_v, color="red", lw=2)
    axes[0].set_title("Vocabulário: Relação entre Pré-teste e Ganho", fontweight="bold")
    axes[0].set_xlabel("Pontuação Pré-teste")
    axes[0].set_ylabel("Ganho (Pós - Pré)")

    df_t = datasets["tde_exp_a"]
    axes[1].scatter(df_t["total_pre"], df_t["ganho"], alpha=0.2, color="#2ca02c")
    m_t, b_t = np.polyfit(df_t["total_pre"], df_t["ganho"], 1)
    axes[1].plot(df_t["total_pre"], m_t * df_t["total_pre"] + b_t, color="red", lw=2)
    axes[1].set_title("TDE II: Relação entre Pré-teste e Ganho", fontweight="bold")
    axes[1].set_xlabel("Pontuação Pré-teste")
    axes[1].set_ylabel("Ganho (Pós - Pré)")

    plt.tight_layout()
    plt.savefig(graf_dir / "efeito_regressao_media.png", dpi=120)
    plt.close()

    print("\nVisualizações geradas e salvas com sucesso em reports/modelagem/graficos/!")
    print("Métricas salvas em reports/modelagem/metricas/!")
    print("Modelos e metadados persistidos em models/modelagem/!")


if __name__ == "__main__":
    executar_pipeline()
