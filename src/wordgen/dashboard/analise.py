"""Regras analíticas, de qualidade e privacidade do dashboard."""

from collections.abc import Iterable

import pandas as pd

DEMOGRAFICAS = ["sexo", "faixa_etaria", "cor_raca", "zona", "necessidade_especial"]
LIMITE_MINIMO_ESTUDANTES = 10

_BOOLEANOS_VERDADEIROS = {"true", "1", "sim", "s", "yes"}
_BOOLEANOS_FALSOS = {"false", "0", "nao", "não", "n", "no"}
_ALGORITMOS = {
    "randomforestregressor": "Random Forest",
    "mlpregressor": "Rede Neural (MLP)",
}


def exigir_colunas(dados: pd.DataFrame, colunas: Iterable[str], contexto: str) -> None:
    """Falha com mensagem clara quando uma base não tem as colunas necessárias."""
    ausentes = sorted(set(colunas).difference(dados.columns))
    if ausentes:
        raise ValueError(f"{contexto} não contém as colunas necessárias: {', '.join(ausentes)}.")


def normalizar_booleano(valores: pd.Series) -> pd.Series:
    """Converte valores booleanos sem tratar strings não vazias como verdadeiras."""

    def converter(valor: object) -> object:
        if pd.isna(valor):
            return pd.NA
        if isinstance(valor, bool):
            return valor
        if isinstance(valor, (int, float)) and valor in (0, 1):
            return bool(valor)
        if isinstance(valor, str):
            normalizado = valor.strip().casefold()
            if normalizado in _BOOLEANOS_VERDADEIROS:
                return True
            if normalizado in _BOOLEANOS_FALSOS:
                return False
        return pd.NA

    return valores.map(converter).astype("boolean")


def quantidade_estudantes(dados: pd.DataFrame) -> int:
    """Conta estudantes distintos somente para cálculos internos agregados."""
    exigir_colunas(dados, ["aluno_id"], "O recorte de avaliações")
    return int(dados["aluno_id"].nunique(dropna=True))


def recorte_atende_privacidade(
    dados: pd.DataFrame, minimo_grupo: int = LIMITE_MINIMO_ESTUDANTES
) -> bool:
    """Indica se o recorte pode ser exibido sem revelar um grupo pequeno."""
    return quantidade_estudantes(dados) >= minimo_grupo


def grupos_atendem_privacidade(
    dados: pd.DataFrame,
    colunas_grupo: list[str],
    minimo_grupo: int = LIMITE_MINIMO_ESTUDANTES,
) -> bool:
    """Exige o limiar em cada célula visível de uma decomposição."""
    exigir_colunas(dados, ["aluno_id", *colunas_grupo], "A decomposição analítica")
    if dados.empty:
        return False
    tamanhos = dados.groupby(colunas_grupo, dropna=False)["aluno_id"].nunique(dropna=True)
    return bool(not tamanhos.empty and (tamanhos >= minimo_grupo).all())


def filtrar_avaliacoes(
    dados: pd.DataFrame,
    fases: Iterable[str] | None = None,
    momentos: Iterable[str] | None = None,
    series: Iterable[str] | None = None,
    escolas: Iterable[str] | None = None,
    turmas: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aplica filtros disponíveis; listas vazias não restringem o recorte."""
    resultado = dados.copy()
    filtros = {
        "fase": fases,
        "periodo": momentos,
        "serie": series,
        "escola": escolas,
        "turma": turmas,
    }
    for coluna, valores in filtros.items():
        if valores:
            exigir_colunas(resultado, [coluna], "O dataset de avaliações")
            resultado = resultado[resultado[coluna].astype(str).isin(map(str, valores))]
    return resultado


def resumo_cobertura(dados: pd.DataFrame, total_itens: int) -> dict[str, float | int | None]:
    """Calcula completude sem assumir que strings são valores booleanos válidos."""
    exigir_colunas(dados, ["aluno_id", "completo"], "O dataset de avaliações")
    completo = normalizar_booleano(dados["completo"])
    observados = int(completo.notna().sum())
    completas = int((completo == True).sum())  # noqa: E712
    incompletas = int((completo == False).sum())  # noqa: E712
    sem_indicador = int(completo.isna().sum())
    return {
        "avaliacoes": len(dados),
        "estudantes": quantidade_estudantes(dados),
        "completas": completas,
        "incompletas": incompletas,
        "sem_indicador": sem_indicador,
        "cobertura": float(completas / observados) if observados else None,
        "itens_esperados": total_itens,
    }


def proporcao_completo(valores: pd.Series) -> float | None:
    """Calcula a proporção de avaliações completas entre indicadores válidos."""
    completo = normalizar_booleano(valores)
    observados = int(completo.notna().sum())
    if not observados:
        return None
    return float((completo == True).sum() / observados)  # noqa: E712


def agregar_desempenho(dados: pd.DataFrame, coluna_total: str) -> pd.DataFrame:
    """Resume pontuação por fase, momento e série sem misturar escalas."""
    exigir_colunas(dados, ["fase", "periodo", "serie", coluna_total], "O dataset de avaliações")
    return (
        dados.groupby(["fase", "periodo", "serie"], dropna=False)[coluna_total]
        .agg(media="mean", mediana="median", desvio="std", n="count")
        .reset_index()
    )


def agregar_demografia(
    avaliacoes: pd.DataFrame,
    estudantes: pd.DataFrame,
    atributo: str,
    coluna_total: str,
    minimo_grupo: int = LIMITE_MINIMO_ESTUDANTES,
) -> pd.DataFrame:
    """Agrega por estudantes distintos e aplica supressão complementar conservadora.

    Se qualquer categoria tiver menos que o limiar, nenhuma categoria da decomposição
    é devolvida. A medida evita inferência por diferença entre totais e categorias visíveis.
    """
    exigir_colunas(avaliacoes, ["aluno_id", coluna_total], "As avaliações")
    exigir_colunas(estudantes, ["aluno_id", atributo], "A dimensão de estudantes")
    dimensao = estudantes[["aluno_id", atributo]].copy()
    if dimensao["aluno_id"].isna().any() or dimensao["aluno_id"].duplicated().any():
        raise ValueError("dim_aluno deve conter exatamente um registro não nulo por aluno_id.")
    combinado = avaliacoes.merge(dimensao, on="aluno_id", how="left", validate="many_to_one")
    combinado[atributo] = combinado[atributo].fillna("Sem informação")
    resumo = (
        combinado.groupby(atributo, dropna=False)
        .agg(n_estudantes=("aluno_id", "nunique"), media=(coluna_total, "mean"))
        .reset_index()
    )
    if (resumo["n_estudantes"] < minimo_grupo).any():
        suprimido = pd.DataFrame(columns=[atributo, "n_estudantes", "media"])
        suprimido.attrs["suprimido_por_privacidade"] = True
        return suprimido
    resumo.attrs["suprimido_por_privacidade"] = False
    return resumo


def _chave_artefato(nome: str) -> tuple[str, str, str] | None:
    partes = nome.casefold().split("_")
    if len(partes) != 4 or partes[1] != "exp" or partes[2] not in {"a", "b"}:
        return None
    instrumento = {"vocabulario": "Vocabulário", "tde": "TDE II"}.get(partes[0])
    algoritmo = {"rf": "Random Forest", "mlp": "Rede Neural (MLP)"}.get(partes[3])
    if not instrumento or not algoritmo:
        return None
    return instrumento, partes[2].upper(), algoritmo


def associar_metadados_modelagem(
    metadados: dict[str, dict], metricas: pd.DataFrame, instrumento: str = "Todos"
) -> tuple[pd.DataFrame, list[str]]:
    """Associa metadados a métricas somente quando chave e métricas conferem exatamente."""
    linhas: list[dict] = []
    avisos: list[str] = []
    for nome, metadado in sorted(metadados.items()):
        chave = _chave_artefato(nome)
        if chave is None:
            avisos.append(f"Artefato ignorado por nome não reconhecido: {nome}.")
            continue
        instrumento_meta, experimento, algoritmo = chave
        if instrumento != "Todos" and instrumento_meta != instrumento:
            continue
        algoritmo_meta = _ALGORITMOS.get(
            str(metadado.get("algoritmo") or metadado.get("modelo") or "").casefold()
        )
        if (
            metadado.get("instrumento") != instrumento_meta
            or str(metadado.get("experimento", "")).upper() != experimento
            or algoritmo_meta != algoritmo
        ):
            avisos.append(f"Metadado não associado por inconsistência interna: {nome}.")
            continue
        correspondente = metricas[
            metricas["Cenário"].str.startswith(f"{instrumento_meta} — Experimento {experimento}")
            & (metricas["Modelo"] == algoritmo)
        ]
        if len(correspondente) != 1:
            avisos.append(f"Metadado sem métrica de teste unívoca: {nome}.")
            continue
        teste = metadado.get("metricas_teste", {})
        mae = teste.get("MAE", metadado.get("MAE_teste"))
        rmse = teste.get("RMSE", metadado.get("RMSE_teste"))
        r2 = teste.get("R2", metadado.get("R2_teste"))
        linha_metrica = correspondente.iloc[0]
        valores = [mae, rmse, r2]
        esperados = [linha_metrica["MAE"], linha_metrica["RMSE"], linha_metrica["R²"]]
        if any(valor is None for valor in valores) or any(
            abs(float(valor) - float(esperado)) > 0.0001
            for valor, esperado in zip(valores, esperados, strict=True)
        ):
            avisos.append(f"Metadado não associado por divergência de métricas: {nome}.")
            continue
        linhas.append(
            {
                "Artefato": nome,
                "Instrumento": instrumento_meta,
                "Experimento": f"Exp {experimento}",
                "Modelo": algoritmo,
                "MAE teste": mae,
                "RMSE teste": rmse,
                "R² teste": r2,
                "CV MAE (treino)": metadado.get("cv_mae_medio"),
                "Tamanho treino": metadado.get("n_treino", metadado.get("amostras_treino")),
                "Tamanho teste": metadado.get("n_teste", metadado.get("amostras_teste")),
            }
        )
    return pd.DataFrame(linhas), avisos


def valor_kpi(kpis: pd.DataFrame, nome: str) -> tuple[float | None, str]:
    """Obtém um KPI mestre exportado pelo notebook, preservando sua definição."""
    exigir_colunas(kpis, ["KPI", "Valor_Numerico", "Valor_Formatado", "Descricao"], "KPIs")
    linha = kpis.loc[kpis["KPI"].eq(nome)]
    if len(linha) != 1:
        return None, "Indicador indisponível no artefato analítico."
    registro = linha.iloc[0]
    return float(registro["Valor_Numerico"]), str(registro["Descricao"])


def resumo_modelagem(metricas: pd.DataFrame) -> pd.DataFrame:
    """Resume o melhor modelo por MAE de teste, sem misturar baselines e modelos."""
    exigir_colunas(metricas, ["Cenário", "Modelo", "MAE", "RMSE", "R²"], "Métricas")
    trabalho = metricas.copy()
    trabalho[["Instrumento", "Experimento"]] = trabalho["Cenário"].str.extract(
        r"^(.*?) — Experimento ([AB])", expand=True
    )
    modelos = trabalho[trabalho["Modelo"].isin(["Random Forest", "Rede Neural (MLP)"])]
    if modelos.empty:
        return pd.DataFrame()
    indices = modelos.groupby(["Instrumento", "Experimento"], dropna=False)["MAE"].idxmin()
    melhores = modelos.loc[indices].copy()
    baseline = trabalho[trabalho["Modelo"].eq("Baseline 2 (Média Treino)")]
    melhores = melhores.merge(
        baseline[["Instrumento", "Experimento", "MAE"]].rename(columns={"MAE": "MAE baseline"}),
        on=["Instrumento", "Experimento"],
        how="left",
        validate="one_to_one",
    )
    melhores["Reducao MAE (%)"] = (
        (melhores["MAE baseline"] - melhores["MAE"]) / melhores["MAE baseline"] * 100
    )
    return melhores.sort_values(["Instrumento", "Experimento"])
