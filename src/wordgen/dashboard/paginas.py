"""Páginas analíticas do dashboard WordGen."""

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st

from wordgen.dashboard.analise import (
    DEMOGRAFICAS,
    LIMITE_MINIMO_ESTUDANTES,
    agregar_demografia,
    associar_metadados_modelagem,
    filtrar_avaliacoes,
    grupos_atendem_privacidade,
    normalizar_booleano,
    recorte_atende_privacidade,
    resumo_cobertura,
    resumo_modelagem,
    valor_kpi,
)
from wordgen.dashboard.visual import CORAL, GOLD, MUTED, TEAL, formatar_figura

INSTRUMENTOS = {
    "Vocabulário": ("vocabulario", "total_vocabulario", 50, 50),
    "TDE II · Escrita": ("tde", "total_tde", 80, 40),
}
ROTULOS_MOMENTO = {"pre": "Pré-teste", "pos": "Pós-teste"}


def cabecalho(titulo: str, texto: str) -> None:
    """Renderiza somente texto sanitizado dentro do HTML de apresentação."""
    st.markdown('<div class="wg-kicker">WORDGEN · PAINEL ANALÍTICO</div>', unsafe_allow_html=True)
    st.title(titulo)
    st.markdown(f'<div class="wg-subtitle">{escape(texto)}</div>', unsafe_allow_html=True)


def sem_dados(mensagem: str = "Nenhuma avaliação corresponde aos filtros selecionados.") -> None:
    st.info(mensagem)


def _recorte_protegido(df: pd.DataFrame) -> bool:
    if recorte_atende_privacidade(df):
        return True
    st.info(
        f"O recorte foi suprimido para preservar a privacidade: são necessários ao menos "
        f"{LIMITE_MINIMO_ESTUDANTES} estudantes distintos."
    )
    return False


def _decomposicao_protegida(df: pd.DataFrame, colunas: list[str]) -> bool:
    if grupos_atendem_privacidade(df, colunas):
        return True
    st.info(
        f"Esta decomposição foi suprimida: cada grupo precisa reunir ao menos "
        f"{LIMITE_MINIMO_ESTUDANTES} estudantes distintos. A supressão é integral para "
        "evitar inferência por diferença entre totais e categorias visíveis."
    )
    return False


def _rotular_momento(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    resultado["Momento"] = (
        resultado["periodo"].astype(str).map(ROTULOS_MOMENTO).fillna("Sem informação")
    )
    return resultado


def _metricas_topo(df: pd.DataFrame, n_itens: int) -> None:
    resumo = resumo_cobertura(df, n_itens)
    cols = st.columns(4)
    cols[0].metric(
        "Avaliações",
        f"{resumo['avaliacoes']:,}".replace(",", "."),
        help="Registros de aplicação, não estudantes únicos.",
    )
    cols[1].metric(
        "Estudantes",
        f"{resumo['estudantes']:,}".replace(",", "."),
        help="Estudantes distintos no recorte.",
    )
    cols[2].metric("Avaliações completas", f"{resumo['completas']:,}".replace(",", "."))
    cobertura = "Não disponível" if resumo["cobertura"] is None else f"{resumo['cobertura']:.1%}"
    cols[3].metric("Completude", cobertura)
    if resumo["sem_indicador"]:
        st.caption(
            "Algumas aplicações não possuem indicador de completude e não entram no denominador."
        )


def _formatar_numero(valor: float | int | None) -> str:
    if valor is None or pd.isna(valor):
        return "Não disponível"
    return f"{valor:,.0f}".replace(",", ".")


def _kpi_mestre(kpis: pd.DataFrame, nome: str) -> tuple[str, str]:
    if kpis.empty:
        return "Não disponível", "Artefato de KPIs não encontrado."
    valor, descricao = valor_kpi(kpis, nome)
    return _formatar_numero(valor), descricao


def visao_geral(dados: dict, filtros: dict) -> None:
    cabecalho(
        "Aprendizagem em perspectiva",
        "Uma leitura agregada das avaliações do WordGen nas fases 2, 3 e 4.",
    )
    st.subheader("Indicadores executivos da base consolidada")
    if filtros["instrumento"] != "Todos" or any(filtros["dimensoes"].values()):
        st.caption(
            "Os indicadores abaixo são mestres da base consolidada e não variam com o recorte. "
            "Os indicadores filtrados aparecem logo após esta seção."
        )
    kpis = dados["kpis"]
    cards = [
        ("Estudantes únicos", "Estudantes Únicos (União Vocabulário e TDE)"),
        ("Turmas identificadas", "Turmas em Escolas Identificadas"),
        ("Escolas públicas", "Escolas Públicas Atendidas"),
        ("Alunos em Vocabulário", "Estudantes Únicos em Vocabulário"),
        ("Alunos em TDE II", "Estudantes Únicos em TDE II"),
        ("Aplicações", "Total de Registros de Testes Aplicados"),
    ]
    colunas = st.columns(3)
    for indice, (rotulo, chave_kpi) in enumerate(cards):
        valor, ajuda = _kpi_mestre(kpis, chave_kpi)
        colunas[indice % 3].metric(rotulo, valor, help=ajuda)
    st.caption(
        "Turmas consideram a combinação válida de escola, série e turma. Estudantes na união "
        "não são obtidos somando os instrumentos."
    )
    st.divider()
    instrumentos = (
        list(INSTRUMENTOS) if filtros["instrumento"] == "Todos" else [filtros["instrumento"]]
    )
    registros = []
    for nome in instrumentos:
        chave, col_total, maximo, itens = INSTRUMENTOS[nome]
        base = filtrar_avaliacoes(dados[chave], **filtros["dimensoes"])
        if not base.empty:
            registros.append((nome, col_total, maximo, itens, base))
    if not registros:
        sem_dados()
        return
    uniao = pd.concat(
        [item[4].assign(Instrumento=item[0]) for item in registros], ignore_index=True
    )
    if not _recorte_protegido(uniao):
        return
    n_avaliacoes = len(uniao)
    n_alunos = uniao["aluno_id"].nunique()
    n_fases = uniao["fase"].astype(str).nunique()
    completo = normalizar_booleano(uniao["completo"])
    observados = completo.notna()
    completude = completo[observados].eq(True).mean() if observados.any() else None
    m = st.columns(4)
    m[0].metric(
        "Registros de avaliação",
        f"{n_avaliacoes:,}".replace(",", "."),
        help="Uma aplicação de instrumento equivale a um registro.",
    )
    m[1].metric(
        "Estudantes únicos",
        f"{n_alunos:,}".replace(",", "."),
        help="União de aluno_id dos instrumentos selecionados, sem somar duplicidades.",
    )
    m[2].metric("Fases no recorte", str(n_fases))
    m[3].metric("Completude média", "Não disponível" if completude is None else f"{completude:.1%}")
    st.caption(
        "Registros de avaliação não são estudantes únicos. O total de estudantes é a união "
        "interna dos identificadores dos instrumentos selecionados; por isso não corresponde "
        "à soma de cada instrumento."
    )
    if not _decomposicao_protegida(uniao, ["Instrumento", "fase", "periodo"]):
        return
    st.subheader("Como as avaliações se distribuem")
    volume = _rotular_momento(
        uniao.groupby(["Instrumento", "fase", "periodo"], dropna=False)
        .size()
        .rename("Avaliações")
        .reset_index()
    )
    volume["fase"] = "Fase " + volume["fase"].astype(str).str.replace("Fase ", "", regex=False)
    fig = px.bar(
        volume,
        x="fase",
        y="Avaliações",
        color="Momento",
        barmode="group",
        facet_col="Instrumento" if len(registros) > 1 else None,
        color_discrete_map={"Pré-teste": TEAL, "Pós-teste": CORAL},
        category_orders={"Momento": ["Pré-teste", "Pós-teste"]},
        labels={"fase": "Fase"},
        title="Volume por fase e momento",
    )
    st.plotly_chart(formatar_figura(fig, 460), width="stretch", theme=None)
    cov = _rotular_momento(
        uniao.assign(_completo=normalizar_booleano(uniao["completo"]))
        .groupby(["Instrumento", "fase", "periodo"], dropna=False)["_completo"]
        .agg(lambda s: s.dropna().eq(True).mean())
        .rename("completude")
        .reset_index()
    )
    cov["fase"] = "Fase " + cov["fase"].astype(str)
    fig = px.line(
        cov,
        x="fase",
        y="completude",
        color="Momento",
        markers=True,
        facet_col="Instrumento" if len(registros) > 1 else None,
        color_discrete_map={"Pré-teste": TEAL, "Pós-teste": CORAL},
        labels={"fase": "Fase", "completude": "Proporção completa"},
        title="Completude por aplicação",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(formatar_figura(fig, 460), width="stretch", theme=None)
    st.markdown(
        '<div class="wg-note">As quantidades representam aplicações. Os instrumentos têm '
        "escalas distintas e suas pontuações brutas não são comparadas entre si.</div>",
        unsafe_allow_html=True,
    )


def pagina_instrumento(dados: dict, nome: str, filtros: dict) -> None:
    chave, coluna, teto, itens = INSTRUMENTOS[nome]
    cabecalho(
        nome, f"Distribuição, posição relativa e evolução entre momentos · escala de 0 a {teto}."
    )
    df = filtrar_avaliacoes(dados[chave], **filtros["dimensoes"])
    if df.empty:
        sem_dados()
        return
    if not _recorte_protegido(df):
        return
    _metricas_topo(df, itens)
    esquerda, direita = st.columns(2)
    with esquerda:
        fig = px.histogram(
            _rotular_momento(df),
            x=coluna,
            color="Momento",
            barmode="overlay",
            opacity=0.72,
            nbins=24,
            color_discrete_map={"Pré-teste": TEAL, "Pós-teste": CORAL},
            range_x=[0, teto],
            labels={coluna: "Pontuação"},
            title="Distribuição das pontuações",
        )
        st.plotly_chart(formatar_figura(fig), width="stretch", theme=None)
    with direita:
        if _decomposicao_protegida(df, ["fase", "periodo"]):
            resumo = _rotular_momento(
                df.groupby(["fase", "periodo"], dropna=False)[coluna]
                .mean()
                .rename("media")
                .reset_index()
            )
            resumo["fase"] = "Fase " + resumo["fase"].astype(str)
            fig = px.line(
                resumo,
                x="fase",
                y="media",
                color="Momento",
                markers=True,
                color_discrete_map={"Pré-teste": TEAL, "Pós-teste": CORAL},
                labels={"fase": "Fase", "media": "Média (pontos)"},
                title="Média por fase",
            )
            fig.update_yaxes(range=[0, teto])
            st.plotly_chart(formatar_figura(fig), width="stretch", theme=None)
    st.subheader("Diferença pareada entre pré e pós")
    if filtros["dimensoes"]["momentos"]:
        st.caption(
            "O ganho exige pré e pós pareados; portanto, o filtro de momento não se aplica "
            "a esta seção."
        )
    paired = dados[f"pareado_{chave}"].copy()
    completos = normalizar_booleano(paired["completo_pre"]).eq(True) & normalizar_booleano(
        paired["completo_pos"]
    ).eq(True)
    paired = paired[completos]
    for coluna_filtro, chave_filtro in [
        ("fase", "fases"),
        ("serie", "series"),
        ("escola", "escolas"),
        ("turma", "turmas"),
    ]:
        valores = filtros["dimensoes"][chave_filtro]
        if valores:
            paired = paired[paired[coluna_filtro].astype(str).isin(valores)]
    if paired.empty:
        sem_dados("Não há pares completos no recorte para calcular ganho.")
    elif _recorte_protegido(paired):
        paired = paired.assign(ganho=paired["total_pos"] - paired["total_pre"])
        a, b = st.columns([1.3, 1])
        with a:
            if _decomposicao_protegida(paired, ["fase"]):
                fig = px.histogram(
                    paired,
                    x="ganho",
                    color="fase",
                    marginal="box",
                    nbins=30,
                    color_discrete_sequence=[TEAL, CORAL, GOLD],
                    labels={"ganho": "Ganho (pontos)", "fase": "Fase"},
                    title="Distribuição do ganho nos pares completos",
                )
                st.plotly_chart(formatar_figura(fig, 480), width="stretch", theme=None)
        with b:
            st.markdown("#### Resumo do pareamento")
            st.metric("Pares completos", f"{len(paired):,}".replace(",", "."))
            st.metric("Ganho médio", f"{paired['ganho'].mean():+.2f} pontos")
            st.metric("Ganho mediano", f"{paired['ganho'].median():+.2f} pontos")
            st.caption(
                "Ganho = total do pós-teste − total do pré-teste. Descrição pareada não "
                "implica efeito causal."
            )
    st.subheader("Pontuação por série")
    if _decomposicao_protegida(df, ["serie", "periodo"]):
        por_serie = _rotular_momento(
            df.groupby(["serie", "periodo"], dropna=False)[coluna]
            .mean()
            .rename("media")
            .reset_index()
        )
        fig = px.bar(
            por_serie,
            x="serie",
            y="media",
            color="Momento",
            barmode="group",
            color_discrete_map={"Pré-teste": TEAL, "Pós-teste": CORAL},
            labels={"serie": "Série", "media": "Média (pontos)"},
            title="Média por série e momento",
        )
        fig.update_yaxes(range=[0, teto])
        st.plotly_chart(formatar_figura(fig, 440), width="stretch", theme=None)


def pagina_demografia(dados: dict, filtros: dict) -> None:
    cabecalho(
        "Contexto dos estudantes",
        "Distribuições agregadas da dimensão demográfica e contextual minimizada.",
    )
    st.markdown(
        f'<div class="wg-note">Diferenças são descritivas e não indicam causalidade, '
        "capacidade ou potencial. A política de privacidade exige pelo menos "
        f"{LIMITE_MINIMO_ESTUDANTES} estudantes distintos em cada categoria; se uma categoria "
        "não atingir o limite, toda a decomposição é ocultada.</div>",
        unsafe_allow_html=True,
    )
    if filtros["instrumento"] == "Todos":
        st.caption(
            "As escalas dos instrumentos são distintas; escolha uma base para esta análise. "
            "Os instrumentos não são mesclados silenciosamente."
        )
        nome = st.radio(
            "Instrumento da análise contextual",
            list(INSTRUMENTOS),
            horizontal=True,
            key="instrumento_contexto",
        )
    else:
        nome = filtros["instrumento"]
        st.caption(f"Base selecionada: {nome}.")
    chave, coluna, teto, _ = INSTRUMENTOS[nome]
    aval = filtrar_avaliacoes(dados[chave], **filtros["dimensoes"])
    if aval.empty:
        sem_dados()
        return
    if not _recorte_protegido(aval):
        return
    atributo_padrao = filtros.get("categoria_analitica", "Sexo")
    atributo_por_rotulo = {
        "Sexo": "sexo",
        "Faixa etária": "faixa_etaria",
        "Cor/raça": "cor_raca",
        "Zona": "zona",
        "Necessidade especial": "necessidade_especial",
    }
    atributo = st.selectbox(
        "Característica",
        DEMOGRAFICAS,
        index=DEMOGRAFICAS.index(atributo_por_rotulo.get(atributo_padrao, "sexo")),
        format_func=lambda x: {
            "sexo": "Sexo",
            "faixa_etaria": "Faixa etária",
            "cor_raca": "Cor/raça",
            "zona": "Zona",
            "necessidade_especial": "Necessidade especial registrada",
        }[x],
    )
    resumo = agregar_demografia(aval, dados["dim_aluno"], atributo, coluna)
    if resumo.empty:
        st.info(
            "A decomposição foi suprimida para evitar divulgação de grupos pequenos ou "
            "inferência por categorias complementares."
        )
        return
    resumo = resumo.sort_values("n_estudantes", ascending=False)
    a, b = st.columns([1.1, 1])
    with a:
        fig = px.bar(
            resumo,
            x=atributo,
            y="n_estudantes",
            color_discrete_sequence=[TEAL],
            labels={atributo: "Categoria", "n_estudantes": "Estudantes distintos"},
            title="Cobertura segura no recorte",
        )
        st.plotly_chart(formatar_figura(fig, 460), width="stretch", theme=None)
    with b:
        fig = px.bar(
            resumo,
            x=atributo,
            y="media",
            color_discrete_sequence=[CORAL],
            labels={atributo: "Categoria", "media": "Média (pontos)"},
            title="Média por categoria",
        )
        fig.update_yaxes(range=[0, teto])
        st.plotly_chart(formatar_figura(fig, 460), width="stretch", theme=None)
    st.caption(
        "As contagens representam estudantes distintos, não aplicações. Identificadores "
        "individuais não são exibidos."
    )
    with st.expander("Panorama das categorias contextuais"):
        st.caption(
            "Distribuições por estudantes distintos na base selecionada. Cada gráfico mantém "
            "a mesma regra de supressão complementar."
        )
        rotulos = {
            "sexo": "Sexo",
            "faixa_etaria": "Faixa etária",
            "cor_raca": "Cor/raça",
            "zona": "Zona",
            "necessidade_especial": "Necessidade especial registrada",
        }
        for atributo_panorama in DEMOGRAFICAS:
            panorama = agregar_demografia(aval, dados["dim_aluno"], atributo_panorama, coluna)
            if panorama.empty:
                st.info(f"{rotulos[atributo_panorama]}: distribuição suprimida por privacidade.")
                continue
            fig = px.bar(
                panorama.sort_values("n_estudantes", ascending=False),
                x=atributo_panorama,
                y="n_estudantes",
                color_discrete_sequence=[TEAL],
                labels={
                    atributo_panorama: rotulos[atributo_panorama],
                    "n_estudantes": "Estudantes distintos",
                },
                title=f"Distribuição por {rotulos[atributo_panorama].lower()}",
            )
            st.plotly_chart(formatar_figura(fig, 400), width="stretch", theme=None)


def pagina_qualidade(dados: dict, filtros: dict) -> None:
    cabecalho(
        "Qualidade e cobertura",
        "Indicadores agregados calculados sobre as bases finais do pipeline de mineração.",
    )
    instrumentos = (
        list(INSTRUMENTOS) if filtros["instrumento"] == "Todos" else [filtros["instrumento"]]
    )
    for nome in instrumentos:
        chave, _, _, itens = INSTRUMENTOS[nome]
        df = filtrar_avaliacoes(dados[chave], **filtros["dimensoes"])
        st.subheader(nome)
        if df.empty:
            sem_dados()
            continue
        if not _recorte_protegido(df):
            continue
        _metricas_topo(df, itens)
        if _decomposicao_protegida(df, ["fase", "periodo"]):
            por_fase = _rotular_momento(
                df.assign(_completo=normalizar_booleano(df["completo"]))
                .groupby(["fase", "periodo"], dropna=False)
                .agg(
                    avaliacoes=("aluno_id", "size"),
                    completas=("_completo", lambda s: s.eq(True).sum()),
                    completude_media=("pct_completude", "mean"),
                )
                .reset_index()
            )
            por_fase["fase"] = "Fase " + por_fase["fase"].astype(str)
            fig = px.bar(
                por_fase,
                x="fase",
                y="avaliacoes",
                color="Momento",
                barmode="group",
                color_discrete_map={"Pré-teste": TEAL, "Pós-teste": CORAL},
                labels={"fase": "Fase", "avaliacoes": "Registros"},
                title="Registros por fase e momento",
            )
            st.plotly_chart(formatar_figura(fig, 420), width="stretch", theme=None)
            with st.expander("Resumo tabular de completude"):
                st.dataframe(
                    por_fase.rename(
                        columns={
                            "completas": "Avaliações completas",
                            "completude_media": "Completude média",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
    st.subheader("Relatório consolidado do pipeline")
    if any(filtros["dimensoes"].values()) or filtros["instrumento"] != "Todos":
        st.caption(
            "O relatório abaixo é consolidado para toda a base e não é recalculado pelos "
            "filtros acima."
        )
    if dados["relatorio_qualidade"]:
        st.markdown(dados["relatorio_qualidade"])
    else:
        st.warning("Relatório de qualidade não encontrado em data/final.")


def pagina_effect_size(dados: dict, instrumento: str = "Todos") -> None:
    """Exibe métricas pareadas previamente calculadas no notebook analítico."""
    cabecalho(
        "Comparação pré e pós",
        "Magnitude padronizada da diferença nos mesmos estudantes com pares completos.",
    )
    metricas = dados["effect_size"].copy()
    if metricas.empty:
        sem_dados("Arquivo de métricas de Effect Size indisponível em data/analytics.")
        return
    metricas = metricas[metricas["Filtro"].eq("Completas (>=80%)")]
    if instrumento != "Todos":
        nome = "TDE II" if instrumento == "TDE II · Escrita" else instrumento
        metricas = metricas[metricas["Instrumento"].eq(nome)]
    if metricas.empty:
        sem_dados()
        return
    st.markdown(
        '<div class="wg-note"><strong>Effect Size pareado (Cohen’s dz)</strong> resume a '
        "diferença média pré–pós em unidades do desvio-padrão das diferenças individuais. "
        "Ele complementa, mas não substitui, a diferença em pontos. O IC 95% indica a "
        "incerteza da estimativa; significância estatística não comprova relevância educacional "
        "nem causalidade.</div>",
        unsafe_allow_html=True,
    )
    tabela = metricas[
        [
            "Instrumento",
            "Fase",
            "N Pares",
            "Média Pré",
            "Média Pós",
            "Ganho Médio",
            "d_z (Cohen)",
            "IC 95% Inf",
            "IC 95% Sup",
            "t (Student)",
            "p-valor (t)",
            "Interpretação",
        ]
    ].copy()
    st.subheader("Resultados por fase")
    st.dataframe(tabela, hide_index=True, width="stretch", height=250)
    nomes = metricas["Instrumento"].drop_duplicates().tolist()
    abas = st.tabs(nomes) if len(nomes) > 1 else [st.container()]
    for nome_instrumento, aba in zip(nomes, abas, strict=True):
        sub = metricas[metricas["Instrumento"].eq(nome_instrumento)].copy()
        sub["Fase"] = pd.Categorical(sub["Fase"], ["Fase 2", "Fase 3", "Fase 4"], ordered=True)
        with aba:
            st.subheader(nome_instrumento)
            medias = sub.melt(
                id_vars="Fase",
                value_vars=["Média Pré", "Média Pós"],
                var_name="Momento",
                value_name="Pontuação média",
            )
            fig = px.bar(
                medias,
                x="Fase",
                y="Pontuação média",
                color="Momento",
                barmode="group",
                color_discrete_map={"Média Pré": TEAL, "Média Pós": CORAL},
                title="Médias pré e pós nos pares completos",
            )
            st.plotly_chart(formatar_figura(fig, 480), width="stretch", theme=None)
            sub = sub.assign(
                erro_ic_superior=sub["IC 95% Sup"] - sub["d_z (Cohen)"],
                erro_ic_inferior=sub["d_z (Cohen)"] - sub["IC 95% Inf"],
            )
            fig = px.scatter(
                sub.sort_values("Fase", ascending=False),
                x="d_z (Cohen)",
                y="Fase",
                error_x="erro_ic_superior",
                error_x_minus="erro_ic_inferior",
                labels={"d_z (Cohen)": "Cohen's dz", "Fase": ""},
                title="Effect Size pareado e intervalo de confiança de 95%",
                color_discrete_sequence=[TEAL if nome_instrumento == "Vocabulário" else CORAL],
            )
            fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
            fig.update_yaxes(categoryorder="array", categoryarray=["Fase 4", "Fase 3", "Fase 2"])
            st.plotly_chart(formatar_figura(fig, 450), width="stretch", theme=None)
    st.caption(
        "Referências usuais de magnitude são contextuais; o arquivo analítico preserva a "
        "interpretação adotada no notebook. Resultados representam pares completos, não "
        "todas as aplicações."
    )


def pagina_longitudinal(dados: dict, instrumento: str = "Todos") -> None:
    """Apresenta transições agregadas pré-computadas para coortes acompanhadas."""
    cabecalho(
        "Acompanhamento longitudinal",
        "Transições entre fases para estudantes que possuem registro nos dois momentos comparados.",
    )
    metricas = dados["longitudinais"].copy()
    painel = dados["painel_longitudinal"].copy()
    if metricas.empty or painel.empty:
        sem_dados("Artefatos longitudinais indisponíveis em data/analytics.")
        return
    nome = "TDE II" if instrumento == "TDE II · Escrita" else instrumento
    if nome != "Todos":
        metricas = metricas[metricas["Instrumento"].eq(nome)]
        painel = painel[painel["instrumento"].eq(nome)]
    transicoes = metricas[
        metricas["Tipo_Metrica"].eq("Transicao_Interfases")
        & metricas["Critério_Amostra"].eq("Completude Estrita (>=80% itens)")
    ].copy()
    if transicoes.empty:
        sem_dados("Não há transições completas para o recorte selecionado.")
        return
    st.markdown(
        '<div class="wg-note">Cada transição compara o <strong>mesmo estudante</strong> no '
        "pós-teste de uma fase e no pré-teste da fase seguinte. Ausência de registro não é "
        "interpretada como queda de desempenho; os resultados refletem somente a coorte "
        "acompanhada.</div>",
        unsafe_allow_html=True,
    )
    coortes = pd.DataFrame(
        {
            "Coorte": ["Fase 2 → Fase 3", "Fase 3 → Fase 4", "Três fases"],
            "Estudantes": [
                int(painel["cohort_f2_f3"].eq(True).sum()),
                int(painel["cohort_f3_f4"].eq(True).sum()),
                int(painel["cohort_continuo_3f"].eq(True).sum()),
            ],
        }
    )
    cols = st.columns(3)
    for col, linha in zip(cols, coortes.itertuples(index=False), strict=True):
        col.metric(
            linha.Coorte,
            _formatar_numero(linha.Estudantes),
            help="Estudantes distintos no painel longitudinal do instrumento selecionado.",
        )
    transicoes["Transição curta"] = transicoes["Transição"].replace(
        {
            "Fase 2 Pós -> Fase 3 Pré": "Fase 2 → Fase 3",
            "Fase 3 Pós -> Fase 4 Pré": "Fase 3 → Fase 4",
        }
    )
    nomes = transicoes["Instrumento"].drop_duplicates().tolist()
    abas = st.tabs(nomes) if len(nomes) > 1 else [st.container()]
    for nome_instrumento, aba in zip(nomes, abas, strict=True):
        sub = transicoes[transicoes["Instrumento"].eq(nome_instrumento)].copy()
        with aba:
            st.subheader(nome_instrumento)
            transicoes_longas = sub.melt(
                id_vars="Transição curta",
                value_vars=["Média Pós Anterior", "Média Pré Seguinte"],
                var_name="Momento",
                value_name="Pontuação média",
            )
            fig = px.bar(
                transicoes_longas,
                y="Transição curta",
                x="Pontuação média",
                color="Momento",
                orientation="h",
                barmode="group",
                color_discrete_map={"Média Pós Anterior": TEAL, "Média Pré Seguinte": CORAL},
                labels={"Transição curta": "Transição"},
                title="Pontuação média nas transições acompanhadas",
            )
            fig.update_yaxes(
                categoryorder="array", categoryarray=["Fase 3 → Fase 4", "Fase 2 → Fase 3"]
            )
            st.plotly_chart(formatar_figura(fig, 430), width="stretch", theme=None)
            fig = px.bar(
                sub,
                y="Transição curta",
                x="N Pares",
                orientation="h",
                color_discrete_sequence=[TEAL if nome_instrumento == "Vocabulário" else CORAL],
                labels={"Transição curta": "Transição", "N Pares": "Estudantes acompanhados"},
                title="Estudantes acompanhados por transição",
            )
            fig.update_yaxes(
                categoryorder="array", categoryarray=["Fase 3 → Fase 4", "Fase 2 → Fase 3"]
            )
            st.plotly_chart(formatar_figura(fig, 400), width="stretch", theme=None)
    st.subheader("Variação observada entre fases")
    tabela_longitudinal = transicoes[
        [
            "Instrumento",
            "Transição curta",
            "N Pares",
            "Média Pós Anterior",
            "Média Pré Seguinte",
            "Variação Média (Delta)",
            "p-valor",
        ]
    ].rename(
        columns={
            "Transição curta": "Transição",
            "Média Pós Anterior": "Pós anterior",
            "Média Pré Seguinte": "Pré seguinte",
            "Variação Média (Delta)": "Variação média",
        }
    )
    st.dataframe(
        tabela_longitudinal,
        hide_index=True,
        width="stretch",
        height=180,
    )


def pagina_modelagem(dados: dict, instrumento: str = "Todos") -> None:
    cabecalho(
        "Experimentos de modelagem",
        "Resultados reportados em teste, com validação cruzada exibida separadamente.",
    )
    st.markdown(
        '<div class="wg-note">Resultados experimentais, não validados para decisões '
        "pedagógicas reais. Experimento B utiliza características demográficas/contextuais "
        "cuja disponibilidade temporal por observação ainda não foi comprovada.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "O alvo é `ganho = total_pos − total_pre`. Foram treinados modelos separados para "
        "Vocabulário e TDE II: o **Experimento A** usa atributos acadêmicos disponíveis antes "
        "do pós-teste; o **B** adiciona atributos demográficos/contextuais de caráter exploratório."
    )
    with st.expander("Como interpretar a comparação"):
        st.markdown(
            "- **Baseline 1:** prevê ganho zero; **Baseline 2:** prevê a média do ganho no "
            "treino.\n"
            "- **Random Forest** e **MLP** são comparados prioritariamente por MAE no teste; "
            "menor é melhor. RMSE penaliza erros grandes e R² resume variação explicada.\n"
            "- A validação cruzada agrupada estima estabilidade no treino; ela não é métrica "
            "do teste."
        )
    metricas = dados["metricas"].copy()
    if metricas.empty:
        sem_dados("Arquivo de métricas comparativas indisponível.")
        return
    selecionados = (
        metricas["Cenário"].str.split(" — ").str[0].unique().tolist()
        if instrumento == "Todos"
        else [instrumento]
    )
    metricas = metricas[metricas["Cenário"].str.split(" — ").str[0].isin(selecionados)]
    if metricas.empty:
        sem_dados()
        return
    melhores = resumo_modelagem(metricas)
    if not melhores.empty:
        st.subheader("Resumo executivo por instrumento e cenário")
        resumo = melhores.rename(
            columns={
                "MAE": "MAE teste",
                "R²": "R² teste",
                "Reducao MAE (%)": "Redução vs. baseline (%)",
            }
        )
        st.dataframe(
            resumo[
                [
                    "Instrumento",
                    "Experimento",
                    "Modelo",
                    "MAE teste",
                    "R² teste",
                    "MAE baseline",
                    "Redução vs. baseline (%)",
                ]
            ].round(4),
            hide_index=True,
            width="stretch",
        )
        melhor_por_instrumento = melhores.loc[melhores.groupby("Instrumento")["MAE"].idxmin()]
        textos = []
        for _, linha in melhor_por_instrumento.iterrows():
            reducao = linha["Reducao MAE (%)"]
            reducao_texto = "não disponível" if pd.isna(reducao) else f"{reducao:.1f}%"
            textos.append(
                f"**{linha['Instrumento']}:** menor MAE de teste no {linha['Modelo']} do "
                f"Experimento {linha['Experimento']} ({linha['MAE']:.2f}); variação frente ao "
                f"baseline da média: {reducao_texto}."
            )
        st.markdown("  \n".join(textos))
    st.caption(
        "MAE, RMSE e R² abaixo são métricas do conjunto de teste. O baseline da média usa "
        "somente o treino. Os filtros demográficos e escolares não se aplicam aos artefatos "
        "persistidos."
    )
    mostrar = metricas.rename(columns={"MAE": "MAE teste", "RMSE": "RMSE teste", "R²": "R² teste"})
    mostrar["Cenário"] = (
        mostrar["Cenário"]
        .str.replace(" — Experimento ", " · Exp ", regex=False)
        .str.replace(" (Acadêmico + Demográfico/Contextual)", "", regex=False)
        .str.replace(" (Acadêmico + Demográfico)", "", regex=False)
        .str.replace(" (Acadêmico)", "", regex=False)
    )
    st.dataframe(
        mostrar[["Cenário", "Modelo", "MAE teste", "RMSE teste", "R² teste"]],
        hide_index=True,
        width="stretch",
        height=min(420, 74 + 36 * len(mostrar)),
    )
    plot = metricas[
        metricas["Modelo"].isin(
            [
                "Baseline 1 (Ganho Zero)",
                "Baseline 2 (Média Treino)",
                "Random Forest",
                "Rede Neural (MLP)",
            ]
        )
    ]
    modelos = plot[plot["Modelo"].isin(["Random Forest", "Rede Neural (MLP)"])]
    plot = plot.assign(
        Instrumento=plot["Cenário"].str.split(" — ").str[0],
        Experimento=plot["Cenário"]
        .str.extract(r"Experimento ([AB])", expand=False)
        .map({"A": "Exp A", "B": "Exp B"}),
    )
    modelos = plot[plot["Modelo"].isin(["Random Forest", "Rede Neural (MLP)"])]
    comparacao = modelos.merge(
        plot[plot["Modelo"].eq("Baseline 2 (Média Treino)")][
            ["Instrumento", "Experimento", "MAE"]
        ].rename(columns={"MAE": "MAE baseline"}),
        on=["Instrumento", "Experimento"],
        how="left",
        validate="many_to_one",
    )
    comparacao["Redução de MAE (%)"] = (
        (comparacao["MAE baseline"] - comparacao["MAE"]) / comparacao["MAE baseline"] * 100
    )
    st.subheader("Comparação visual por instrumento")
    st.caption("Exp A: atributos acadêmicos. Exp B: acadêmicos + demográficos/contextuais.")
    nomes = plot["Instrumento"].drop_duplicates().tolist()
    abas = st.tabs(nomes) if len(nomes) > 1 else [st.container()]
    for nome_instrumento, aba in zip(nomes, abas, strict=True):
        sub_plot = plot[plot["Instrumento"].eq(nome_instrumento)]
        sub_modelos = modelos[modelos["Instrumento"].eq(nome_instrumento)]
        sub_comparacao = comparacao[comparacao["Instrumento"].eq(nome_instrumento)]
        with aba:
            st.subheader(nome_instrumento)
            st.caption("Menor MAE indica menor erro médio na escala do respectivo instrumento.")
            fig = px.bar(
                sub_plot,
                y="Modelo",
                x="MAE",
                color="Modelo",
                orientation="h",
                facet_col="Experimento",
                color_discrete_map={
                    "Baseline 1 (Ganho Zero)": MUTED,
                    "Baseline 2 (Média Treino)": GOLD,
                    "Random Forest": TEAL,
                    "Rede Neural (MLP)": CORAL,
                },
                labels={"MAE": "MAE no teste", "Modelo": ""},
                title="MAE por experimento: modelos e baselines",
            )
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=[
                    "Rede Neural (MLP)",
                    "Random Forest",
                    "Baseline 2 (Média Treino)",
                    "Baseline 1 (Ganho Zero)",
                ],
            )
            st.plotly_chart(formatar_figura(fig, 500), width="stretch", theme=None)
            fig = px.bar(
                sub_modelos,
                y="Modelo",
                x="R²",
                color="Modelo",
                orientation="h",
                facet_col="Experimento",
                color_discrete_map={"Random Forest": TEAL, "Rede Neural (MLP)": CORAL},
                labels={"R²": "R² no teste", "Modelo": ""},
                title="R² dos modelos no conjunto de teste",
            )
            fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
            st.plotly_chart(formatar_figura(fig, 440), width="stretch", theme=None)
            fig = px.bar(
                sub_comparacao,
                y="Modelo",
                x="Redução de MAE (%)",
                color="Modelo",
                orientation="h",
                facet_col="Experimento",
                color_discrete_map={"Random Forest": TEAL, "Rede Neural (MLP)": CORAL},
                labels={"Redução de MAE (%)": "Redução do MAE (%)", "Modelo": ""},
                title="Variação percentual do MAE versus baseline da média",
            )
            fig.add_vline(x=0, line_dash="dash", line_color=MUTED)
            st.plotly_chart(formatar_figura(fig, 440), width="stretch", theme=None)
    with st.expander("Validação cruzada e metadados"):
        associados, avisos = associar_metadados_modelagem(
            dados["metadados_modelos"], metricas, instrumento
        )
        if associados.empty:
            st.info(
                "Não há metadados associados de forma segura ao instrumento e aos cenários "
                "exibidos."
            )
        else:
            st.dataframe(
                associados.round({"MAE teste": 4, "RMSE teste": 4, "R² teste": 4, "CV MAE": 4}),
                hide_index=True,
                width="stretch",
                height=min(420, 74 + 36 * len(associados)),
            )
        if avisos:
            st.warning(
                "Há metadados sem associação segura com métricas persistidas; eles foram ocultados."
            )
