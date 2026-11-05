"""Aplicação Streamlit do Dashboard Analítico WordGen."""

import streamlit as st

from wordgen.agente.app import renderizar_assistente
from wordgen.dashboard.dados import carregar_dados
from wordgen.dashboard.paginas import (
    INSTRUMENTOS,
    pagina_demografia,
    pagina_effect_size,
    pagina_instrumento,
    pagina_longitudinal,
    pagina_modelagem,
    pagina_qualidade,
    visao_geral,
)
from wordgen.dashboard.visual import aplicar_tema

st.set_page_config(
    page_title="WordGen · Painel Analítico",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
aplicar_tema(st)


@st.cache_data(show_spinner="Carregando bases consolidadas…")
def obter_dados():
    return carregar_dados()


with st.sidebar:
    st.markdown('<div class="wg-brand">word<span>gen</span></div>', unsafe_allow_html=True)
    st.caption("ANÁLISE EDUCACIONAL · UFPA")
    st.divider()
    pagina = st.radio(
        "Explorar",
        [
            "Visão geral",
            "Vocabulário",
            "TDE II · Escrita",
            "Contexto dos estudantes",
            "Comparação pré e pós",
            "Acompanhamento longitudinal",
            "Qualidade dos dados",
            "Modelagem",
            "Assistente analítico",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    if pagina == "Assistente analítico":
        st.caption("O assistente usa somente fontes agregadas e documentação publicada.")

if pagina == "Assistente analítico":
    renderizar_assistente()
    st.stop()

try:
    data = obter_dados()
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Não foi possível carregar as bases finais: {exc}")
    st.info(
        "Configure os artefatos privados de produção e reinicie a aplicação. "
        "Consulte o README para os diretórios exigidos."
    )
    st.stop()

if data.get("modo_demonstracao"):
    st.warning(
        "Modo de demonstração: esta versão usa dados inteiramente sintéticos, "
        "criados somente para navegação pública. Nenhum dado de estudante real é exibido."
    )

with st.sidebar:
    st.markdown("**Recorte das avaliações**")
    filtro_instrumento = st.selectbox(
        "Instrumento",
        ["Todos", *INSTRUMENTOS.keys()],
        disabled=pagina in ["Vocabulário", "TDE II · Escrita"],
    )
    bases = (
        [data["vocabulario"], data["tde"]]
        if filtro_instrumento == "Todos"
        else [data[INSTRUMENTOS[filtro_instrumento][0]]]
    )

    def all_values(col: str) -> list[str]:
        return sorted(set().union(*(set(df[col].dropna().astype(str)) for df in bases)))

    categoria_analitica = "Sexo"
    if pagina != "Modelagem":
        fases = st.multiselect(
            "Fase", all_values("fase"), format_func=lambda x: f"Fase {x.removeprefix('Fase ')}"
        )
        momentos = st.multiselect(
            "Momento",
            all_values("periodo"),
            format_func=lambda x: "Pré-teste" if x == "pre" else "Pós-teste" if x == "pos" else x,
        )
        series = st.multiselect("Série", all_values("serie"))
        escolas = st.multiselect("Escola", all_values("escola"))
        turmas = st.multiselect("Turma", all_values("turma"))
        categoria_analitica = st.selectbox(
            "Categoria analítica",
            ["Sexo", "Faixa etária", "Cor/raça", "Zona", "Necessidade especial"],
            help=(
                "Não há categoria de desempenho metodologicamente definida nas bases. "
                "Este seletor orienta apenas a dimensão exibida na página de Contexto."
            ),
        )
    else:
        fases = momentos = series = escolas = turmas = []
    st.divider()
    st.caption("Dados agregados. Identificadores e trilhas técnicas não são exibidos.")

dimensoes = {
    "fases": fases,
    "momentos": momentos,
    "series": series,
    "escolas": escolas,
    "turmas": turmas,
}
filtros = {
    "instrumento": filtro_instrumento,
    "dimensoes": dimensoes,
    "categoria_analitica": categoria_analitica,
}

if pagina == "Visão geral":
    visao_geral(data, filtros)
elif pagina in INSTRUMENTOS:
    pagina_instrumento(data, pagina, filtros)
elif pagina == "Contexto dos estudantes":
    pagina_demografia(data, filtros)
elif pagina == "Comparação pré e pós":
    pagina_effect_size(data, filtro_instrumento)
elif pagina == "Acompanhamento longitudinal":
    pagina_longitudinal(data, filtro_instrumento)
elif pagina == "Qualidade dos dados":
    pagina_qualidade(data, filtros)
else:
    pagina_modelagem(data, filtro_instrumento)
