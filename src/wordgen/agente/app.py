"""Interface Streamlit independente para o agente analítico WordGen."""

from __future__ import annotations

import streamlit as st

from wordgen.agente.orquestrador import RespostaAgente, construir_agente
from wordgen.dashboard.visual import aplicar_tema


@st.cache_resource(show_spinner=False)
def _obter_agente():
    return construir_agente()


def _mostrar_resposta(resposta: RespostaAgente) -> None:
    st.markdown(resposta.texto)
    if resposta.aviso:
        st.info(resposta.aviso)
    if resposta.fontes:
        st.caption("Fontes consultadas: " + " · ".join(fonte.nome for fonte in resposta.fontes))


def renderizar_assistente(exibir_sidebar: bool = False) -> None:
    """Renderiza o assistente dentro do dashboard ou como aplicação independente."""
    st.title("Assistente analítico WordGen")
    st.caption(
        "Pergunte sobre resultados agregados, metodologia, instrumentos, análises e modelagem. "
        "O agente não acessa registros individuais."
    )

    if exibir_sidebar:
        with st.sidebar:
            st.subheader("Como funciona")
            st.write("Métricas quantitativas usam artefatos publicados; conceitos usam RAG.")
            st.caption(
                "Respostas são experimentais e não substituem avaliação pedagógica profissional."
            )
            limpar = st.button("Limpar conversa", use_container_width=True)
    else:
        limpar = st.button("Limpar conversa", type="secondary")
    if limpar:
        st.session_state.pop("historico_agente", None)
        st.rerun()

    if "historico_agente" not in st.session_state:
        st.session_state.historico_agente = []

    try:
        agente = _obter_agente()
    except (FileNotFoundError, OSError, ValueError) as erro:
        st.error(f"Não foi possível preparar a base de conhecimento: {erro}")
        st.info("Confirme os artefatos privados configurados para a aplicação e tente novamente.")
        st.stop()

    for pergunta, resposta in st.session_state.historico_agente:
        with st.chat_message("user"):
            st.write(pergunta)
        with st.chat_message("assistant"):
            _mostrar_resposta(resposta)

    pergunta = st.chat_input("Ex.: Como foi feita a validação dos modelos?")
    if pergunta:
        with st.chat_message("user"):
            st.write(pergunta)
        with st.chat_message("assistant"):
            with st.spinner("Consultando fontes publicadas..."):
                resposta = agente.responder(pergunta)
            _mostrar_resposta(resposta)
        st.session_state.historico_agente.append((pergunta, resposta))


def main() -> None:
    st.set_page_config(page_title="WordGen · Assistente", page_icon="WG", layout="wide")
    aplicar_tema(st)
    renderizar_assistente(exibir_sidebar=True)


if __name__ == "__main__":
    main()
