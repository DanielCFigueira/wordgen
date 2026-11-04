"""Testes locais do agente RAG e das consultas analíticas agregadas."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from wordgen.agente.avaliacao import avaliar_benchmark, resumo_benchmark
from wordgen.agente.config import ConfiguracaoAgente
from wordgen.agente.consultas_analiticas import ConsultasAnaliticas
from wordgen.agente.ingestao import DocumentoConhecimento, carregar_base_conhecimento
from wordgen.agente.llm import ClienteGroq, LLMIndisponivel
from wordgen.agente.orquestrador import construir_agente
from wordgen.agente.vetor_store import IndiceVetorial


def test_carrega_base_conhecimento_publicada() -> None:
    documentos = carregar_base_conhecimento(ConfiguracaoAgente().base_conhecimento)
    assert len(documentos) == 10
    assert documentos[0].identificador == "wordgen_visao_geral"
    assert "estudantes" in documentos[0].conteudo.lower()


def test_indice_recupera_documento_e_preserva_fonte(tmp_path: Path) -> None:
    documentos = [
        DocumentoConhecimento(
            "metodo", "Metodologia", "metodo", "Cohen dz mede efeito pareado.", (), {}
        ),
        DocumentoConhecimento("dados", "Dados", "dados", "Vocabulário possui 50 itens.", (), {}),
    ]
    indice = IndiceVetorial.carregar_ou_construir(documentos, tmp_path / "indice.joblib")
    resultado = indice.buscar("O que é Cohen dz?")
    assert resultado[0][0].identificador == "metodo"
    assert resultado[0][0].fonte.endswith("metodo")
    assert (tmp_path / "indice.joblib").exists()


def test_indice_reconstroi_quando_conteudo_muda(tmp_path: Path) -> None:
    caminho = tmp_path / "indice.joblib"
    primeiro = [DocumentoConhecimento("um", "Um", "x", "texto inicial", (), {})]
    segundo = [DocumentoConhecimento("um", "Um", "x", "texto atualizado", (), {})]
    IndiceVetorial.carregar_ou_construir(primeiro, caminho)
    indice = IndiceVetorial.carregar_ou_construir(segundo, caminho)
    assert indice.documentos[0].conteudo == "texto atualizado"


def test_consulta_modelagem_usa_artefato_real() -> None:
    resultado = ConsultasAnaliticas().consultar(
        "Qual foi o MAE do Random Forest no Experimento B para TDE II?"
    )
    assert resultado is not None
    assert resultado.tipo == "modelagem"
    assert "TDE II" in resultado.resposta
    assert "Random Forest" in resultado.resposta
    assert resultado.fontes == ("reports/modelagem/metricas/metricas_comparativas_gerais.csv",)


def test_consulta_effect_size_usa_arquivo_agregado() -> None:
    resultado = ConsultasAnaliticas().consultar(
        "Qual foi o Effect Size de Cohen dz do Vocabulário na Fase 2?"
    )
    assert resultado is not None
    assert resultado.tipo == "effect_size"
    assert "0.168" in resultado.resposta


def test_consulta_longitudinal_tem_precedencia_sobre_pre_pos() -> None:
    resultado = ConsultasAnaliticas().consultar(
        "O que aconteceu entre o pós-teste da Fase 2 e o pré-teste da Fase 3 em Vocabulário?"
    )
    assert resultado is not None
    assert resultado.tipo == "longitudinal"
    assert "Fase 2 Pós -> Fase 3 Pré" in resultado.resposta


def test_consulta_kpi_nao_soma_instrumentos() -> None:
    resultado = ConsultasAnaliticas().consultar("Quantos estudantes únicos participaram no total?")
    assert resultado is not None
    assert "2,670" in resultado.resposta
    assert "união" in resultado.resposta.lower()


def test_consulta_escolas_e_turmas_distingue_cobertura_e_total() -> None:
    resultado = ConsultasAnaliticas().consultar(
        "Quantas escolas públicas municipais e turmas participaram da intervenção?"
    )
    assert resultado is not None
    assert "6 instituições" in resultado.resposta
    assert "84 turmas" in resultado.resposta
    assert "92 turmas" in resultado.resposta


def test_orquestrador_recusa_identificadores_individuais() -> None:
    resposta = construir_agente().responder("Qual é o aluno_id do estudante com maior ganho?")
    assert resposta.rota == "privacidade"
    assert not resposta.fontes


def test_orquestrador_responde_rag_sem_credencial(monkeypatch: pytest.MonkeyPatch) -> None:
    resposta = construir_agente().responder("O que é Cohen's dz?", usar_llm=False)
    assert resposta.rota == "rag_extrativo"
    assert resposta.fontes
    assert "Cohen" in resposta.texto


def test_cliente_groq_sem_chave_informa_configuracao(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wordgen.agente.llm.obter_chave_groq", lambda: None)
    with pytest.raises(LLMIndisponivel, match="GROQ_API_KEY"):
        ClienteGroq().gerar("pergunta", "contexto")


def test_benchmark_tem_cobertura_offline() -> None:
    configuracao = ConfiguracaoAgente()
    resultados = avaliar_benchmark(construir_agente(configuracao), configuracao.benchmark)
    resumo = resumo_benchmark(resultados)
    assert resumo["total"] == 12
    assert resumo["respondidas"] == 12
    assert resumo["com_fontes"] == 12
    assert resumo["aprovadas"] == 12


@pytest.mark.parametrize(
    "pergunta",
    [
        "Qual é a capital da França?",
        "Como preparar uma receita de bolo?",
        "Quem venceu a Copa do Mundo de 2002?",
    ],
)
def test_orquestrador_recusa_perguntas_fora_do_escopo(pergunta: str) -> None:
    resposta = construir_agente().responder(pergunta, usar_llm=False)
    assert resposta.rota == "fora_escopo"
    assert not resposta.fontes


def test_orquestrador_pede_esclarecimento_para_pergunta_ambigua() -> None:
    resposta = construir_agente().responder("Quais dados foram usados?", usar_llm=False)
    assert resposta.rota == "esclarecimento_escopo"


def test_orquestrador_admite_ausencia_de_evidencia_para_renda() -> None:
    resposta = construir_agente().responder(
        "Qual foi a renda média familiar dos estudantes participantes?", usar_llm=False
    )
    assert resposta.rota == "sem_evidencia"
    assert not resposta.fontes


@pytest.mark.parametrize(
    ("pergunta", "trecho_esperado"),
    [
        ("Qual é o critério técnico para considerar uma avaliação pareada completa?", "80%"),
        ("Como foi feita a validação dos modelos?", "GroupShuffleSplit"),
        ("Por que o Experimento B é considerado exploratório?", "múltiplas fases"),
        (
            "Quais são as limitações da utilização dos modelos para decisões pedagógicas?",
            "experimentais",
        ),
    ],
)
def test_rotas_metodologicas_usam_evidencia_especifica(pergunta: str, trecho_esperado: str) -> None:
    resposta = construir_agente().responder(pergunta, usar_llm=False)
    assert resposta.rota == "rag_metodologia"
    assert trecho_esperado.lower() in resposta.texto.lower()
    assert all("wordgen_visao_geral" not in fonte.nome for fonte in resposta.fontes)


def test_interpretacao_cohen_usa_rag_sem_consulta_longitudinal() -> None:
    resposta = construir_agente().responder(
        "O que significa um Cohen dz positivo na comparação entre pré e pós-teste?", usar_llm=False
    )
    assert resposta.rota == "rag_metodologia"
    assert "pós-teste foi maior" in resposta.texto


def test_retriever_rejeita_chunk_abaixo_do_limiar() -> None:
    indice = construir_agente().indice
    assert indice.buscar("receita de bolo com chocolate") == []


def test_llm_converte_erro_404_em_mensagem_configuravel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Erro404(Exception):
        status_code = 404

    class ClienteFalso:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: (_ for _ in ()).throw(Erro404())
                )
            )

    monkeypatch.setattr("wordgen.agente.llm.obter_chave_groq", lambda: "teste")
    monkeypatch.setitem(sys.modules, "groq", SimpleNamespace(Groq=ClienteFalso))
    with pytest.raises(LLMIndisponivel, match="modelo Groq configurado") as erro:
        ClienteGroq().gerar("pergunta", "contexto")
    assert erro.value.detalhe_tecnico == "Erro404 (HTTP 404)"


def test_dashboard_principal_expoe_navegacao_para_assistente() -> None:
    caminho_app = Path(__file__).resolve().parents[1] / "src" / "wordgen" / "dashboard" / "app.py"
    app = AppTest.from_file(caminho_app).run(timeout=45)
    assert not app.exception
    app.radio[0].set_value("Assistente analítico").run(timeout=45)
    assert not app.exception
    assert app.title[0].value == "Assistente analítico WordGen"
    assert len(app.chat_input) == 1


def test_cliente_groq_aplica_fallback_para_modelo_compativel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cliente = ClienteGroq()
    monkeypatch.setattr("wordgen.agente.llm.obter_modelo_groq", lambda: "modelo-ausente")
    modelo = cliente._resolver_modelo(
        SimpleNamespace(
            models=SimpleNamespace(
                list=lambda: SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-20b")])
            )
        )
    )
    assert modelo == "openai/gpt-oss-20b"
