"""Orquestra recuperação documental, consultas agregadas e geração opcional."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from logging import getLogger

from wordgen.agente.config import ConfiguracaoAgente
from wordgen.agente.consultas_analiticas import ConsultasAnaliticas
from wordgen.agente.ingestao import carregar_base_conhecimento, carregar_documentacao_modelagem
from wordgen.agente.llm import ClienteGroq, LLMIndisponivel
from wordgen.agente.vetor_store import IndiceVetorial

LOGGER = getLogger(__name__)


@dataclass(frozen=True)
class FonteResposta:
    nome: str
    relevancia: float | None = None


@dataclass(frozen=True)
class RespostaAgente:
    texto: str
    rota: str
    fontes: tuple[FonteResposta, ...]
    aviso: str | None = None


_PADROES_IDENTIFICADORES = (
    r"\baluno[_\s-]?id\b",
    r"\bid\s*(do|da)?\s*aluno\b",
    r"\bnome\s*(do|da)?\s*(aluno|estudante|responsavel)\b",
    r"\be-?mail\b",
    r"\bcpf\b",
)
_TERMOS_SEM_EVIDENCIA = ("renda familiar", "renda media", "escolaridade familiar")
_TERMOS_ESCOPO = (
    "wordgen",
    "vocabulario",
    "tde",
    "cohen",
    "effect size",
    "paread",
    "pre-teste",
    "pos-teste",
    "fase ",
    "longitud",
    "random forest",
    "mlp",
    "baseline",
    "validacao dos modelos",
    "validacao cruzada",
    "validacao",
    "experimento a",
    "experimento b",
    "modelos",
    "decisoes pedagogicas",
    "completude",
    "pares completos",
    "estudantes unicos",
    "escola",
    "turma",
    "necessidade especial",
    "necessidades especiais",
    "necessidad",
    "sexo",
    "cor/raca",
    "zona urbana",
    "zona rural",
    "dim_aluno",
    "aluno_id",
    "mae",
    "rmse",
    "r²",
    "r2",
    "ganho de desempenho",
)


def _solicita_dado_individual(pergunta: str) -> bool:
    return any(
        re.search(padrao, pergunta, flags=re.IGNORECASE) for padrao in _PADROES_IDENTIFICADORES
    )


def _classificar_escopo(pergunta: str) -> str:
    """Classificação local e conservadora para impedir RAG fora do domínio WordGen."""
    texto = unicodedata.normalize("NFKD", pergunta.lower())
    texto = "".join(caractere for caractere in texto if not unicodedata.combining(caractere))
    if any(termo in texto for termo in _TERMOS_SEM_EVIDENCIA):
        return "wordgen"
    if any(termo in texto for termo in _TERMOS_ESCOPO):
        return "wordgen"
    if any(termo in texto for termo in ("dados", "analise", "resultado", "modelo", "metodologia")):
        return "ambiguo"
    return "fora_escopo"


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(caractere for caractere in texto if not unicodedata.combining(caractere))


class AgenteWordGen:
    """Agente com rotas fechadas para métricas e RAG fundamentado em fontes publicadas."""

    def __init__(
        self,
        indice: IndiceVetorial,
        consultas: ConsultasAnaliticas,
        llm: ClienteGroq | None = None,
    ) -> None:
        self.indice = indice
        self.consultas = consultas
        self.llm = llm or ClienteGroq()

    def responder(self, pergunta: str, usar_llm: bool = True) -> RespostaAgente:
        pergunta = pergunta.strip()
        if not pergunta:
            return RespostaAgente(
                "Envie uma pergunta sobre os dados, análises, metodologia ou modelagem do WordGen.",
                "esclarecimento",
                (),
            )
        if _solicita_dado_individual(pergunta):
            return RespostaAgente(
                "Por privacidade, o agente não consulta nem divulga identificadores "
                "ou registros individuais. "
                "Posso ajudar com indicadores e análises agregadas publicadas pelo projeto.",
                "privacidade",
                (),
            )
        texto_normalizado = _normalizar(pergunta)
        if any(termo in texto_normalizado for termo in _TERMOS_SEM_EVIDENCIA):
            return RespostaAgente(
                "Não encontrei informações suficientes na base de conhecimento do WordGen "
                "para responder a essa pergunta.",
                "sem_evidencia",
                (),
            )
        escopo = _classificar_escopo(pergunta)
        if escopo == "fora_escopo":
            return RespostaAgente(
                "Essa pergunta está fora do escopo do Assistente WordGen. Posso ajudar com "
                "dados, instrumentos, análises, resultados e metodologia do projeto.",
                "fora_escopo",
                (),
            )
        if escopo == "ambiguo":
            return RespostaAgente(
                "Não consegui determinar se a pergunta se refere ao WordGen. Informe o "
                "instrumento, "
                "a análise ou o resultado do projeto que deseja consultar.",
                "esclarecimento_escopo",
                (),
            )

        resposta_dirigida = self._responder_metodologia(texto_normalizado)
        if resposta_dirigida:
            return resposta_dirigida

        resultado = self.consultas.consultar(pergunta)
        if resultado:
            return RespostaAgente(
                resultado.resposta,
                f"consulta_{resultado.tipo}",
                tuple(FonteResposta(fonte) for fonte in resultado.fontes),
            )

        consulta_rag = pergunta
        if "critério" in pergunta.lower() and "completa" in pergunta.lower():
            consulta_rag += " completude pares 80 por cento itens"
        recuperados = self.indice.buscar(consulta_rag)
        if not recuperados:
            return RespostaAgente(
                "Não encontrei informações suficientes na base de conhecimento do WordGen para "
                "responder a essa pergunta.",
                "sem_evidencia",
                (),
            )
        fontes = tuple(FonteResposta(documento.fonte, score) for documento, score in recuperados)
        contexto = "\n\n".join(
            f"[Fonte: {documento.fonte}]\n{documento.conteudo}" for documento, _ in recuperados
        )
        if usar_llm and self.llm.disponivel():
            try:
                return RespostaAgente(self.llm.gerar(pergunta, contexto), "rag_llm", fontes)
            except LLMIndisponivel as erro:
                LOGGER.warning(
                    "Síntese Groq indisponível: %s", erro.detalhe_tecnico or type(erro).__name__
                )
                aviso = str(erro)
        else:
            aviso = (
                "Síntese por LLM indisponível: configure GROQ_API_KEY para respostas geradas. "
                "A resposta abaixo é um excerto recuperado da base publicada."
            )
        documento, _ = recuperados[0]
        return RespostaAgente(documento.conteudo, "rag_extrativo", fontes, aviso)

    def _responder_metodologia(self, pergunta: str) -> RespostaAgente | None:
        """Encaminha perguntas metodológicas recorrentes a evidências específicas."""
        if "paread" in pergunta and any(termo in pergunta for termo in ("criterio", "completa")):
            documentos = self._documentos_por_id("wordgen_vocab_fase_2")
            if documentos:
                return RespostaAgente(
                    "Uma avaliação pareada é completa quando há pelo menos 80% dos itens "
                    "válidos no pré-teste e no pós-teste. Somente esses pares completos entram nas "
                    "análises que usam o critério de completude estrita.",
                    "rag_metodologia",
                    documentos,
                )
        if "contagem" in pergunta and "paread" in pergunta and "estudantes unicos" in pergunta:
            documentos = self._documentos_por_id("wordgen_visao_geral")
            if documentos:
                return RespostaAgente(
                    "Linhas de avaliações pareadas contam ocorrências de pares pré/pós e um mesmo "
                    "estudante pode contribuir com mais de uma linha em fases diferentes. "
                    "Estudantes únicos eliminam essa repetição por aluno_id. "
                    "No critério de pares completos, há "
                    "7.116 linhas para 2.327 estudantes únicos com pelo menos um par completo.",
                    "rag_metodologia",
                    documentos,
                )
        if "media global" in pergunta and "vocab" in pergunta and "fase 3" in pergunta:
            documentos = self._documentos_por_id("wordgen_longitudinal_transicao")
            if documentos:
                return RespostaAgente(
                    "A média global de Vocabulário na Fase 3 é influenciada pela composição da "
                    "amostra. Na transição, 531 estudantes acompanhados tiveram variação média de "
                    "-3,91 pontos; com completude estrita, foram 520 estudantes e -4,06 pontos. "
                    "Além disso, 370 novos estudantes entraram com média pré de 19,03, enquanto os "
                    "veteranos tinham média pré de 22,69. Esses resultados são descritivos "
                    "e não demonstram causalidade.",
                    "rag_metodologia",
                    documentos,
                )
        if "validacao" in pergunta and "modelo" in pergunta:
            documentos = self._documentos_por_id(
                "modelagem_estrategia_validacao_3", "modelagem_estrategia_validacao_5"
            )
            if documentos:
                return RespostaAgente(
                    "A separação final usa GroupShuffleSplit por aluno_id, com aproximadamente 80% "
                    "para treino e 20% para teste, evitando que o mesmo estudante apareça nos dois "
                    "conjuntos. A busca de hiperparâmetros usa GroupKFold com 5 folds; o "
                    "pré-processamento é ajustado dentro de cada fold. O early_stopping do MLP foi "
                    "desabilitado para evitar divisão interna incompatível com os grupos.",
                    "rag_metodologia",
                    documentos,
                )
        if "experimento b" in pergunta and "explorator" in pergunta:
            documentos = self._documentos_por_id("modelagem_definicao_problema_7")
            if documentos:
                return RespostaAgente(
                    "O Experimento B acrescenta atributos demográficos e contextuais ao cenário "
                    "acadêmico. Ele é exploratório porque dim_aluno consolida atributos de "
                    "múltiplas fases e não há comprovação de disponibilidade individual no "
                    "momento da previsão.",
                    "rag_metodologia",
                    documentos,
                )
        if "limitac" in pergunta and "modelo" in pergunta:
            documentos = self._documentos_por_id("modelagem_avaliacao_modelos_10")
            if documentos:
                return RespostaAgente(
                    "Os modelos foram treinados apenas com pares completos, têm variância residual "
                    "não explicada e expressam associações preditivas, não causalidade. Eles são "
                    "experimentais: antes de qualquer triagem pedagógica, exigem validação "
                    "em novos "
                    "dados, avaliação por grupos, auditoria de vieses, confirmação temporal das "
                    "variáveis e avaliação da utilidade dos alertas. Não substituem profissionais "
                    "da educação.",
                    "rag_metodologia",
                    documentos,
                )
        if "cohen" in pergunta and "positivo" in pergunta:
            documentos = self._documentos_por_id("wordgen_vocab_fase_2")
            if documentos:
                return RespostaAgente(
                    "Um d_z de Cohen positivo indica que, na amostra pareada analisada, a média do "
                    "pós-teste foi maior que a do pré-teste. A magnitude deve ser interpretada "
                    "junto com o intervalo de confiança, o tamanho da amostra e o contexto do "
                    "instrumento; "
                    "não demonstra causalidade por si só.",
                    "rag_metodologia",
                    documentos,
                )
        return None

    def _documentos_por_id(self, *identificadores: str) -> tuple[FonteResposta, ...]:
        documentos = [self.indice.obter(identificador) for identificador in identificadores]
        return tuple(
            FonteResposta(documento.fonte, 1.0) for documento in documentos if documento is not None
        )


def construir_agente(configuracao: ConfiguracaoAgente | None = None) -> AgenteWordGen:
    """Monta o agente sem alterar os dados nem reexecutar os notebooks analíticos."""
    configuracao = configuracao or ConfiguracaoAgente()
    documentos = carregar_base_conhecimento(configuracao.base_conhecimento)
    documentos.extend(carregar_documentacao_modelagem(configuracao.raiz / "docs" / "modelagem"))
    indice = IndiceVetorial.carregar_ou_construir(documentos, configuracao.indice_vetorial)
    return AgenteWordGen(indice, ConsultasAnaliticas(configuracao))
