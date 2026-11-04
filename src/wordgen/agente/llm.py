"""Adaptador opcional do Groq, isolado da recuperação e das métricas exatas."""

from __future__ import annotations

from dataclasses import dataclass

from wordgen.agente.config import ConfiguracaoAgente, obter_chave_groq, obter_modelo_groq


class LLMIndisponivel(RuntimeError):
    """Erro previsível quando a configuração ou serviço de geração não está disponível."""

    def __init__(self, mensagem: str, detalhe_tecnico: str | None = None) -> None:
        super().__init__(mensagem)
        self.detalhe_tecnico = detalhe_tecnico


@dataclass
class ClienteGroq:
    """Cliente mínimo e testável; a chave não é armazenada em arquivos do projeto."""

    modelo: str | None = None

    def disponivel(self) -> bool:
        return obter_chave_groq() is not None

    def gerar(self, pergunta: str, contexto: str) -> str:
        chave = obter_chave_groq()
        if not chave:
            raise LLMIndisponivel(
                "GROQ_API_KEY não está configurada. "
                "Configure-a no ambiente para habilitar sínteses geradas."
            )
        try:
            from groq import Groq

            cliente = Groq(api_key=chave, timeout=20.0, max_retries=1)
            modelo = self._resolver_modelo(cliente)
            resposta = cliente.chat.completions.create(
                model=modelo,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é o assistente do projeto acadêmico WordGen. Responda apenas com "
                            "base no contexto fornecido. Não invente números, fontes ou "
                            "causalidade; admita limites de evidência e não revele dados "
                            "individuais. "
                            "Cite as fontes "
                            "no formato [Fonte: ...]."
                        ),
                    },
                    {"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}"},
                ],
            )
        except Exception as erro:  # A SDK expõe classes específicas que variam com a versão.
            codigo = getattr(erro, "status_code", None)
            detalhe = f"{type(erro).__name__}" + (f" (HTTP {codigo})" if codigo else "")
            if codigo in {401, 403}:
                mensagem = "A configuração da Groq foi recusada. Verifique a chave no ambiente."
            elif codigo == 404:
                mensagem = "O modelo Groq configurado não está disponível nesta conta."
            elif codigo == 429:
                mensagem = "O limite temporário da Groq foi atingido. Tente novamente mais tarde."
            elif codigo and codigo >= 500:
                mensagem = (
                    "O serviço Groq está temporariamente indisponível. Tente novamente mais tarde."
                )
            else:
                mensagem = "Não foi possível consultar o serviço Groq no momento."
            raise LLMIndisponivel(mensagem, detalhe) from erro

        texto = resposta.choices[0].message.content
        if not texto or not texto.strip():
            raise LLMIndisponivel("O serviço Groq retornou uma resposta vazia.")
        return texto.strip()

    def _resolver_modelo(self, cliente: object) -> str:
        """Usa o modelo configurado quando disponível e um padrão compatível como contingência."""
        configurado = self.modelo or obter_modelo_groq()
        if self.modelo:
            return configurado
        try:
            disponiveis = {modelo.id for modelo in cliente.models.list().data}
        except Exception:
            return configurado
        if configurado in disponiveis:
            return configurado
        padrao = ConfiguracaoAgente().modelo_groq
        if padrao in disponiveis:
            return padrao
        raise LLMIndisponivel(
            "Nenhum modelo de chat configurado está disponível nesta conta Groq.",
            "modelo configurado ausente da lista de modelos",
        )
