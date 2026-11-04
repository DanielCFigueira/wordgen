"""Consultas agregadas e determinísticas aos artefatos publicados do WordGen."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wordgen.agente.config import ConfiguracaoAgente


@dataclass(frozen=True)
class ResultadoAnalitico:
    resposta: str
    fontes: tuple[str, ...]
    tipo: str


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(caractere for caractere in texto if not unicodedata.combining(caractere))


def _ler_csv(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        return pd.DataFrame()
    return pd.read_csv(caminho, encoding="utf-8-sig", low_memory=False)


def _instrumento(texto: str) -> str | None:
    if "vocab" in texto:
        return "Vocabulário"
    if "tde" in texto or "escrita" in texto:
        return "TDE II"
    return None


def _fase(texto: str) -> str | None:
    busca = re.search(r"fase\s*([234])", texto)
    return f"Fase {busca.group(1)}" if busca else None


class ConsultasAnaliticas:
    """Roteador fechado: não executa SQL ou código fornecido pela pergunta do usuário."""

    def __init__(self, configuracao: ConfiguracaoAgente | None = None) -> None:
        self.configuracao = configuracao or ConfiguracaoAgente()

    def consultar(self, pergunta: str) -> ResultadoAnalitico | None:
        texto = _normalizar(pergunta)
        if any(termo in texto for termo in ("mae", "rmse", "r2", "r²", "random forest", "mlp")):
            return self._modelagem(texto)
        if "sexo" in texto and _instrumento(texto) is not None and _fase(texto) is not None:
            return self._equidade(texto)
        if "necessidade" in texto or "socioeconom" in texto or "zona" in texto:
            return None
        if any(
            termo in texto for termo in ("longitud", "transicao", "fase 2 pos", "fase 3 pre")
        ) or ("entre" in texto and "fase" in texto and "pre" in texto and "pos" in texto):
            return self._longitudinal(texto)
        tem_medidas_pre_pos = "pre" in texto and "pos" in texto
        if (
            any(
                termo in texto
                for termo in ("effect size", "cohen", "d_z", "dz", "media pre", "media pos")
            )
            or tem_medidas_pre_pos
            or (
                "resultado" in texto
                and _instrumento(texto) is not None
                and _fase(texto) is not None
            )
        ) and (_instrumento(texto) is not None or _fase(texto) is not None):
            return self._effect_size(texto)
        if any(
            termo in texto
            for termo in (
                "quantos estudantes",
                "quantos alunos",
                "quantas escolas",
                "quantas turmas",
                "total de testes",
                "cobertura",
            )
        ):
            return self._kpis(texto)
        return None

    def _kpis(self, texto: str) -> ResultadoAnalitico | None:
        dados = _ler_csv(self.configuracao.diretorio_analytics / "dashboard_kpis.csv")
        if dados.empty:
            return None
        alvo = None
        if "escola" in texto and "turma" in texto:
            itens = dados.loc[
                dados["KPI"].isin(
                    [
                        "Escolas Públicas Atendidas",
                        "Turmas em Escolas Identificadas",
                        "Turmas Totais Registradas (incl. Não Informada)",
                    ]
                )
            ]
            if len(itens) != 3:
                return None
            resposta = " ".join(
                f"{item['KPI']}: {item['Valor_Formatado']} {item['Unidade'].lower()}."
                for _, item in itens.iterrows()
            )
            return ResultadoAnalitico(resposta, ("data/analytics/dashboard_kpis.csv",), "kpi")
        if "escola" in texto:
            alvo = "Escolas Públicas Atendidas"
        elif "turma" in texto:
            alvo = "Turmas em Escolas Identificadas"
        elif "teste" in texto:
            alvo = "Total de Registros de Testes Aplicados"
        elif "cobertura" in texto:
            alvo = "Cobertura Socioeconômica"
        elif "vocab" in texto:
            alvo = "Estudantes Únicos em Vocabulário"
        elif "tde" in texto:
            alvo = "Estudantes Únicos em TDE II"
        elif "estudante" in texto or "aluno" in texto:
            alvo = "Estudantes Únicos (União Vocabulário e TDE)"
        linha = dados.loc[dados["KPI"].eq(alvo)] if alvo else pd.DataFrame()
        if linha.empty:
            return None
        item = linha.iloc[0]
        resposta = (
            f"{item['KPI']}: {item['Valor_Formatado']} {item['Unidade'].lower()}. "
            f"{item['Descricao']}"
        )
        return ResultadoAnalitico(
            resposta,
            ("data/analytics/dashboard_kpis.csv",),
            "kpi",
        )

    def _modelagem(self, texto: str) -> ResultadoAnalitico | None:
        dados = _ler_csv(self.configuracao.metricas_modelagem)
        if dados.empty:
            return None
        cenario = dados["Cenário"].astype(str).map(_normalizar)
        instrumento = _instrumento(texto)
        if instrumento:
            dados = dados.loc[cenario.str.contains(_normalizar(instrumento), regex=False)].copy()
        experimento = re.search(r"experimento\s*([ab])", texto)
        if experimento:
            dados = dados.loc[
                dados["Cenário"]
                .astype(str)
                .str.contains(
                    f"Experimento {experimento.group(1).upper()}", case=False, regex=False
                )
            ].copy()
        if "random forest" in texto:
            dados = dados.loc[dados["Modelo"].astype(str).str.contains("Random Forest", case=False)]
        elif "mlp" in texto or "rede neural" in texto:
            dados = dados.loc[dados["Modelo"].astype(str).str.contains("MLP", case=False)]
        if dados.empty:
            return None
        linhas = []
        for _, linha in dados.iterrows():
            linhas.append(
                f"{linha['Cenário']} — {linha['Modelo']}: MAE {linha['MAE']:.4f}, "
                f"RMSE {linha['RMSE']:.4f}, R² {linha['R²']:.4f}."
            )
        return ResultadoAnalitico(
            "\n".join(linhas),
            ("reports/modelagem/metricas/metricas_comparativas_gerais.csv",),
            "modelagem",
        )

    def _effect_size(self, texto: str) -> ResultadoAnalitico | None:
        dados = _ler_csv(self.configuracao.diretorio_analytics / "metricas_effect_size.csv")
        if dados.empty:
            return None
        instrumento, fase = _instrumento(texto), _fase(texto)
        if instrumento:
            dados = dados.loc[dados["Instrumento"].eq(instrumento)]
        if fase:
            dados = dados.loc[dados["Fase"].eq(fase)]
        dados = dados.loc[dados["Estratificacao"].eq("Geral_Completo")]
        if dados.empty:
            return None
        linhas = [
            (
                f"{linha['Instrumento']} — {linha['Fase']}: "
                f"{int(linha['N Pares'])} pares completos; "
                f"média pré {linha['Média Pré']:.2f}, pós {linha['Média Pós']:.2f}, "
                f"ganho médio {linha['Ganho Médio']:+.2f}; "
                f"d_z de Cohen {linha['d_z (Cohen)']:+.3f} "
                f"(IC 95% [{linha['IC 95% Inf']:.3f}, {linha['IC 95% Sup']:.3f}])."
            )
            for _, linha in dados.iterrows()
        ]
        return ResultadoAnalitico(
            "\n".join(linhas), ("data/analytics/metricas_effect_size.csv",), "effect_size"
        )

    def _longitudinal(self, texto: str) -> ResultadoAnalitico | None:
        dados = _ler_csv(self.configuracao.diretorio_analytics / "metricas_longitudinais.csv")
        if dados.empty:
            return None
        dados = dados.loc[dados["Tipo_Metrica"].eq("Transicao_Interfases")]
        instrumento = _instrumento(texto)
        if instrumento:
            dados = dados.loc[dados["Instrumento"].eq(instrumento)]
        if "complet" in texto:
            dados = dados.loc[
                dados["Critério_Amostra"].astype(str).str.contains("Completude", case=False)
            ]
        if dados.empty:
            return None
        linhas = [
            (
                f"{linha['Instrumento']} — {linha['Transição']} ({linha['Critério_Amostra']}): "
                f"{int(linha['N Pares'])} estudantes acompanhados; média pós anterior "
                f"{linha['Média Pós Anterior']:.2f}, média pré seguinte "
                f"{linha['Média Pré Seguinte']:.2f}, "
                f"variação média {linha['Variação Média (Delta)']:+.2f}."
            )
            for _, linha in dados.iterrows()
        ]
        return ResultadoAnalitico(
            "\n".join(linhas), ("data/analytics/metricas_longitudinais.csv",), "longitudinal"
        )

    def _equidade(self, texto: str) -> ResultadoAnalitico | None:
        dados = _ler_csv(self.configuracao.diretorio_analytics / "metricas_equidade.csv")
        if dados.empty:
            return None
        instrumento, fase = _instrumento(texto), _fase(texto)
        dados = dados.loc[
            dados["Instrumento"].eq(instrumento)
            & dados["Fase"].eq(fase)
            & dados["Dimensao"].eq("sexo")
        ]
        if dados.empty:
            return None
        linhas = [
            (
                f"{linha['Instrumento']} — {linha['Fase']}, {linha['Criterio_Amostra']}, "
                f"sexo {linha['Subgrupo']}: "
                f"{int(linha['N Pares'])} pares; média pré {linha['Média Pré']:.2f}, "
                f"pós {linha['Média Pós']:.2f}, ganho {linha['Ganho Médio']:+.2f}, "
                f"d_z {linha['d_z']:+.3f}."
            )
            for _, linha in dados.iterrows()
        ]
        return ResultadoAnalitico(
            "\n".join(linhas), ("data/analytics/metricas_equidade.csv",), "equidade"
        )
