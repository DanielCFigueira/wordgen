"""Executa a avaliação local de cobertura do benchmark do agente WordGen."""

from wordgen.agente.avaliacao import avaliar_benchmark, resumo_benchmark
from wordgen.agente.config import ConfiguracaoAgente
from wordgen.agente.orquestrador import construir_agente


def main() -> None:
    configuracao = ConfiguracaoAgente()
    resultados = avaliar_benchmark(construir_agente(configuracao), configuracao.benchmark)
    resumo = resumo_benchmark(resultados)
    print(
        "Benchmark local: "
        f"{resumo['aprovadas']}/{resumo['total']} aprovadas; "
        f"{resumo['respondidas']}/{resumo['total']} respondidas; "
        f"{resumo['com_fontes']}/{resumo['total']} com fonte identificada."
    )
    for resultado in resultados:
        estado = "APROVADA" if resultado.aprovado else "REPROVADA"
        print(f"{estado} {resultado.identificador}: {resultado.rota} — {resultado.justificativa}")


if __name__ == "__main__":
    main()
