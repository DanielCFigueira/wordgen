"""Módulo de Modelagem Preditiva de Machine Learning do WordGen."""

from wordgen.modelagem.avaliacao import (
    avaliar_modelo,
    calcular_metricas,
    consolidar_tabela_metricas,
)
from wordgen.modelagem.baselines import BaselineGanhoZero, BaselineMediaTreino
from wordgen.modelagem.interpretacao import (
    calcular_importancia_rf,
    calcular_permutation_importance,
    extrair_nomes_features,
)
from wordgen.modelagem.persistencia import carregar_artefato_modelo, salvar_artefato_modelo
from wordgen.modelagem.pipelines import criar_pipeline_mlp, criar_pipeline_rf, obter_pre_processador
from wordgen.modelagem.preparacao import (
    carregar_bases_brutas,
    carregar_e_preparar_dados,
    preparar_base_instrumento,
)
from wordgen.modelagem.treinamento import treinar_e_otimizar_mlp, treinar_e_otimizar_rf
from wordgen.modelagem.validacao import dividir_treino_teste_agrupado, obter_cv_agrupado

__all__ = [
    "carregar_bases_brutas",
    "preparar_base_instrumento",
    "carregar_e_preparar_dados",
    "dividir_treino_teste_agrupado",
    "obter_cv_agrupado",
    "BaselineGanhoZero",
    "BaselineMediaTreino",
    "obter_pre_processador",
    "criar_pipeline_rf",
    "criar_pipeline_mlp",
    "treinar_e_otimizar_rf",
    "treinar_e_otimizar_mlp",
    "calcular_metricas",
    "avaliar_modelo",
    "consolidar_tabela_metricas",
    "extrair_nomes_features",
    "calcular_importancia_rf",
    "calcular_permutation_importance",
    "salvar_artefato_modelo",
    "carregar_artefato_modelo",
]
