"""
Dataset pra responder à pergunta de negócio "quais regiões possuem
padrões semelhantes?" (estratégia do Tech Challenge) -- perfil
socioeconômico + educacional de cada município em 2025, pronto pra
clustering (`run_municipal_clustering.py`).
"""
from __future__ import annotations
import pandas as pd

from ..preprocessing import commons as c
from ..preprocessing.gold import _build_dim_municipio_socioeconomico

FEATURES_PERFIL = [
    "PC_ALUNO_ALFABETIZADO",
    "DISTANCIA_META_2030",
    "PC_FAMILIAS_POBREZA",
    "RENDA_PER_CAPITA_MEDIA",
    "MEDIA_INSE",
    "PC_ESCOLAS_BIBLIOTECA",
    "PC_ESCOLAS_LAB_INFORMATICA",
    "PC_ESCOLAS_INTERNET_ALUNOS",
]


def build_perfil_municipios(ano: int = 2025) -> pd.DataFrame:
    meta = c.read_table(c.GOLD_PATH, c.FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO)
    perfil = meta[meta["ANO_REFERENCIA"] == ano][
        ["CO_MUNICIPIO", "NO_MUNICIPIO", "SG_UF", "REGIAO", "PC_ALUNO_ALFABETIZADO", "DISTANCIA_META_2030"]
    ].copy()

    dim = _build_dim_municipio_socioeconomico()
    perfil = perfil.merge(dim, on="CO_MUNICIPIO", how="left")

    return perfil.dropna(subset=FEATURES_PERFIL)
