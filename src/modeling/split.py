"""
Split treino/teste para a FT_MACHINE_LEARNING.

Estratégia: split temporal, não aleatório -- treina nos anos anteriores,
testa no ano mais recente. Simula o uso real do modelo (prever o ano
seguinte com base em anos anteriores).

Nota: cogitamos inicialmente agrupar o split por ID_ALUNO para evitar o
mesmo aluno em treino e teste, mas o campo não é um identificador
persistente entre anos (ver `features.IDENTIFICADORES_METADADOS`) -- a
faixa numérica se repete a cada ano e, dos IDs coincidentes entre anos, 0%
correspondem à mesma escola. Não há vazamento de aluno para evitar; o
split temporal simples já é suficiente.
"""
from __future__ import annotations
import pandas as pd

TRAIN_YEARS = (2023, 2024)
TEST_YEAR = 2025


def split_temporal(
    df: pd.DataFrame,
    train_years: tuple[int, ...] = TRAIN_YEARS,
    test_year: int = TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df["ANO_REFERENCIA"].isin(train_years)].copy()
    test_df = df[df["ANO_REFERENCIA"] == test_year].copy()
    return train_df, test_df
