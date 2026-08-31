"""
Split treino/teste para a FT_MACHINE_LEARNING.

Estratégia: restringe a modelagem ao ano de 2025 e faz um split aleatório
estratificado por TARGET (70% treino / 30% teste).

Por que só 2025 (decisão do time, substitui o split temporal usado
antes): evita o *dataset shift* real que o split temporal (2023-2024 →
2025) introduzia -- a taxa de alfabetização mudou entre os recortes
(51,3% vs. 58,6%), o que dificultava separar "o modelo generaliza mal"
de "o mundo mudou entre os anos". Com um único ano e split aleatório
estratificado, treino e teste têm a mesma distribuição por construção.

Por que estratificado por TARGET: garante a mesma proporção de
alfabetizados/não-alfabetizados nos dois lados, eliminando de vez esse
tipo de diferença como fator de confusão nas métricas.
"""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

ANO_MODELAGEM = 2025
TEST_SIZE = 0.30
RANDOM_STATE = 42


def split_2025(
    df: pd.DataFrame,
    ano: int = ANO_MODELAGEM,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_ano = df[df["ANO_REFERENCIA"] == ano]
    train_df, test_df = train_test_split(
        df_ano, test_size=test_size, random_state=random_state, stratify=df_ano["TARGET"],
    )
    return train_df, test_df
