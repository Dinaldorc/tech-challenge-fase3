"""
Dataset e pipeline pra responder à pergunta de negócio "como prever
municípios que podem não atingir metas futuras?" (estratégia do Tech
Challenge, ver README). Modelagem a nível de **município**, não de aluno.

Alvo: `IN_META_ATINGIDA` -- o município bateu a meta de alfabetização
daquele ano (rede Municipal, `FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO`)?

Features: desempenho do **ano anterior** (não vaza o resultado do ano que
está sendo previsto) + o mesmo enriquecimento socioeconômico/infraestrutura
usado no modelo de aluno (`_build_dim_municipio_socioeconomico`, estático
por município) + território (`REGIAO`/`SG_UF`).

Diferente de `ID_ALUNO`/`ID_ESCOLA` (máscaras re-sorteadas a cada ano pelo
INEP -- ver `features.py`), `CO_MUNICIPIO` é o código IBGE oficial e é
estável entre anos. Por isso aqui um split **temporal de verdade** (treinar
no passado, testar no futuro) é válido e não tem o problema de vazamento
por reciclagem de ID que inviabilizou o split temporal no modelo de aluno.
"""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..preprocessing import commons as c
from ..preprocessing.gold import _build_dim_municipio_socioeconomico

TARGET_COL = "IN_META_ATINGIDA"

NUMERIC_COLS = [
    "PC_ALUNO_ALFABETIZADO_ANTERIOR",
    "DIF_META_ALFABETIZACAO_ANTERIOR",
    "PC_FAMILIAS_POBREZA",
    "RENDA_PER_CAPITA_MEDIA",
    "MEDIA_INSE",
    "PC_ESCOLAS_BIBLIOTECA",
    "PC_ESCOLAS_LAB_INFORMATICA",
    "PC_ESCOLAS_INTERNET_ALUNOS",
]
CATEGORICAL_COLS = ["REGIAO", "SG_UF", "IN_META_ATINGIDA_ANTERIOR"]


def build_dataset() -> pd.DataFrame:
    """Monta o dataset com lag de 1 ano: para cada (município, ano N),
    junta o resultado do ano N (target) com os indicadores do ano N-1
    (features) e o enriquecimento estático por município."""
    meta = c.read_table(c.GOLD_PATH, c.FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO)
    cols = ["ANO_REFERENCIA", "CO_MUNICIPIO", "SG_UF", "REGIAO",
            "PC_ALUNO_ALFABETIZADO", "DIF_META_ALFABETIZACAO", "IN_META_ATINGIDA"]
    atual = meta[cols].copy()

    anterior = atual.copy()
    anterior["ANO_REFERENCIA"] = anterior["ANO_REFERENCIA"] + 1
    anterior = anterior.rename(columns={
        "PC_ALUNO_ALFABETIZADO": "PC_ALUNO_ALFABETIZADO_ANTERIOR",
        "DIF_META_ALFABETIZACAO": "DIF_META_ALFABETIZACAO_ANTERIOR",
        "IN_META_ATINGIDA": "IN_META_ATINGIDA_ANTERIOR",
    })[["ANO_REFERENCIA", "CO_MUNICIPIO", "PC_ALUNO_ALFABETIZADO_ANTERIOR",
        "DIF_META_ALFABETIZACAO_ANTERIOR", "IN_META_ATINGIDA_ANTERIOR"]]

    df = atual.merge(anterior, on=["ANO_REFERENCIA", "CO_MUNICIPIO"], how="inner")

    dim_socioeconomico = _build_dim_municipio_socioeconomico()
    df = df.merge(dim_socioeconomico, on="CO_MUNICIPIO", how="left")

    return df


def split_by_year(
    df: pd.DataFrame, train_year: int = 2024, test_year: int = 2025
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal: treina prevendo `train_year` (com indicadores de
    `train_year - 1`), testa prevendo `test_year` (com indicadores de
    `test_year - 1`) -- genuinamente "passado prevê futuro", município a
    município."""
    train_df = df[df["ANO_REFERENCIA"] == train_year].copy()
    test_df = df[df["ANO_REFERENCIA"] == test_year].copy()
    return train_df, test_df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    for col in X.columns:
        dtype = X[col].dtype
        if pd.api.types.is_extension_array_dtype(dtype) and pd.api.types.is_numeric_dtype(dtype):
            X[col] = X[col].astype("float64")
    y = df[TARGET_COL].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])
