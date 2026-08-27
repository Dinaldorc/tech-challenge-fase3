"""
Esqueleto do pipeline sklearn (pré-processamento + modelo) para a
FT_MACHINE_LEARNING, usando a seleção de features de `features.py`.

Numéricas: imputação por mediana + padronização.
Categóricas: imputação pelo valor mais frequente + one-hot
(`handle_unknown="ignore"` porque SG_UF/REGIAO/TP_DEPENDENCIA do teste
-- ano 2025 -- devem ser as mesmas categorias do treino, mas o ignore
evita que uma categoria rara ausente no treino quebre o predict).
"""
from __future__ import annotations
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FEATURES_SOCIOECONOMICAS

NUMERIC_BASE = ["ANO_REFERENCIA"]
CATEGORICAL_BASE = ["REGIAO", "SG_UF", "TP_DEPENDENCIA"]


def get_column_types(include_socioeconomico: bool = True) -> tuple[list[str], list[str]]:
    """Espelha `features.get_feature_columns`, separando em numéricas e
    categóricas para o ColumnTransformer."""
    numeric_cols = list(NUMERIC_BASE)
    if include_socioeconomico:
        numeric_cols += FEATURES_SOCIOECONOMICAS
    return numeric_cols, list(CATEGORICAL_BASE)


def build_preprocessor(include_socioeconomico: bool = True) -> ColumnTransformer:
    numeric_cols, categorical_cols = get_column_types(include_socioeconomico)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def build_pipeline(classifier: ClassifierMixin, include_socioeconomico: bool = True) -> Pipeline:
    """Pipeline completo: pré-processamento + `classifier`.

    `classifier` já vem instanciado (ex.: `LogisticRegression(max_iter=1000)`)
    -- este módulo não decide o algoritmo, só monta o pipeline em torno dele.
    """
    return Pipeline([
        ("preprocessor", build_preprocessor(include_socioeconomico)),
        ("classifier", classifier),
    ])
