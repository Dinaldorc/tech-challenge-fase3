"""
Interpretabilidade via SHAP (SHapley Additive exPlanations) do melhor
cenário do passo 4 (RandomForest + enriquecimento socioeconômico).

Usa `TreeExplainer` (otimizado pra modelos em árvore) numa amostra do
teste -- calcular SHAP nas ~2,2 milhões de linhas inteiras seria caro
demais sem ganho real de precisão na análise agregada.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse
import shap
from sklearn.pipeline import Pipeline


def get_expanded_feature_names(preprocessor, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    """Nomes das colunas após o ColumnTransformer, na mesma ordem em que
    saem da transformação (numéricas primeiro, depois cada categórica
    expandida em uma coluna por categoria, ex.: "REGIAO=Norte")."""
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    names = list(numeric_cols)
    for col, categorias in zip(categorical_cols, onehot.categories_):
        names += [f"{col}={cat}" for cat in categorias]
    return names


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    sample_size: int = 20_000,
    random_state: int = 42,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Retorna (shap_values, feature_names, X_amostra) para a classe
    positiva (TARGET=1, alfabetizado)."""
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    X_transformed = preprocessor.transform(X_sample)
    if scipy.sparse.issparse(X_transformed):
        # TreeExplainer (shap 0.52) nao trata matriz esparsa da OneHotEncoder
        # corretamente -- vira array numpy de dtype object e quebra no
        # np.isnan interno. Amostra e' pequena o suficiente pra densificar.
        X_transformed = X_transformed.toarray()

    feature_names = get_expanded_feature_names(preprocessor, numeric_cols, categorical_cols)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):  # versões antigas do shap: uma matriz por classe
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:  # versões novas: (n, features, classes)
        shap_values = shap_values[:, :, 1]

    return shap_values, feature_names, X_sample


def importancia_media_por_feature_expandida(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    """|SHAP| médio por coluna expandida (ex.: "REGIAO=Norte" isolado)."""
    return pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)


def importancia_por_variavel_original(
    shap_values: np.ndarray, feature_names: list[str], variaveis: list[str]
) -> pd.Series:
    """Soma o |SHAP| médio das colunas expandidas de volta pra variável
    original (ex.: soma REGIAO=Norte + REGIAO=Sul + ... em "REGIAO")."""
    medio_expandido = importancia_media_por_feature_expandida(shap_values, feature_names)
    agregada = {}
    for var in variaveis:
        cols = [f for f in feature_names if f == var or f.startswith(f"{var}=")]
        agregada[var] = medio_expandido[cols].sum()
    return pd.Series(agregada).sort_values(ascending=False)


def efeito_medio_por_categoria(
    shap_values: np.ndarray, feature_names: list[str], X_sample: pd.DataFrame, coluna_original: str
) -> pd.DataFrame:
    """SHAP médio (com sinal) de cada categoria de `coluna_original`,
    calculado apenas nas linhas que de fato pertencem a cada categoria --
    responde "pra alunos do Norte, o quanto REGIAO=Norte empurra a
    previsão pra baixo?", não só a força média entre todas as linhas."""
    linhas = []
    for categoria in X_sample[coluna_original].unique():
        col_name = f"{coluna_original}={categoria}"
        if col_name not in feature_names:
            continue
        idx = feature_names.index(col_name)
        mask = (X_sample[coluna_original] == categoria).to_numpy()
        linhas.append({
            "categoria": categoria,
            "n": int(mask.sum()),
            "shap_medio": shap_values[mask, idx].mean(),
        })
    return pd.DataFrame(linhas).sort_values("shap_medio")
