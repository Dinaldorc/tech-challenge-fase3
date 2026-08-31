"""
Métricas de avaliação para os baselines de alfabetização.

Inclui a acurácia da classe majoritária como referência -- "acurácia"
sozinha pode parecer melhor do que realmente é quando o target não é
50/50, já que um modelo que sempre prevê a classe majoritária acerta essa
fração sem aprender nada. Suporta `sample_weight` (peso amostral oficial
do INEP, `VL_PESO_ALUNO_LP` -- ver `features.select_weights`) em todas as
métricas, pra refletir a representatividade populacional de cada aluno.
"""
from __future__ import annotations
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def majority_class_baseline(y_test: pd.Series, sample_weight: pd.Series | None = None) -> float:
    if sample_weight is None:
        return y_test.value_counts(normalize=True).max()
    pesos_por_classe = pd.Series(sample_weight.to_numpy(), index=y_test.index).groupby(y_test).sum()
    return (pesos_por_classe / pesos_por_classe.sum()).max()


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, sample_weight: pd.Series | None = None
) -> dict:
    """`sample_weight`, quando informado (ex.: `VL_PESO_ALUNO_LP` -- ver
    `features.select_weights`), pondera todas as métricas pra refletir a
    representatividade populacional de cada aluno, não contagem simples de
    linhas -- ver branch feature/correcao-peso-amostral."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    sw = sample_weight.to_numpy() if sample_weight is not None else None
    return {
        "accuracy": accuracy_score(y_test, pred, sample_weight=sw),
        "baseline_classe_majoritaria": majority_class_baseline(y_test, sample_weight),
        "precision": precision_score(y_test, pred, sample_weight=sw),
        "recall": recall_score(y_test, pred, sample_weight=sw),
        "f1": f1_score(y_test, pred, sample_weight=sw),
        "roc_auc": roc_auc_score(y_test, proba, sample_weight=sw),
        "confusion_matrix": confusion_matrix(y_test, pred, sample_weight=sw).tolist(),
    }


def evaluate_by_group(
    model, X_test: pd.DataFrame, y_test: pd.Series, group_col: str, sample_weight: pd.Series | None = None
) -> pd.DataFrame:
    """Quebra accuracy/precision/recall por `group_col` (ex.: "REGIAO" ou
    "SG_UF"). A EDA (Seção 5, revisada) mostrou que a desigualdade real é
    bem maior por UF do que por região -- aqui verificamos se o modelo
    reproduz, atenua ou agrava essa disparidade em cada granularidade.
    `sample_weight` pondera a `taxa_real_alfabetizacao` e as métricas por
    grupo -- ver `evaluate_model`."""
    pred = model.predict(X_test)
    df = pd.DataFrame({
        "grupo": X_test[group_col].to_numpy(),
        "y_true": y_test.to_numpy(),
        "y_pred": pred,
        "peso": sample_weight.to_numpy() if sample_weight is not None else 1.0,
    })

    linhas = []
    for grupo, g in df.groupby("grupo"):
        sw = g["peso"].to_numpy() if sample_weight is not None else None
        taxa = (g["y_true"] * g["peso"]).sum() / g["peso"].sum() if sample_weight is not None else g["y_true"].mean()
        linhas.append({
            group_col: grupo,
            "n": len(g),
            "taxa_real_alfabetizacao": taxa,
            "accuracy": accuracy_score(g["y_true"], g["y_pred"], sample_weight=sw),
            "precision": precision_score(g["y_true"], g["y_pred"], sample_weight=sw, zero_division=0),
            "recall": recall_score(g["y_true"], g["y_pred"], sample_weight=sw, zero_division=0),
        })
    return pd.DataFrame(linhas).sort_values("taxa_real_alfabetizacao").reset_index(drop=True)
