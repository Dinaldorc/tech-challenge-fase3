"""
Métricas de avaliação para os baselines de alfabetização.

Inclui a acurácia da classe majoritária como referência: a base de teste
(2025) tem 58,6% de alunos alfabetizados (ver split temporal do passo 2),
então "acurácia" sozinha pode parecer melhor do que realmente é -- um
modelo que sempre prevê "alfabetizado" já acerta 58,6% sem aprender nada.
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


def majority_class_baseline(y_test: pd.Series) -> float:
    return y_test.value_counts(normalize=True).max()


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "baseline_classe_majoritaria": majority_class_baseline(y_test),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def evaluate_by_region(model, X_test: pd.DataFrame, y_test: pd.Series, region_col: str = "REGIAO") -> pd.DataFrame:
    """Quebra accuracy/precision/recall por REGIAO. A EDA (Seção 5) já
    mostrava ~15pp de diferença na taxa de alfabetização entre a melhor
    região (Centro-Oeste) e a pior (Norte) -- aqui verificamos se o
    modelo reproduz, atenua ou agrava essa disparidade."""
    pred = model.predict(X_test)
    df = pd.DataFrame({
        "regiao": X_test[region_col].to_numpy(),
        "y_true": y_test.to_numpy(),
        "y_pred": pred,
    })

    linhas = []
    for regiao, g in df.groupby("regiao"):
        linhas.append({
            "regiao": regiao,
            "n": len(g),
            "taxa_real_alfabetizacao": g["y_true"].mean(),
            "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
            "precision": precision_score(g["y_true"], g["y_pred"], zero_division=0),
            "recall": recall_score(g["y_true"], g["y_pred"], zero_division=0),
        })
    return pd.DataFrame(linhas).sort_values("taxa_real_alfabetizacao").reset_index(drop=True)
