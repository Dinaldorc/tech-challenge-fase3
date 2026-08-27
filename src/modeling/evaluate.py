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
