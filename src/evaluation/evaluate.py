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
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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


def calibrar_limiar_decisao(
    model, X_test: pd.DataFrame, y_test: pd.Series, classe_alvo=0, recall_minimo: float = 0.80,
) -> pd.DataFrame:
    """Compara o limiar de decisão padrão (0,5, embutido em `.predict()`)
    com dois limiares alternativos sobre `predict_proba`, pra classe
    `classe_alvo` (default 0 -- ex.: "não atingiu meta", a classe que
    importa pra priorização de política pública, ver README):

    - **F1 ótimo**: o limiar que maximiza a média harmônica entre precisão
      e recall -- um critério estatístico neutro, sem juízo sobre qual erro
      (falso positivo ou falso negativo) é mais caro.
    - **Recall mínimo** (`recall_minimo`, default 0,80): o limiar mais alto
      (logo, de maior precisão) que ainda garante pelo menos esse recall --
      uma escolha de política, não estatística, pra quando deixar de fora
      um caso positivo é considerado mais caro que investigar um falso
      alarme.

    Retorna um DataFrame com uma linha por estratégia de limiar, pronta pra
    salvar em CSV e apresentar ao gestor escolher o trade-off."""
    proba_alvo = model.predict_proba(X_test)[:, classe_alvo]
    y_bin = (y_test == classe_alvo).astype(int).to_numpy()

    precision, recall, thresholds = precision_recall_curve(y_bin, proba_alvo)
    f1_scores = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0,
    )
    idx_f1 = int(np.argmax(f1_scores[:-1]))  # ultimo ponto da curva nao tem limiar

    candidatos_recall = np.where(recall[:-1] >= recall_minimo)[0]
    idx_recall = int(candidatos_recall[np.argmax(precision[candidatos_recall])]) if len(candidatos_recall) else None

    def _linha(nome: str, limiar: float) -> dict:
        pred = (proba_alvo >= limiar).astype(int)
        return {
            "estrategia": nome,
            "limiar": limiar,
            "precision": precision_score(y_bin, pred, zero_division=0),
            "recall": recall_score(y_bin, pred, zero_division=0),
            "f1": f1_score(y_bin, pred, zero_division=0),
        }

    linhas = [
        _linha("padrao_0.5", 0.5),
        _linha("f1_otimo", float(thresholds[idx_f1])),
    ]
    if idx_recall is not None:
        linhas.append(_linha(f"recall_minimo_{recall_minimo:.0%}", float(thresholds[idx_recall])))
    return pd.DataFrame(linhas)


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
