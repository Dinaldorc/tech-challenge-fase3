"""
Gráficos reutilizáveis a partir dos relatórios já gerados em `reports/`
(não retreina nenhum modelo -- só visualiza o que já foi calculado pelos
scripts de `src/modeling`).
"""
from __future__ import annotations
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import pandas as pd

COR_PADRAO = "#2c7fb8"
COR_DEGENERADO = "#d94801"
COR_OK = "#2c7fb8"


def salvar(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bar_importancia_shap(importancia: pd.Series, titulo: str) -> plt.Figure:
    """`importancia`: index = variável, valores = |SHAP| médio, já ordenado."""
    fig, ax = plt.subplots(figsize=(8, 4))
    importancia.sort_values().plot(kind="barh", ax=ax, color=COR_PADRAO)
    ax.set_xlabel("Importância média (|SHAP|)")
    ax.set_title(titulo, fontsize=11)
    return fig


def bar_metricas_por_grupo(
    df: pd.DataFrame, grupo_col: str, metrica_col: str, titulo: str, destacar_degenerado: bool = False,
) -> plt.Figure:
    """`df` como salvo por `evaluate.evaluate_by_group` (colunas: grupo_col,
    metrica_col, entre outras). `destacar_degenerado=True` pinta de outra
    cor as barras com valor 0 ou >=0.99 (achado da modelagem municipal)."""
    dados = df.sort_values(metrica_col)
    cores = [COR_PADRAO] * len(dados)
    if destacar_degenerado:
        cores = [COR_DEGENERADO if (v == 0 or v >= 0.99) else COR_OK for v in dados[metrica_col]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(dados[grupo_col].astype(str), dados[metrica_col], color=cores)
    ax.set_ylabel(metrica_col)
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=90 if len(dados) > 10 else 30)
    return fig


def bar_comparacao_modelos(df: pd.DataFrame, titulo: str) -> plt.Figure:
    """`df` como salvo por `run_baseline.run()` (colunas: modelo,
    enriquecimento_socioeconomico, accuracy, f1, roc_auc, ...)."""
    df = df.copy()
    df["cenario"] = df["modelo"] + " (" + df["enriquecimento_socioeconomico"].map({True: "com enriq.", False: "sem enriq."}) + ")"

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    largura = 0.25
    for i, metrica in enumerate(["accuracy", "f1", "roc_auc"]):
        ax.bar([xi + i * largura for xi in x], df[metrica], width=largura, label=metrica)
    ax.set_xticks([xi + largura for xi in x])
    ax.set_xticklabels(df["cenario"], rotation=20, ha="right")
    ax.set_title(titulo)
    ax.legend()
    return fig


def matriz_confusao(cm_df: pd.DataFrame, titulo: str, labels_display: tuple[str, str] = ("0", "1")) -> plt.Figure:
    """`cm_df` como salvo por `evaluate.confusion_matrix_df` (index=real_0/
    real_1, colunas=predito_0/predito_1). Anota cada célula com a contagem
    (ou soma de peso amostral, se pesado) e o % em relação ao total."""
    valores = cm_df.to_numpy()
    total = valores.sum()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(valores, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"Previsto:\n{lbl}" for lbl in labels_display])
    ax.set_yticklabels([f"Real:\n{lbl}" for lbl in labels_display])
    for i in range(2):
        for j in range(2):
            v = valores[i, j]
            pct = 100 * v / total if total else 0
            cor_texto = "white" if v > valores.max() / 2 else "black"
            ax.text(j, i, f"{v:,.0f}\n({pct:.1f}%)", ha="center", va="center", color=cor_texto, fontsize=11)
    ax.set_title("\n".join(textwrap.wrap(titulo, 40)), fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


# Colunas onde valor ALTO e' RUIM (ex.: pobreza) -- invertidas na cor pra
# verde continuar significando "melhor" em todo o heatmap.
_COLUNAS_INVERTIDAS = {"PC_FAMILIAS_POBREZA"}


def bar_perfil_clusters(df: pd.DataFrame, titulo: str) -> plt.Figure:
    """`df` como salvo por `run_municipal_clustering.run()` (index=CLUSTER,
    colunas=features do perfil, já com médias por cluster). Heatmap
    (z-score por coluna, anotado com o valor real) -- com só 3 clusters,
    normalização 0-1 por coluna vira sempre {0, 1, algo no meio} e esconde
    diferenças reais (ex.: dois clusters com pobreza quase igual)."""
    features = [c for c in df.columns if c != "N_MUNICIPIOS"]
    valores = df[features]
    zscore = (valores - valores.mean()) / valores.std()
    for col in _COLUNAS_INVERTIDAS:
        if col in zscore.columns:
            zscore[col] = -zscore[col]

    fig, ax = plt.subplots(figsize=(9, 4 + 0.3 * len(df)))
    im = ax.imshow(zscore.T, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"Cluster {c}\n(n={n})" for c, n in zip(df.index, df["N_MUNICIPIOS"])])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    for i, feat in enumerate(features):
        for j, cluster in enumerate(df.index):
            ax.text(j, i, f"{valores.loc[cluster, feat]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="z-score (relativo aos outros clusters)")
    ax.set_title(titulo)
    return fig
