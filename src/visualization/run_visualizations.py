"""
Gera as imagens salvas em `images/` a partir dos relatórios já calculados
em `reports/` (não retreina nada -- só visualiza resultados existentes).

python -m src.visualization.run_visualizations
"""
from __future__ import annotations
import pandas as pd

from ..preprocessing import commons as c
from . import charts as ch

REPORTS = c.BASE_DIR / "reports"
IMAGES = c.BASE_DIR / "images"


def run() -> None:
    shap_aluno = pd.read_csv(REPORTS / "shap_importancia.csv", index_col=0).iloc[:, 0]
    fig = ch.bar_importancia_shap(shap_aluno, "Importância média (SHAP) -- modelo de aluno")
    ch.salvar(fig, IMAGES / "shap_importancia_aluno.png")

    shap_municipio = pd.read_csv(REPORTS / "municipal_metas_shap_importancia.csv", index_col=0).iloc[:, 0]
    fig = ch.bar_importancia_shap(shap_municipio, "Importância média (SHAP) -- modelo municipal (metas)")
    ch.salvar(fig, IMAGES / "shap_importancia_municipal.png")

    por_uf = pd.read_csv(REPORTS / "baseline_metricas_por_uf.csv")
    fig = ch.bar_metricas_por_grupo(
        por_uf, "SG_UF", "recall",
        "Recall por UF -- modelo de aluno (laranja = degenerado: 0 ou >=99%)",
        destacar_degenerado=True,
    )
    ch.salvar(fig, IMAGES / "recall_por_uf_aluno.png")

    por_regiao = pd.read_csv(REPORTS / "baseline_metricas_por_regiao.csv")
    fig = ch.bar_metricas_por_grupo(por_regiao, "REGIAO", "recall", "Recall por região -- modelo de aluno")
    ch.salvar(fig, IMAGES / "recall_por_regiao_aluno.png")

    comparacao = pd.read_csv(REPORTS / "baseline_comparison.csv")
    fig = ch.bar_comparacao_modelos(comparacao, "Comparação de baselines -- modelo de aluno")
    ch.salvar(fig, IMAGES / "baseline_comparison_aluno.png")

    perfil_clusters = pd.read_csv(REPORTS / "municipal_clusters_perfil.csv", index_col=0)
    fig = ch.bar_perfil_clusters(perfil_clusters, "Perfil médio por cluster municipal (valores normalizados)")
    ch.salvar(fig, IMAGES / "municipal_clusters_perfil.png")

    print(f"Imagens salvas em {IMAGES}")


if __name__ == "__main__":
    run()
