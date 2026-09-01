"""
Responde a pergunta de negócio "quais regiões possuem padrões
semelhantes?" -- clustering (K-Means) dos municípios por perfil
socioeconômico + educacional (2025), e cruzamento com REGIAO/SG_UF pra
ver se os agrupamentos naturais coincidem com a divisão regional oficial
do IBGE ou revelam outra estrutura.

python -m src.modeling.run_municipal_clustering
"""
from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..preprocessing import commons as c
from . import municipal_clustering as mc

PERFIL_PATH = c.BASE_DIR / "reports" / "municipal_clusters_perfil.csv"
MUNICIPIOS_PATH = c.BASE_DIR / "reports" / "municipal_clusters_municipios.csv"
CROSSTAB_PATH = c.BASE_DIR / "reports" / "municipal_clusters_x_regiao.csv"


def escolher_k(X_scaled, k_min: int = 3, k_max: int = 8) -> int:
    melhor_k, melhor_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"  k={k} -> silhouette={score:.4f}")
        if score > melhor_score:
            melhor_k, melhor_score = k, score
    return melhor_k


def run() -> pd.DataFrame:
    perfil = mc.build_perfil_municipios()
    print(f"Municipios com perfil completo: {len(perfil):,}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(perfil[mc.FEATURES_PERFIL])

    print("\nEscolhendo k via silhouette score:")
    k = escolher_k(X_scaled)
    print(f"Melhor k: {k}")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    perfil = perfil.copy()
    perfil["CLUSTER"] = kmeans.fit_predict(X_scaled)

    perfil_cluster = perfil.groupby("CLUSTER")[mc.FEATURES_PERFIL].mean().round(2)
    perfil_cluster["N_MUNICIPIOS"] = perfil.groupby("CLUSTER").size()
    print("\nPerfil medio por cluster:")
    print(perfil_cluster.to_string())
    perfil_cluster.to_csv(PERFIL_PATH)
    print(f"\nSalvo em {PERFIL_PATH}")

    crosstab = pd.crosstab(perfil["CLUSTER"], perfil["REGIAO"], normalize="index").round(3) * 100
    print("\n% de cada REGIAO dentro de cada cluster (cluster segue a regiao oficial ou nao?):")
    print(crosstab.to_string())
    crosstab.to_csv(CROSSTAB_PATH)
    print(f"\nSalvo em {CROSSTAB_PATH}")

    perfil[["CO_MUNICIPIO", "NO_MUNICIPIO", "SG_UF", "REGIAO", "CLUSTER"] + mc.FEATURES_PERFIL].to_csv(
        MUNICIPIOS_PATH, index=False
    )
    print(f"Atribuicao de cluster por municipio salva em {MUNICIPIOS_PATH}")

    return perfil


if __name__ == "__main__":
    run()
