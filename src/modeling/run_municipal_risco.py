"""
Responde a pergunta de negócio "quais municípios apresentam maior risco
educacional?" -- usa o modelo de metas (municipal_metas.py, validado em
run_municipal_metas.py) treinado com TODOS os anos disponíveis (2023→2024
e 2024→2025) pra pontuar o risco de CADA município não bater a meta no
próximo ciclo, usando os indicadores mais recentes (2025) como entrada.

Valida a pontuação cruzando com o INDICE_RISCO_ESTRUTURAL já existente em
ANALISE_NIVEIS_MUNICIPIO (calculado de forma independente, a partir da
distribuição de níveis de proficiência -- sem usar o modelo de metas).

python -m src.modeling.run_municipal_risco
"""
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..preprocessing import commons as c
from . import municipal_metas as mm

RANKING_PATH = c.BASE_DIR / "reports" / "municipal_ranking_risco.csv"


def run() -> pd.DataFrame:
    df = mm.build_dataset()

    # Modelo "de producao": usa todas as transicoes disponiveis (2024 e
    # 2025), nao so a fatia de treino do script de validacao -- aqui o
    # objetivo e' pontuar o presente, nao medir generalizacao (isso ja foi
    # validado em run_municipal_metas.py).
    # Hiperparametros otimizados via GridSearchCV -- ver
    # run_municipal_metas_tuning.py e a nota em run_municipal_metas.py.
    X_full, y_full = mm.select_features(df)
    model = mm.build_pipeline(
        RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=20, n_jobs=-1, random_state=42)
    )
    model.fit(X_full, y_full)

    # Projeta o risco do proximo ciclo usando os indicadores mais recentes
    # (2025) como "ano anterior" -- ou seja, reaproveita o desempenho de
    # 2025 no lugar de PC_ALUNO_ALFABETIZADO_ANTERIOR etc.
    meta = c.read_table(c.GOLD_PATH, c.FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO)
    atual_2025 = meta[meta["ANO_REFERENCIA"] == 2025][
        ["CO_MUNICIPIO", "SG_UF", "REGIAO", "PC_ALUNO_ALFABETIZADO", "DIF_META_ALFABETIZACAO", "IN_META_ATINGIDA"]
    ].rename(columns={
        "PC_ALUNO_ALFABETIZADO": "PC_ALUNO_ALFABETIZADO_ANTERIOR",
        "DIF_META_ALFABETIZACAO": "DIF_META_ALFABETIZACAO_ANTERIOR",
        "IN_META_ATINGIDA": "IN_META_ATINGIDA_ANTERIOR",
    })

    from ..preprocessing.gold import _build_dim_municipio_socioeconomico
    dim = _build_dim_municipio_socioeconomico()
    projecao = atual_2025.merge(dim, on="CO_MUNICIPIO", how="left")

    X_proj = projecao[mm.NUMERIC_COLS + mm.CATEGORICAL_COLS].copy()
    for col in X_proj.columns:
        dtype = X_proj[col].dtype
        if pd.api.types.is_extension_array_dtype(dtype) and pd.api.types.is_numeric_dtype(dtype):
            X_proj[col] = X_proj[col].astype("float64")

    projecao["PROBABILIDADE_RISCO_NAO_ATINGIR_META"] = model.predict_proba(X_proj)[:, 0]

    ranking = projecao[[
        "CO_MUNICIPIO", "SG_UF", "REGIAO", "PC_ALUNO_ALFABETIZADO_ANTERIOR",
        "PROBABILIDADE_RISCO_NAO_ATINGIR_META",
    ]].rename(columns={"PC_ALUNO_ALFABETIZADO_ANTERIOR": "PC_ALUNO_ALFABETIZADO_2025"})
    ranking = ranking.sort_values("PROBABILIDADE_RISCO_NAO_ATINGIR_META", ascending=False).reset_index(drop=True)

    print(f"Municipios pontuados: {len(ranking):,}")
    print("\nTop 15 municipios de MAIOR risco (projecao pro proximo ciclo):")
    print(ranking.head(15).to_string(index=False))

    # Validacao cruzada com o indice de risco estrutural ja existente
    # (calculado sem usar o modelo de metas -- so a distribuicao de niveis
    # de proficiencia em 2025).
    niveis = c.read_table(c.GOLD_PATH, c.ANALISE_NIVEIS_MUNICIPIO)
    niveis_2025 = niveis[niveis["ANO_REFERENCIA"] == 2025][["CO_MUNICIPIO", "INDICE_RISCO_ESTRUTURAL"]]
    comparacao = ranking.merge(niveis_2025, on="CO_MUNICIPIO", how="inner")
    corr = comparacao["PROBABILIDADE_RISCO_NAO_ATINGIR_META"].corr(comparacao["INDICE_RISCO_ESTRUTURAL"])
    print(f"\nCorrelacao com o INDICE_RISCO_ESTRUTURAL (calculado independente, so por niveis de proficiencia): {corr:.3f}")

    ranking.to_csv(RANKING_PATH, index=False)
    print(f"\nRanking completo salvo em {RANKING_PATH}")
    return ranking


if __name__ == "__main__":
    run()
