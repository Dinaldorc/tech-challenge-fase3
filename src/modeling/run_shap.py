"""
SHAP no melhor cenário do passo 4 (RandomForest + enriquecimento
socioeconômico) -- investiga se REGIAO/SG_UF dominam a decisão do jeito
que a disparidade regional do passo 5 sugere, e confirma o peso real das
3 features socioeconômicas (Seção 8 da EDA).

python -m src.modeling.run_shap
"""
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier

from ..evaluation import explain as ex
from ..preprocessing import commons as c
from . import features as feat
from . import pipeline as pl
from . import split as sp

REPORT_PATH = c.BASE_DIR / "reports" / "shap_importancia.csv"
REGIAO_REPORT_PATH = c.BASE_DIR / "reports" / "shap_efeito_regiao.csv"


def run() -> None:
    df = c.read_table(c.GOLD_PATH, c.FT_MACHINE_LEARNING)
    train_df, test_df = sp.split_2025(df)

    X_train, y_train = feat.select_features(train_df, include_socioeconomico=True)
    X_test, _ = feat.select_features(test_df, include_socioeconomico=True)
    numeric_cols, categorical_cols = pl.get_column_types(include_socioeconomico=True)

    model = pl.build_pipeline(
        RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42),
        include_socioeconomico=True,
    )
    model.fit(X_train, y_train, classifier__sample_weight=feat.select_weights(train_df))

    shap_values, feature_names, X_sample = ex.compute_shap_values(model, X_test, numeric_cols, categorical_cols)

    print("Importancia media (|SHAP|) por variavel original:")
    variaveis = feat.get_feature_columns(include_socioeconomico=True)
    importancia = ex.importancia_por_variavel_original(shap_values, feature_names, variaveis)
    print(importancia.round(4).to_string())
    importancia.round(6).to_csv(REPORT_PATH, header=["importancia_media_abs_shap"])
    print(f"\nSalvo em {REPORT_PATH}")

    print("\nTop 15 colunas expandidas por |SHAP| medio (mostra quais categorias pesam mais):")
    top_expandido = ex.importancia_media_por_feature_expandida(shap_values, feature_names)
    print(top_expandido.head(15).round(4).to_string())

    print("\nEfeito medio (com sinal) de cada REGIAO na previsao de 'alfabetizado':")
    efeito_regiao = ex.efeito_medio_por_categoria(shap_values, feature_names, X_sample, "REGIAO")
    print(efeito_regiao.to_string(index=False))
    efeito_regiao.to_csv(REGIAO_REPORT_PATH, index=False)
    print(f"\nSalvo em {REGIAO_REPORT_PATH}")


if __name__ == "__main__":
    run()
