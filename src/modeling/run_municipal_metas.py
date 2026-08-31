"""
Responde a pergunta de negócio "como prever municípios que podem não
atingir metas futuras?" -- treina em 2024 (usando indicadores de 2023),
testa em 2025 (usando indicadores de 2024), compara LogisticRegression x
RandomForest e roda SHAP no melhor cenário.

python -m src.modeling.run_municipal_metas
"""
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from ..evaluation import evaluate as ev
from ..evaluation import explain as ex
from ..preprocessing import commons as c
from . import municipal_metas as mm

REPORT_PATH = c.BASE_DIR / "reports" / "municipal_metas_comparison.csv"
SHAP_REPORT_PATH = c.BASE_DIR / "reports" / "municipal_metas_shap_importancia.csv"

CLASSIFICADORES = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=20, n_jobs=-1, random_state=42,
    ),
}


def run() -> pd.DataFrame:
    df = mm.build_dataset()
    train_df, test_df = mm.split_by_year(df)
    print(f"Treino (prevendo 2024 c/ indicadores de 2023): {len(train_df):,} municipios")
    print(f"Teste (prevendo 2025 c/ indicadores de 2024): {len(test_df):,} municipios")

    X_train, y_train = mm.select_features(train_df)
    X_test, y_test = mm.select_features(test_df)

    resultados = []
    modelos_ajustados = {}
    for nome, factory in CLASSIFICADORES.items():
        model = mm.build_pipeline(factory())
        model.fit(X_train, y_train)
        metrics = ev.evaluate_model(model, X_test, y_test)
        metrics["modelo"] = nome
        resultados.append(metrics)
        modelos_ajustados[nome] = model

        print(
            f"{nome:<18} | acc={metrics['accuracy']:.4f} (classe majoritaria={metrics['baseline_classe_majoritaria']:.4f}) | "
            f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f} | auc={metrics['roc_auc']:.4f}"
        )

    resultado_df = pd.DataFrame(resultados)
    cols_resumo = [col for col in resultado_df.columns if col != "confusion_matrix"]
    resultado_df[cols_resumo].to_csv(REPORT_PATH, index=False)
    print(f"\nResumo salvo em {REPORT_PATH}")

    melhor_nome = resultado_df.loc[resultado_df["roc_auc"].idxmax(), "modelo"]
    melhor_model = modelos_ajustados[melhor_nome]

    # As metricas acima (precision/recall/f1) sao pra classe "atingiu meta"
    # (pos_label=1, default do evaluate_model) -- a pergunta de negocio e'
    # sobre a classe "NAO atingiu" (0), que e' minoritaria. Acuracia geral
    # engana aqui porque a classe majoritaria (atingiu) ja da' 71% sozinha.
    pred_melhor = melhor_model.predict(X_test)
    print(f"\nRelatorio por classe ({melhor_nome}) -- a classe 0 (NAO atingiu meta) e' a que importa pra priorizacao:")
    print(classification_report(y_test, pred_melhor, target_names=["NAO_atingiu(0)", "atingiu(1)"]))

    print(f"Melhor cenario: {melhor_nome} -- rodando SHAP...")

    shap_values, feature_names, _ = ex.compute_shap_values(
        melhor_model, X_test, mm.NUMERIC_COLS, mm.CATEGORICAL_COLS, sample_size=len(X_test),
    )
    importancia = ex.importancia_por_variavel_original(
        shap_values, feature_names, mm.NUMERIC_COLS + mm.CATEGORICAL_COLS,
    )
    print(importancia.round(4).to_string())
    importancia.round(6).to_csv(SHAP_REPORT_PATH, header=["importancia_media_abs_shap"])
    print(f"\nSalvo em {SHAP_REPORT_PATH}")

    return resultado_df


if __name__ == "__main__":
    run()
