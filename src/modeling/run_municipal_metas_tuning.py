"""
Otimização de hiperparâmetros do modelo municipal de metas (pergunta 4 da
estratégia) via `GridSearchCV` com validação cruzada estratificada -- exigência
explícita do Tech Challenge ("estratégias de otimização e validação dos
modelos... aumentar a capacidade de generalização e reduzir overfitting").

Busca só no conjunto de TREINO (2024, indicadores de 2023) via 5-fold CV;
o conjunto de TESTE (2025) fica intocado até a avaliação final, pra manter
a mesma garantia de generalização que já tínhamos no baseline.

python -m src.modeling.run_municipal_metas_tuning
"""
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from ..evaluation import evaluate as ev
from ..preprocessing import commons as c
from . import municipal_metas as mm

REPORT_PATH = c.BASE_DIR / "reports" / "municipal_metas_tuning_cv_results.csv"

PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [4, 6, 8, None],
    "classifier__min_samples_leaf": [5, 20, 50],
}


def run() -> None:
    df = mm.build_dataset()
    train_df, test_df = mm.split_by_year(df)
    X_train, y_train = mm.select_features(train_df)
    X_test, y_test = mm.select_features(test_df)

    pipeline = mm.build_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Buscando entre {len(PARAM_GRID['classifier__n_estimators']) * len(PARAM_GRID['classifier__max_depth']) * len(PARAM_GRID['classifier__min_samples_leaf'])} combinacoes x 5 folds...")
    grid = GridSearchCV(pipeline, PARAM_GRID, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"\nMelhores hiperparametros (por CV, AUC medio = {grid.best_score_:.4f}):")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")

    cv_results = pd.DataFrame(grid.cv_results_).sort_values("rank_test_score")
    cols = [c for c in cv_results.columns if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]
    cv_results[cols].to_csv(REPORT_PATH, index=False)
    print(f"\nResultados completos da CV salvos em {REPORT_PATH}")

    # Avaliacao final no teste (2025), nunca visto durante a busca -- compara
    # com o baseline (RandomForest default: n_estimators=300, max_depth=6,
    # min_samples_leaf=20, AUC=0,6596 no teste -- ver run_municipal_metas.py).
    melhor_modelo = grid.best_estimator_
    metrics = ev.evaluate_model(melhor_modelo, X_test, y_test)
    print(
        f"\nMelhor modelo (via CV) no teste 2025 -- "
        f"acc={metrics['accuracy']:.4f} | precision={metrics['precision']:.4f} | "
        f"recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f} | auc={metrics['roc_auc']:.4f}"
    )
    print("(baseline sem tuning: acc=0,6389 | precision=0,7769 | recall=0,6925 | f1=0,7323 | auc=0,6596)")


if __name__ == "__main__":
    run()
