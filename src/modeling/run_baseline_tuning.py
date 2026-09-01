"""
Otimização de hiperparâmetros do modelo de aluno via `RandomizedSearchCV`
com validação cruzada -- mesma exigência do Tech Challenge atendida pro
modelo municipal em `run_municipal_metas_tuning.py`.

Diferente do municipal (~5,4 mil linhas, GridSearchCV completo cabe fácil),
o treino de aluno tem ~1,56 milhão de linhas -- greid search exaustivo com
CV ficaria caro demais. Busca numa AMOSTRA do treino (300 mil linhas,
estratificada por TARGET) com `RandomizedSearchCV`, depois re-treina os
hiperparâmetros vencedores no treino completo e avalia no teste real
(2025), igual ao municipal. Já sabíamos, de testes manuais anteriores
(`max_depth` 3/4/5/10 -- ver README "Evoluções futuras"), que profundidade
maior tende a ajudar; esta busca cobre um grid mais amplo de forma
sistemática.

python -m src.modeling.run_baseline_tuning
"""
from __future__ import annotations
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from ..evaluation import evaluate as ev
from ..preprocessing import commons as c
from . import features as feat
from . import pipeline as pl
from . import split as sp

REPORT_PATH = c.BASE_DIR / "reports" / "baseline_tuning_cv_results.csv"
UF_REPORT_PATH = c.BASE_DIR / "reports" / "baseline_metricas_por_uf_tuned.csv"

SAMPLE_SIZE = 300_000
PARAM_DIST = {
    "classifier__n_estimators": randint(100, 400),
    "classifier__max_depth": [10, 15, 20, 30, None],
    "classifier__min_samples_leaf": randint(10, 200),
}


def run() -> None:
    df = c.read_table(c.GOLD_PATH, c.FT_MACHINE_LEARNING)
    train_df, test_df = sp.split_2025(df)

    _, amostra = train_test_split(
        train_df, test_size=SAMPLE_SIZE / len(train_df), stratify=train_df["TARGET"], random_state=42,
    )
    X_amostra, y_amostra = feat.select_features(amostra)
    w_amostra = feat.select_weights(amostra)

    pipeline = pl.build_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print(f"Buscando em amostra de {len(amostra):,} linhas (10 combinacoes x 3 folds)...")
    busca = RandomizedSearchCV(
        pipeline, PARAM_DIST, n_iter=10, scoring="roc_auc", cv=cv, random_state=42, n_jobs=-1,
    )
    busca.fit(X_amostra, y_amostra, classifier__sample_weight=w_amostra)

    print(f"\nMelhores hiperparametros (por CV na amostra, AUC medio = {busca.best_score_:.4f}):")
    for k, v in busca.best_params_.items():
        print(f"  {k}: {v}")

    cv_results = pd.DataFrame(busca.cv_results_).sort_values("rank_test_score")
    cols = [c for c in cv_results.columns if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "rank_test_score")]
    cv_results[cols].to_csv(REPORT_PATH, index=False)
    print(f"Resultados completos da CV salvos em {REPORT_PATH}")

    # Re-treina os hiperparametros vencedores no TREINO COMPLETO (nao so a
    # amostra da busca) e avalia no teste real de 2025.
    print("\nRe-treinando no treino completo com os hiperparametros vencedores...")
    X_train, y_train = feat.select_features(train_df)
    X_test, y_test = feat.select_features(test_df)
    w_train = feat.select_weights(train_df)
    w_test = feat.select_weights(test_df)

    params_finais = {k.replace("classifier__", ""): v for k, v in busca.best_params_.items()}
    modelo_final = pl.build_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1, **params_finais))
    modelo_final.fit(X_train, y_train, classifier__sample_weight=w_train)

    metrics = ev.evaluate_model(modelo_final, X_test, y_test, sample_weight=w_test)
    print(
        f"\nModelo tunado no teste 2025 -- acc={metrics['accuracy']:.4f} | "
        f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | "
        f"f1={metrics['f1']:.4f} | auc={metrics['roc_auc']:.4f}"
    )
    print("(baseline sem tuning, max_depth=10: acc=0,6109 | precision=0,6102 | recall=0,9383 | f1=0,7395 | auc=0,6383)")

    uf_df = ev.evaluate_by_group(modelo_final, X_test, y_test, "SG_UF", sample_weight=w_test)
    n_zero, n_full = (uf_df["recall"] == 0).sum(), (uf_df["recall"] >= 0.99).sum()
    print(f"UFs degeneradas (recall=0 ou >=0.99): {n_zero + n_full}/{len(uf_df)} (baseline sem tuning: 18/27)")
    uf_df.to_csv(UF_REPORT_PATH, index=False)
    print(f"Quebra por UF salva em {UF_REPORT_PATH}")


if __name__ == "__main__":
    run()
