"""
Compara baselines de modelagem -- LogisticRegression x RandomForest, com e
sem o enriquecimento socioeconômico (CadÚnico/Censo/INSE, Seção 8 da EDA)
-- usando o split do passo 2 (só ano de 2025, 70/30 aleatório estratificado
por TARGET -- ver split.py).

python -m src.modeling.run_baseline
"""
from __future__ import annotations
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ..evaluation import evaluate as ev
from ..preprocessing import commons as c
from . import features as feat
from . import pipeline as pl
from . import split as sp

REPORT_PATH = c.BASE_DIR / "reports" / "baseline_comparison.csv"
REGIAO_REPORT_PATH = c.BASE_DIR / "reports" / "baseline_metricas_por_regiao.csv"
UF_REPORT_PATH = c.BASE_DIR / "reports" / "baseline_metricas_por_uf.csv"
UF_INFRA_REPORT_PATH = c.BASE_DIR / "reports" / "baseline_metricas_por_uf_com_infraestrutura.csv"

CLASSIFICADORES = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=1000),
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=50, n_jobs=-1, random_state=42,
    ),
}


def run() -> pd.DataFrame:
    df = c.read_table(c.GOLD_PATH, c.FT_MACHINE_LEARNING)
    train_df, test_df = sp.split_2025(df)

    resultados = []
    modelos_ajustados = {}
    for nome, factory in CLASSIFICADORES.items():
        for include_socio in [False, True]:
            X_train, y_train = feat.select_features(train_df, include_socioeconomico=include_socio)
            X_test, y_test = feat.select_features(test_df, include_socioeconomico=include_socio)

            model = pl.build_pipeline(factory(), include_socioeconomico=include_socio)
            model.fit(X_train, y_train)
            metrics = ev.evaluate_model(model, X_test, y_test)
            metrics["modelo"] = nome
            metrics["enriquecimento_socioeconomico"] = include_socio
            resultados.append(metrics)
            modelos_ajustados[(nome, include_socio)] = (model, X_test, y_test)

            print(
                f"{nome:<18} | socioeconomico={include_socio!s:<5} | "
                f"acc={metrics['accuracy']:.4f} (classe majoritaria={metrics['baseline_classe_majoritaria']:.4f}) | "
                f"precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | "
                f"f1={metrics['f1']:.4f} | auc={metrics['roc_auc']:.4f}"
            )

    resultado_df = pd.DataFrame(resultados)
    cols_resumo = [col for col in resultado_df.columns if col != "confusion_matrix"]
    resultado_df[cols_resumo].to_csv(REPORT_PATH, index=False)
    print(f"\nResumo salvo em {REPORT_PATH}")

    # Melhor cenario (maior ROC-AUC) -- quebra por regiao E por UF pra
    # checar se o modelo reproduz/agrava a disparidade achada na EDA
    # (Secao 5, revisada: ~13pp por regiao, mas 47pp por UF -- CE x RN).
    melhor_chave = resultado_df.loc[resultado_df["roc_auc"].idxmax(), ["modelo", "enriquecimento_socioeconomico"]]
    melhor = (melhor_chave["modelo"], bool(melhor_chave["enriquecimento_socioeconomico"]))
    model, X_test, y_test = modelos_ajustados[melhor]

    print(f"\nMelhor cenario ({melhor[0]}, socioeconomico={melhor[1]}) -- metricas por regiao:")
    regiao_df = ev.evaluate_by_group(model, X_test, y_test, "REGIAO")
    print(regiao_df.to_string(index=False))
    regiao_df.to_csv(REGIAO_REPORT_PATH, index=False)
    print(f"Quebra por regiao salva em {REGIAO_REPORT_PATH}")

    print(f"\nMelhor cenario ({melhor[0]}, socioeconomico={melhor[1]}) -- metricas por UF:")
    uf_df = ev.evaluate_by_group(model, X_test, y_test, "SG_UF")
    print(uf_df.to_string(index=False))
    uf_df.to_csv(UF_REPORT_PATH, index=False)
    print(f"Quebra por UF salva em {UF_REPORT_PATH}")
    n_zero, n_full = (uf_df["recall"] == 0).sum(), (uf_df["recall"] >= 0.99).sum()
    print(f"UFs degeneradas (recall=0 ou >=0.99): {n_zero + n_full}/{len(uf_df)}")

    # Teste extra: infraestrutura escolar (Censo Escolar) em cima do melhor
    # cenario, pra ver se reduz a degeneracao por UF.
    factory_melhor = CLASSIFICADORES[melhor[0]]
    X_train_i, y_train_i = feat.select_features(train_df, include_socioeconomico=True, include_infraestrutura=True)
    X_test_i, y_test_i = feat.select_features(test_df, include_socioeconomico=True, include_infraestrutura=True)
    model_infra = pl.build_pipeline(factory_melhor(), include_socioeconomico=True, include_infraestrutura=True)
    model_infra.fit(X_train_i, y_train_i)
    metrics_infra = ev.evaluate_model(model_infra, X_test_i, y_test_i)
    uf_infra_df = ev.evaluate_by_group(model_infra, X_test_i, y_test_i, "SG_UF")
    n_zero_i, n_full_i = (uf_infra_df["recall"] == 0).sum(), (uf_infra_df["recall"] >= 0.99).sum()

    print(
        f"\n{melhor[0]} + infraestrutura escolar | acc={metrics_infra['accuracy']:.4f} | "
        f"f1={metrics_infra['f1']:.4f} | auc={metrics_infra['roc_auc']:.4f} | "
        f"UFs degeneradas: {n_zero_i + n_full_i}/{len(uf_infra_df)}"
    )
    uf_infra_df.to_csv(UF_INFRA_REPORT_PATH, index=False)
    print(f"Quebra por UF (com infraestrutura) salva em {UF_INFRA_REPORT_PATH}")

    return resultado_df


if __name__ == "__main__":
    run()
