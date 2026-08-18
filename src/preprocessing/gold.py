"""
Camada Gold: agregação analítica final, pronta para BI / ML.

Equivalente pandas de notebooks/gold/gd_indicador_municipio.ipynb,
gd_metas_municipio.ipynb e gd_machine_learning.ipynb.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import commons as c


def _classificacao(pct: pd.Series) -> pd.Series:
    conditions = [pct >= 80, pct >= 70, pct >= 60, pct >= 50, pct >= 40]
    choices = ["Nível 5", "Nível 4", "Nível 3", "Nível 2", "Nível 1"]
    return pd.Series(np.select(conditions, choices, default="Abaixo do Nível 1"), index=pct.index)


_ORDEM_CLASSIFICACAO = {
    "Abaixo do Nível 1": 0, "Nível 1": 1, "Nível 2": 2,
    "Nível 3": 3, "Nível 4": 4, "Nível 5": 5,
}


def build_ft_machine_learning() -> pd.DataFrame:
    aluno = c.read_table(c.SILVER_PATH, c.TS_ALUNO)
    municipio = c.read_table(c.SILVER_PATH, c.TS_MUNICIPIO)
    estado = c.read_table(c.SILVER_PATH, c.TS_ESTADO)

    # Só faz sentido comparar dentro da mesma rede (ID_TIPO_REDE == TP_DEPENDENCIA)
    df = aluno.merge(
        municipio[["ANO_REFERENCIA", "CO_MUNICIPIO", "ID_TIPO_REDE",
                   "PC_ALUNO_ALFABETIZADO", "VL_MEDIA_LP", "FAIXA_ALFABETIZACAO", "FAIXA_MEDIA_LP"]],
        left_on=["ANO_REFERENCIA", "CO_MUNICIPIO", "TP_DEPENDENCIA"],
        right_on=["ANO_REFERENCIA", "CO_MUNICIPIO", "ID_TIPO_REDE"],
        how="left",
    ).drop(columns=["ID_TIPO_REDE"])

    df = df.merge(
        estado[["ANO_REFERENCIA", "CO_UF", "ID_TIPO_REDE", "PC_ALUNO_ALFABETIZADO", "VL_MEDIA_LP"]]
        .rename(columns={
            "PC_ALUNO_ALFABETIZADO": "PC_ALUNO_ALFABETIZADO_ESTADO",
            "VL_MEDIA_LP": "VL_MEDIA_LP_ESTADO",
        }),
        left_on=["ANO_REFERENCIA", "CO_UF", "TP_DEPENDENCIA"],
        right_on=["ANO_REFERENCIA", "CO_UF", "ID_TIPO_REDE"],
        how="left",
    ).drop(columns=["ID_TIPO_REDE"])

    df["DIF_MEDIA_MUNICIPIO"] = (df["VL_PROFICIENCIA_LP"] - df["VL_MEDIA_LP"]).round(2)
    df["DIF_MEDIA_ESTADO"] = (df["VL_PROFICIENCIA_LP"] - df["VL_MEDIA_LP_ESTADO"]).round(2)
    df["IN_ACIMA_MEDIA_MUNICIPIO"] = (df["VL_PROFICIENCIA_LP"] >= df["VL_MEDIA_LP"]).astype("Int64")
    df["IN_ACIMA_MEDIA_ESTADO"] = (df["VL_PROFICIENCIA_LP"] >= df["VL_MEDIA_LP_ESTADO"]).astype("Int64")
    df["DIF_ALFABETIZACAO_MUNICIPIO"] = (df["PC_ALUNO_ALFABETIZADO_ESTADO"] - df["PC_ALUNO_ALFABETIZADO"]).round(2)

    conditions = [df["DIF_MEDIA_ESTADO"] >= 30, df["DIF_MEDIA_ESTADO"] >= 10,
                  df["DIF_MEDIA_ESTADO"] >= -10, df["DIF_MEDIA_ESTADO"] >= -30]
    choices = ["Muito Acima", "Acima", "Na Média", "Abaixo"]
    df["DESEMPENHO_RELATIVO"] = np.select(conditions, choices, default="Muito Abaixo")
    df.loc[df["DIF_MEDIA_ESTADO"].isna(), "DESEMPENHO_RELATIVO"] = None

    df["TARGET"] = df["IN_ALFABETIZADO"]
    df["IN_PROVA_VALIDA"] = ((df["IN_PRESENCA_LP"] == 1) & (df["IN_PREENCHIMENTO_LP"] == 1)).astype(int)
    df["IN_POSSUI_PROFICIENCIA"] = df["VL_PROFICIENCIA_LP"].notna().astype(int)

    c.write_table(df, c.GOLD_PATH, c.FT_MACHINE_LEARNING)
    return df


def build_ft_indicador_municipio() -> pd.DataFrame:
    municipio = c.read_table(c.SILVER_PATH, c.TS_MUNICIPIO)
    estado = c.read_table(c.SILVER_PATH, c.TS_ESTADO)
    metas_br, metas_uf, metas_mun = (
        c.read_table(c.SILVER_PATH, c.METAS_BR),
        c.read_table(c.SILVER_PATH, c.METAS_UF),
        c.read_table(c.SILVER_PATH, c.METAS_MUNICIPIO),
    )

    # remove agregados "Total" (0, 5) -- só redes específicas (Estadual/Municipal/Privada)
    mun = municipio[~municipio["ID_TIPO_REDE"].isin([0, 5])].copy()

    # meta mais recente por município / UF (equivalente ao row_number().over(...))
    meta_mun_latest = (
        metas_mun.sort_values("ANO_REFERENCIA")
        .groupby("CO_MUNICIPIO", as_index=False).tail(1)
        [["CO_MUNICIPIO", "META_FINAL_2024", "META_FINAL_2025", "META_FINAL_2026",
          "META_FINAL_2027", "META_FINAL_2028", "META_FINAL_2029", "META_FINAL_2030"]]
        .rename(columns={f"META_FINAL_{y}": f"META_MUNICIPIO_{y}" for y in range(2024, 2031)})
    )
    meta_uf_latest = (
        metas_uf.sort_values("ANO_REFERENCIA")
        .groupby("SIGLA_UF", as_index=False).tail(1)
        [["SIGLA_UF"] + [f"META_FINAL_{y}" for y in range(2024, 2031)]]
        .rename(columns={f"META_FINAL_{y}": f"META_UF_{y}" for y in range(2024, 2031)})
    )
    meta_br_latest = (
        metas_br.sort_values("ANO_REFERENCIA").tail(1)
        [[f"META_FINAL_{y}" for y in range(2024, 2031)]]
        .rename(columns={f"META_FINAL_{y}": f"META_BRASIL_{y}" for y in range(2024, 2031)})
    )

    df = mun.merge(meta_mun_latest, on="CO_MUNICIPIO", how="left")
    df = df.merge(meta_uf_latest, left_on="SG_UF", right_on="SIGLA_UF", how="left")
    for k, v in meta_br_latest.iloc[0].items():
        df[k] = v

    df = df.merge(
        estado[["ANO_REFERENCIA", "CO_UF", "ID_TIPO_REDE", "PC_ALUNO_ALFABETIZADO", "VL_MEDIA_LP"]]
        .rename(columns={"PC_ALUNO_ALFABETIZADO": "PC_ALUNO_ALFABETIZADO_ESTADO",
                          "VL_MEDIA_LP": "VL_MEDIA_LP_ESTADO"}),
        on=["ANO_REFERENCIA", "CO_UF", "ID_TIPO_REDE"], how="left",
    )

    def _meta_do_ano(row, prefix):
        col = f"{prefix}_{row['ANO_REFERENCIA']}"
        return row[col] if col in row.index else np.nan

    df["META_ALFABETIZACAO_MUNICIPIO"] = df.apply(lambda r: _meta_do_ano(r, "META_MUNICIPIO"), axis=1)
    df["META_ALFABETIZACAO_UF"] = df.apply(lambda r: _meta_do_ano(r, "META_UF"), axis=1)
    df["META_ALFABETIZACAO_BRASIL"] = df.apply(lambda r: _meta_do_ano(r, "META_BRASIL"), axis=1)

    df["DIF_META_ALFABETIZACAO_MUNICIPIO"] = (df["PC_ALUNO_ALFABETIZADO"] - df["META_ALFABETIZACAO_MUNICIPIO"]).round(2)
    df["DIF_META_ALFABETIZACAO_UF"] = (df["PC_ALUNO_ALFABETIZADO"] - df["META_ALFABETIZACAO_UF"]).round(2)
    df["DIF_META_ALFABETIZACAO_BRASIL"] = (df["PC_ALUNO_ALFABETIZADO"] - df["META_ALFABETIZACAO_BRASIL"]).round(2)

    df["ATINGIU_META_MUNICIPIO"] = np.where(df["DIF_META_ALFABETIZACAO_MUNICIPIO"] >= 0, "Sim", "Não")
    df["ATINGIU_META_UF"] = np.where(df["DIF_META_ALFABETIZACAO_UF"] >= 0, "Sim", "Não")
    df["ATINGIU_META_BRASIL"] = np.where(df["DIF_META_ALFABETIZACAO_BRASIL"] >= 0, "Sim", "Não")

    df["CLASSIFICACAO"] = _classificacao(df["PC_ALUNO_ALFABETIZADO"])
    df["ORDEM_CLASSIFICACAO"] = df["CLASSIFICACAO"].map(_ORDEM_CLASSIFICACAO)

    df["DIF_ALFABETIZACAO_ESTADO"] = (df["PC_ALUNO_ALFABETIZADO"] - df["PC_ALUNO_ALFABETIZADO_ESTADO"]).round(2)
    df["DIF_MEDIA_LP_ESTADO"] = (df["VL_MEDIA_LP"] - df["VL_MEDIA_LP_ESTADO"]).round(2)
    df["ACIMA_MEDIA_ESTADO"] = np.where(df["DIF_MEDIA_LP_ESTADO"] >= 0, "Sim", "Não")

    # série temporal por município + rede
    df = df.sort_values(["CO_MUNICIPIO", "ID_TIPO_REDE", "ANO_REFERENCIA"])
    grp = df.groupby(["CO_MUNICIPIO", "ID_TIPO_REDE"])
    df["VARIACAO_ALFABETIZACAO"] = grp["PC_ALUNO_ALFABETIZADO"].diff().round(2)
    df["VARIACAO_MEDIA_LP"] = grp["VL_MEDIA_LP"].diff().round(2)
    df["TENDENCIA"] = np.select(
        [df["VARIACAO_ALFABETIZACAO"] > 0, df["VARIACAO_ALFABETIZACAO"] < 0],
        ["Melhorou", "Piorou"], default="Estável",
    )
    df.loc[df["VARIACAO_ALFABETIZACAO"].isna(), "TENDENCIA"] = None

    c.write_table(df, c.GOLD_PATH, c.FT_INDICADOR_MUNICIPIO)
    return df


def build_metas_municipio_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    municipio = c.read_table(c.SILVER_PATH, c.TS_MUNICIPIO)
    metas_mun = c.read_table(c.SILVER_PATH, c.METAS_MUNICIPIO)

    mun = municipio[municipio["ID_TIPO_REDE"] == 3].copy()  # rede Municipal

    meta_latest = (
        metas_mun.sort_values("ANO_REFERENCIA")
        .groupby("CO_MUNICIPIO", as_index=False).tail(1)
    )
    meta_cols = [f"META_FINAL_{y}" for y in range(2024, 2031)]
    df = mun.merge(meta_latest[["CO_MUNICIPIO"] + meta_cols], on="CO_MUNICIPIO", how="left")

    def _meta_ano_avaliacao(row):
        col = f"META_FINAL_{row['ANO_REFERENCIA']}"
        return row[col] if col in row.index and pd.notna(row.get(col)) else np.nan

    df["META_ANO_AVALIACAO"] = df.apply(_meta_ano_avaliacao, axis=1)
    LIMIAR_PADRAO = 80.0
    df["DIF_META_ALFABETIZACAO"] = np.where(
        df["META_ANO_AVALIACAO"].notna(),
        (df["PC_ALUNO_ALFABETIZADO"] - df["META_ANO_AVALIACAO"]).round(2),
        (df["PC_ALUNO_ALFABETIZADO"] - LIMIAR_PADRAO).round(2),
    )
    df["IN_META_ATINGIDA"] = (df["DIF_META_ALFABETIZACAO"] >= 0).astype(int)
    # Meta 2030 do Compromisso Nacional é universal (80%): quando a planilha do
    # INEP não traz um valor explícito para o município (geralmente porque ele
    # já superou 80% e não há trajetória de meta calculada), assume-se 80.
    META_2030_PADRAO = 80.0
    meta_2030 = df["META_FINAL_2030"].fillna(META_2030_PADRAO)
    df["DISTANCIA_META_2030"] = (df["PC_ALUNO_ALFABETIZADO"] - meta_2030).round(2)
    df["IN_META_2030_ATINGIDA"] = (df["DISTANCIA_META_2030"] >= 0).astype(int)

    ft_meta_vs_resultado = df.copy()
    c.write_table(ft_meta_vs_resultado, c.GOLD_PATH, c.FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO)

    # --- ANALISE_NIVEIS_MUNICIPIO ------------------------------------------
    niv = mun.copy()
    n = {i: niv[f"PC_ALUNO_NIVEL_{i}_LP"] for i in range(9)}
    niv["PC_PERFIL_EXTREMA_DEFASAGEM"] = (n[0] + n[1]).round(2)
    niv["PC_PERFIL_EM_DESENVOLVIMENTO"] = (n[2] + n[3]).round(2)
    niv["PC_PERFIL_LIMITROFE"] = (n[4] + n[1]).round(2)  # replica literalmente a lógica original (reaproveita nível 1)
    niv["PC_PERFIL_AVANCADO"] = (n[5] + n[6]).round(2)
    niv["PC_TAXA_EXCELENCIA"] = (n[7] + n[8]).round(2)

    niv["INDICE_POLARIZACAO"] = (niv["PC_TAXA_EXCELENCIA"] / (niv["PC_PERFIL_EXTREMA_DEFASAGEM"] + 0.01)).round(2)
    soma_risco = n[0] * 4 + n[1] * 3 + n[2] * 2 + n[3] * 1
    soma_defasagem_dev = niv["PC_PERFIL_EXTREMA_DEFASAGEM"] + niv["PC_PERFIL_EM_DESENVOLVIMENTO"]
    niv["INDICE_RISCO_ESTRUTURAL"] = np.where(
        soma_defasagem_dev == 0, 0, (soma_risco / 400.0).round(4)
    )

    c.write_table(niv, c.GOLD_PATH, c.ANALISE_NIVEIS_MUNICIPIO)
    return ft_meta_vs_resultado, niv


def run_all() -> None:
    print("Gold: FT_MACHINE_LEARNING...")
    ml = build_ft_machine_learning()
    print(f"  -> {len(ml):,} linhas, target=IN_ALFABETIZADO: {ml['TARGET'].value_counts(dropna=False).to_dict()}")

    print("Gold: FT_INDICADOR_MUNICIPIO...")
    ind = build_ft_indicador_municipio()
    print(f"  -> {len(ind):,} linhas")

    print("Gold: FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO + ANALISE_NIVEIS_MUNICIPIO...")
    meta_res, niveis = build_metas_municipio_tables()
    print(f"  -> meta_vs_resultado={len(meta_res):,} linhas, niveis={len(niveis):,} linhas")


if __name__ == "__main__":
    run_all()
