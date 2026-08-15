"""
Camada Silver: enriquecimento com regras de negócio (faixas, região,
flags descritivas) + validações de qualidade antes de gravar.

Equivalente pandas de notebooks/silver/bz_sv_*.ipynb.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import commons as c


def _add_ano_carga(df: pd.DataFrame) -> pd.DataFrame:
    df["ANO_CARGA"] = pd.to_datetime(df["TS_PROCESSAMENTO"]).dt.year
    df["MES_CARGA"] = pd.to_datetime(df["TS_PROCESSAMENTO"]).dt.month
    return df


def build_ts_aluno_silver() -> pd.DataFrame:
    df = c.read_table(c.BRONZE_PATH, c.TS_ALUNO).copy()

    bins = [-np.inf, 649.99, 699.99, 742.99, 799.99, np.inf]
    labels = ["< 650", "650-699", "700-742", "743-799", "800+"]
    df["FAIXA_PROFICIENCIA"] = pd.cut(df["VL_PROFICIENCIA_LP"], bins=bins, labels=labels)
    df["FAIXA_PROFICIENCIA"] = df["FAIXA_PROFICIENCIA"].astype(object)
    df.loc[df["VL_PROFICIENCIA_LP"].isna(), "FAIXA_PROFICIENCIA"] = "Não avaliado"

    df["IN_PARTICIPOU_AVALIACAO"] = (
        (df["IN_PRESENCA_LP"] == 1) & (df["IN_PREENCHIMENTO_LP"] == 1)
    ).astype(int)
    df["DS_PARTICIPACAO"] = np.where(df["IN_PARTICIPOU_AVALIACAO"] == 1, "Participou", "Não Participou")
    df["DS_SITUACAO_AVALIACAO"] = np.where(df["VL_PROFICIENCIA_LP"].notna(), "Avaliado", "Não Avaliado")

    df["CO_MUNICIPIO_IBGE"] = df["CO_MUNICIPIO"].astype(str).str.zfill(7)
    df["NO_MUNICIPIO_UF"] = df["NO_MUNICIPIO"] + " - " + df["SG_UF"]
    df["DS_DEPENDENCIA"] = df["TP_DEPENDENCIA"].map(c.DEPENDENCIA_MAP)
    df["DS_SERIE"] = df["TP_SERIE"].map({2: "2º Ano do Ensino Fundamental"}).fillna(df["TP_SERIE"].astype(str))
    df["DS_ALFABETIZADO"] = df["IN_ALFABETIZADO"].map({1: "Sim", 0: "Não"})
    df["REGIAO"] = df["SG_UF"].map(c.REGIAO_MAP)
    df = _add_ano_carga(df)

    c.validate_primary_key(df, "SK_ALUNO")
    c.validate_not_null(df, ["SK_ALUNO", "NU_ANO_AVALIACAO"])
    df = df.rename(columns={"NU_ANO_AVALIACAO": "ANO_REFERENCIA"})
    c.validate_years(df)

    c.write_table(df, c.SILVER_PATH, c.TS_ALUNO)
    return df


def _add_faixa_municipio_estado(df: pd.DataFrame) -> pd.DataFrame:
    bins_lp = [-np.inf, 699.99, 742.99, 799.99, np.inf]
    labels_lp = ["< 700", "700-742", "743-799", "800+"]
    df["FAIXA_MEDIA_LP"] = pd.cut(df["VL_MEDIA_LP"], bins=bins_lp, labels=labels_lp).astype(object)

    bins_alf = [-np.inf, 39.99, 59.99, 79.99, np.inf]
    labels_alf = ["Muito Baixa", "Baixa", "Boa", "Excelente"]
    df["FAIXA_ALFABETIZACAO"] = pd.cut(df["PC_ALUNO_ALFABETIZADO"], bins=bins_alf, labels=labels_alf).astype(object)

    nivel_cols = [f"PC_ALUNO_NIVEL_{i}_LP" for i in range(9)]
    df["IN_POSSUI_DISTRIBUICAO_NIVEIS"] = df[nivel_cols].notna().any(axis=1).astype(int)
    df["REGIAO"] = df["SG_UF"].map(c.REGIAO_MAP)
    return df


def build_ts_municipio_silver() -> pd.DataFrame:
    df = c.read_table(c.BRONZE_PATH, c.TS_MUNICIPIO).copy()
    df = _add_faixa_municipio_estado(df)
    df["NO_MUNICIPIO_UF"] = df["NO_MUNICIPIO"] + " - " + df["SG_UF"]
    df = _add_ano_carga(df)

    c.validate_primary_key(df, "SK_MUNICIPIO")
    c.validate_not_null(df, ["SK_MUNICIPIO", "NU_ANO_AVALIACAO"])
    df = df.rename(columns={"NU_ANO_AVALIACAO": "ANO_REFERENCIA"})
    c.validate_years(df)

    c.write_table(df, c.SILVER_PATH, c.TS_MUNICIPIO)
    return df


def build_ts_estado_silver() -> pd.DataFrame:
    df = c.read_table(c.BRONZE_PATH, c.TS_ESTADO).copy()
    df = _add_faixa_municipio_estado(df)
    df = _add_ano_carga(df)

    c.validate_primary_key(df, "SK_ESTADO")
    c.validate_not_null(df, ["SK_ESTADO", "NU_ANO_AVALIACAO"])
    df = df.rename(columns={"NU_ANO_AVALIACAO": "ANO_REFERENCIA"})
    c.validate_years(df)

    c.write_table(df, c.SILVER_PATH, c.TS_ESTADO)
    return df


def build_metas_silver() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta_cols = [f"META_FINAL_{y}" for y in range(2024, 2031)]

    br = c.read_table(c.BRONZE_PATH, c.METAS_BR).copy()
    br["QTD_METAS_DEFINIDAS"] = br[meta_cols].notna().sum(axis=1)
    br["IN_POSSUI_META"] = (br["QTD_METAS_DEFINIDAS"] > 0).astype(int)
    br["IN_POSSUI_RESULTADO"] = br["PC_ALUNO_ALFABETIZADO_2024"].notna().astype(int)
    c.validate_not_null(br, ["ANO_REFERENCIA"])
    c.write_table(br, c.SILVER_PATH, c.METAS_BR)

    uf = c.read_table(c.BRONZE_PATH, c.METAS_UF).copy()
    uf["REGIAO"] = uf["SIGLA_UF"].map(c.REGIAO_MAP)
    uf["QTD_METAS_DEFINIDAS"] = uf[meta_cols].notna().sum(axis=1)
    uf["IN_POSSUI_META"] = (uf["QTD_METAS_DEFINIDAS"] > 0).astype(int)
    c.validate_primary_key(uf, ["ANO_REFERENCIA", "SIGLA_UF"])
    c.write_table(uf, c.SILVER_PATH, c.METAS_UF)

    mun = c.read_table(c.BRONZE_PATH, c.METAS_MUNICIPIO).copy()
    mun["QTD_METAS_DEFINIDAS"] = mun[meta_cols].notna().sum(axis=1)
    mun["IN_POSSUI_META"] = (mun["QTD_METAS_DEFINIDAS"] > 0).astype(int)
    # CO_NIVEL_ALFABETIZACAO já vem do INEP na mesma escala 0..5 usada na
    # classificação de município (0 = Abaixo do Nível 1, 1..5 = Nível 1..5)
    mun["DS_NIVEL_ALFABETIZACAO"] = np.where(
        mun["CO_NIVEL_ALFABETIZACAO"] == 0, "Abaixo do Nível 1",
        "Nível " + mun["CO_NIVEL_ALFABETIZACAO"].astype(str)
    )
    c.validate_primary_key(mun, ["ANO_REFERENCIA", "CO_MUNICIPIO"])
    c.write_table(mun, c.SILVER_PATH, c.METAS_MUNICIPIO)

    return br, uf, mun


def run_all() -> None:
    print("Silver: TS_ALUNO...")
    aluno = build_ts_aluno_silver()
    print(f"  -> {len(aluno):,} linhas")
    print("Silver: TS_MUNICIPIO...")
    municipio = build_ts_municipio_silver()
    print(f"  -> {len(municipio):,} linhas")
    print("Silver: TS_ESTADO...")
    estado = build_ts_estado_silver()
    print(f"  -> {len(estado):,} linhas")
    print("Silver: METAS...")
    br, uf, mun = build_metas_silver()
    print(f"  -> BR={len(br)} UF={len(uf)} MUNICIPIO={len(mun)}")


if __name__ == "__main__":
    run_all()
