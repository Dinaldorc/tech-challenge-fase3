"""
Camada Bronze: ingestão do raw com casting de tipos, padronização de texto
e geração de chave técnica (SK_*) via hash SHA-256.

Equivalente pandas de notebooks/bronze/raw_bz_*.ipynb.
"""
from __future__ import annotations
import pandas as pd
from datetime import datetime

from . import commons as c


def _stamp(df: pd.DataFrame) -> pd.DataFrame:
    df["DT_PROCESSAMENTO"] = datetime.now().date()
    df["TS_PROCESSAMENTO"] = datetime.now()
    return df


_TS_ALUNO_COLS = [
    "NU_ANO_AVALIACAO", "CO_UF", "SG_UF", "ID_ALUNO", "TP_SERIE", "ID_ESCOLA",
    "TP_DEPENDENCIA", "CO_MUNICIPIO", "NO_MUNICIPIO", "IN_PRESENCA_LP",
    "IN_PREENCHIMENTO_LP", "VL_PESO_ALUNO_LP", "VL_PROFICIENCIA_LP", "IN_ALFABETIZADO",
]


def build_ts_aluno() -> pd.DataFrame:
    df = c.read_raw_csv_all_years("TS_ALUNO.csv", usecols=_TS_ALUNO_COLS)
    df["NU_ANO_AVALIACAO"] = df["NU_ANO_AVALIACAO"].astype(int)
    df["CO_UF"] = df["CO_UF"].astype(int)
    df["ID_ALUNO"] = df["ID_ALUNO"].astype("int64")
    # Nullable (Int64) porque a base de 2025 tem 628 registros preliminares
    # sem ID_ESCOLA/TP_DEPENDENCIA/CO_MUNICIPIO preenchidos (ver README).
    df["ID_ESCOLA"] = df["ID_ESCOLA"].astype("Int64")
    df["TP_SERIE"] = df["TP_SERIE"].astype(int)
    df["TP_DEPENDENCIA"] = df["TP_DEPENDENCIA"].astype("Int64")
    df["CO_MUNICIPIO"] = df["CO_MUNICIPIO"].astype("Int64")
    df["IN_PRESENCA_LP"] = pd.to_numeric(df["IN_PRESENCA_LP"], errors="coerce").astype("Int64")
    df["IN_PREENCHIMENTO_LP"] = pd.to_numeric(df["IN_PREENCHIMENTO_LP"], errors="coerce").astype("Int64")
    df["IN_ALFABETIZADO"] = pd.to_numeric(df["IN_ALFABETIZADO"], errors="coerce").astype("Int64")
    df["VL_PESO_ALUNO_LP"] = pd.to_numeric(df["VL_PESO_ALUNO_LP"], errors="coerce")
    df["VL_PROFICIENCIA_LP"] = pd.to_numeric(df["VL_PROFICIENCIA_LP"], errors="coerce")
    df["NO_MUNICIPIO"] = df["NO_MUNICIPIO"].str.strip().str.title().astype("category")
    df["SG_UF"] = df["SG_UF"].str.strip().str.upper().astype("category")

    df["SK_ALUNO"] = c.surrogate_key(df, ["NU_ANO_AVALIACAO", "ID_ALUNO"])
    df = _stamp(df)
    c.write_table(df, c.BRONZE_PATH, c.TS_ALUNO)
    return df


def build_ts_municipio() -> pd.DataFrame:
    df = c.read_raw_csv_all_years("TS_MUNICIPIO.csv")
    df["NU_ANO_AVALIACAO"] = df["NU_ANO_AVALIACAO"].astype(int)
    df["CO_UF"] = df["CO_UF"].astype(int)
    df["CO_MUNICIPIO"] = df["CO_MUNICIPIO"].astype(int)
    df["TP_SERIE"] = df["TP_SERIE"].astype(int)
    df["ID_TIPO_REDE"] = df["ID_TIPO_REDE"].astype(int)
    for col in ["PC_ALUNO_ALFABETIZADO", "VL_MEDIA_LP"] + [f"PC_ALUNO_NIVEL_{i}_LP" for i in range(9)]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["NO_MUNICIPIO"] = df["NO_MUNICIPIO"].str.strip().str.title()
    df["SG_UF"] = df["SG_UF"].str.strip().str.upper()
    df["DS_TIPO_REDE"] = df["ID_TIPO_REDE"].map(c.TIPO_REDE_MAP)

    df = df.drop_duplicates(subset=["NU_ANO_AVALIACAO", "CO_MUNICIPIO", "TP_SERIE", "ID_TIPO_REDE"])
    df["SK_MUNICIPIO"] = c.sha256_key(df, ["NU_ANO_AVALIACAO", "CO_MUNICIPIO", "TP_SERIE", "ID_TIPO_REDE"])
    df = _stamp(df)
    c.write_table(df, c.BRONZE_PATH, c.TS_MUNICIPIO)
    return df


def build_ts_estado() -> pd.DataFrame:
    df = c.read_raw_csv_all_years("TS_ESTADO.csv")
    df["NU_ANO_AVALIACAO"] = df["NU_ANO_AVALIACAO"].astype(int)
    df["CO_UF"] = df["CO_UF"].astype(int)
    df["TP_SERIE"] = df["TP_SERIE"].astype(int)
    df["ID_TIPO_REDE"] = df["ID_TIPO_REDE"].astype(int)
    for col in ["PC_ALUNO_ALFABETIZADO", "VL_MEDIA_LP"] + [f"PC_ALUNO_NIVEL_{i}_LP" for i in range(9)]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["SG_UF"] = df["SG_UF"].str.strip().str.upper()
    df["DS_TIPO_REDE"] = df["ID_TIPO_REDE"].map(c.TIPO_REDE_MAP)

    df = df.drop_duplicates(subset=["NU_ANO_AVALIACAO", "CO_UF", "TP_SERIE", "ID_TIPO_REDE"])
    df["SK_ESTADO"] = c.sha256_key(df, ["NU_ANO_AVALIACAO", "CO_UF", "TP_SERIE", "ID_TIPO_REDE"])
    df = _stamp(df)
    c.write_table(df, c.BRONZE_PATH, c.TS_ESTADO)
    return df


def _clean_meta_value(series: pd.Series) -> pd.Series:
    """Trata '- ' (sem meta) e '> 80' (meta 2030, sempre >=80) vindos do INEP."""
    s = series.astype(str).str.strip()
    s = s.replace({"-": None, "": None})
    s = s.str.replace(">", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def build_metas() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uf_raw = pd.read_excel(c.RAW_PATH / "resultados_e_metas_ufs_2025_v1.xlsx", sheet_name=0, header=1)
    mun_raw = pd.read_excel(c.RAW_PATH / "resultados_e_metas_municipios_2025_v2.xlsx", sheet_name=0, header=1)
    # As planilhas do INEP têm linhas de rodapé/observações soltas no final
    # (ANO/NOME_UF/CO_MUNICIPIO vazios) -- descarta antes de processar.
    uf_raw = uf_raw[uf_raw["NOME_UF"].notna()].copy()
    mun_raw = mun_raw[mun_raw["CO_MUNICIPIO"].notna()].copy()

    meta_cols = [f"META_FINAL_{y}" for y in range(2024, 2031)]

    # --- METAS_BR (linha "Brasil" dentro do arquivo de UFs) ---------------
    br = uf_raw[uf_raw["NOME_UF"] == "Brasil"].copy()
    for col in meta_cols + ["PC_ALUNO_ALFABETIZADO_2023", "PC_ALUNO_ALFABETIZADO_2024", "PC_ALUNO_ALFABETIZADO_2025", "PC_AVALIADOS_LP"]:
        br[col] = _clean_meta_value(br[col])
    br["REDE"] = br["REDE"].str.strip().str.title()
    br["ANO_REFERENCIA"] = br["ANO"].astype(int)
    br["SK_META_ALFABETIZACAO"] = c.sha256_key(
        br.assign(REDE_UPPER=br["REDE"].str.upper()), ["ANO_REFERENCIA", "REDE_UPPER"]
    )
    br = _stamp(br)
    c.write_table(br, c.BRONZE_PATH, c.METAS_BR)

    # --- METAS_UF (demais linhas do arquivo de UFs) ------------------------
    uf = uf_raw[uf_raw["NOME_UF"] != "Brasil"].copy()
    for col in meta_cols + ["PC_ALUNO_ALFABETIZADO_2023", "PC_ALUNO_ALFABETIZADO_2024", "PC_ALUNO_ALFABETIZADO_2025", "PC_AVALIADOS_LP"]:
        uf[col] = _clean_meta_value(uf[col])
    uf["SIGLA_UF"] = uf["SIGLA_UF"].str.strip().str.upper()
    uf["CO_UF"] = uf["CD_UF"].astype(int)
    uf["REDE"] = uf["REDE"].str.strip().str.title()
    uf["ANO_REFERENCIA"] = uf["ANO"].astype(int)
    uf["SK_META_ESTADO"] = c.sha256_key(
        uf.assign(REDE_UPPER=uf["REDE"].str.upper()), ["ANO_REFERENCIA", "SIGLA_UF", "REDE_UPPER"]
    )
    uf = _stamp(uf)
    c.write_table(uf, c.BRONZE_PATH, c.METAS_UF)

    # --- METAS_MUNICIPIO -----------------------------------------------------
    mun = mun_raw.copy()
    for col in meta_cols + ["PC_ALUNO_ALFABETIZADO_2023", "PC_ALUNO_ALFABETIZADO_2024", "PC_ALUNO_ALFABETIZADO_2025", "PC_AVALIADOS_LP"]:
        mun[col] = _clean_meta_value(mun[col])
    mun["CO_UF"] = mun["CO_UF"].astype(int)
    mun["SG_UF"] = mun["SG_UF"].str.strip().str.upper()
    mun["CO_MUNICIPIO"] = mun["CO_MUNICIPIO"].astype(int)
    mun["NO_MUNICIPIO"] = mun["NO_MUNICIPIO"].str.strip().str.title()
    mun["NO_TP_REDE"] = mun["NO_TP_REDE"].str.strip().str.title()
    mun["ANO_REFERENCIA"] = mun["ANO"].astype(int)
    mun["CO_NIVEL_ALFABETIZACAO"] = mun["CO_NIVEL_ALFABETIZACAO"].astype(int)
    mun["SK_META_MUNICIPIO"] = c.sha256_key(mun, ["ANO_REFERENCIA", "CO_MUNICIPIO"])
    mun = _stamp(mun)
    c.write_table(mun, c.BRONZE_PATH, c.METAS_MUNICIPIO)

    return br, uf, mun


def run_all() -> None:
    print("Bronze: TS_ALUNO...")
    aluno = build_ts_aluno()
    print(f"  -> {len(aluno):,} linhas")
    print("Bronze: TS_MUNICIPIO...")
    municipio = build_ts_municipio()
    print(f"  -> {len(municipio):,} linhas")
    print("Bronze: TS_ESTADO...")
    estado = build_ts_estado()
    print(f"  -> {len(estado):,} linhas")
    print("Bronze: METAS (BR/UF/MUNICIPIO)...")
    br, uf, mun = build_metas()
    print(f"  -> BR={len(br)} UF={len(uf)} MUNICIPIO={len(mun)}")


if __name__ == "__main__":
    run_all()
