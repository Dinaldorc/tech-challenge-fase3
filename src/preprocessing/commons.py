"""
Módulos utilitários compartilhados pela pipeline (equivalente ao commons/
do projeto Databricks da Fase 2), portados para pandas puro.

Reconstrói localmente, sem Spark/Delta/Databricks, a mesma lógica usada
em https://github.com/Macedim/tech-challenge-ETL (branch develop).
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (equivalente a commons/config.ipynb)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # raiz do repo tech-challenge-fase3
RAW_PATH = BASE_DIR / "data" / "raw"
BRONZE_PATH = BASE_DIR / "data" / "bronze"
SILVER_PATH = BASE_DIR / "data" / "silver"
GOLD_PATH = BASE_DIR / "data" / "gold"

TS_ALUNO = "TS_ALUNO"
TS_ESTADO = "TS_ESTADO"
TS_MUNICIPIO = "TS_MUNICIPIO"
METAS_BR = "METAS_BR"
METAS_UF = "METAS_UF"
METAS_MUNICIPIO = "METAS_MUNICIPIO"
FT_MACHINE_LEARNING = "FT_MACHINE_LEARNING"
FT_INDICADOR_MUNICIPIO = "FT_INDICADOR_MUNICIPIO"
FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO = "FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO"
ANALISE_NIVEIS_MUNICIPIO = "ANALISE_NIVEIS_MUNICIPIO"

CSV_OPTIONS = dict(sep=";", encoding="ISO-8859-1", low_memory=False)

# ---------------------------------------------------------------------------
# Mapas de negócio (equivalente a commons_imports.ipynb)
# ---------------------------------------------------------------------------
REGIAO_MAP = {
    **{uf: "Norte" for uf in ["AC", "AP", "AM", "PA", "RO", "RR", "TO"]},
    **{uf: "Nordeste" for uf in ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]},
    **{uf: "Centro-Oeste" for uf in ["DF", "GO", "MT", "MS"]},
    **{uf: "Sudeste" for uf in ["ES", "MG", "RJ", "SP"]},
    **{uf: "Sul" for uf in ["PR", "RS", "SC"]},
}

UF_TO_CO_UF = {
    "RO": 11, "AC": 12, "AM": 13, "RR": 14, "PA": 15, "AP": 16, "TO": 17,
    "MA": 21, "PI": 22, "CE": 23, "RN": 24, "PB": 25, "PE": 26, "AL": 27, "SE": 28, "BA": 29,
    "MG": 31, "ES": 32, "RJ": 33, "SP": 35,
    "PR": 41, "SC": 42, "RS": 43,
    "MS": 50, "MT": 51, "GO": 52, "DF": 53,
}

DEPENDENCIA_MAP = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
TIPO_REDE_MAP = {0: "Total", 2: "Estadual", 3: "Municipal", 4: "Privada", 5: "Total (Estadual+Municipal)"}


# ---------------------------------------------------------------------------
# Chave técnica (equivalente a sha2(concat_ws("|", cols), 256))
# ---------------------------------------------------------------------------
def sha256_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    concat = df[cols].astype(str).agg("|".join, axis=1)
    return concat.apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())


# ---------------------------------------------------------------------------
# Leitura / escrita (equivalente a readers.ipynb / writers.ipynb)
# ---------------------------------------------------------------------------
def read_raw_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_PATH / "DADOS" / filename, **CSV_OPTIONS)


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


_PARQUET_OK = _has_parquet_engine()


def write_table(df: pd.DataFrame, layer_path: Path, table_name: str) -> Path:
    """Grava a tabela em Parquet (preferido) ou, se nenhum engine de parquet
    estiver disponível no ambiente, em CSV (fallback automático)."""
    layer_path.mkdir(parents=True, exist_ok=True)
    if _PARQUET_OK:
        out = layer_path / f"{table_name}.parquet"
        df.to_parquet(out, index=False)
    else:
        out = layer_path / f"{table_name}.csv"
        df.to_csv(out, index=False)
    return out


def read_table(layer_path: Path, table_name: str) -> pd.DataFrame:
    parquet_path = layer_path / f"{table_name}.parquet"
    if parquet_path.exists() and _PARQUET_OK:
        return pd.read_parquet(parquet_path)
    csv_path = layer_path / f"{table_name}.csv"
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Validações (equivalente a validators.ipynb)
# ---------------------------------------------------------------------------
def validate_primary_key(df: pd.DataFrame, pk) -> None:
    pk = [pk] if isinstance(pk, str) else pk
    dups = df.duplicated(subset=pk).sum()
    if dups > 0:
        raise ValueError(f"validate_primary_key: {dups} linhas duplicadas em {pk}")


def validate_not_null(df: pd.DataFrame, columns: list[str]) -> None:
    for c in columns:
        n = df[c].isna().sum()
        if n > 0:
            raise ValueError(f"validate_not_null: coluna {c} possui {n} valores nulos")


def validate_years(df: pd.DataFrame, year_column: str = "ANO_REFERENCIA") -> None:
    if df[year_column].nunique() == 0:
        raise ValueError(f"validate_years: nenhum ano encontrado em {year_column}")


def validate_schema(df: pd.DataFrame, expected_schema: list[str]) -> None:
    missing = set(expected_schema) - set(df.columns)
    if missing:
        raise ValueError(f"validate_schema: colunas faltantes {missing}")


def validate_foreign_key(fact_df: pd.DataFrame, dim_df: pd.DataFrame, key: str) -> None:
    orphans = ~fact_df[key].isin(dim_df[key])
    n = orphans.sum()
    if n > 0:
        raise ValueError(f"validate_foreign_key: {n} linhas sem correspondência em {key}")
