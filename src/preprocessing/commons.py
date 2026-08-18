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

# Anos de avaliação cobertos pela reconstrução (mesmo escopo do projeto original
# na AWS/Databricks da Fase 2: 2023, 2024 e 2025).
YEARS = [2023, 2024, 2025]

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
    # Concatenação vetorizada (Series + Series) em vez de .agg("|".join, axis=1),
    # que é ordens de magnitude mais lenta/mais pesada em memória para milhões
    # de linhas (invoca uma função Python por linha via apply interno do pandas).
    parts = [df[col].astype(str) for col in cols]
    concat = parts[0]
    for p in parts[1:]:
        concat = concat + "|" + p
    return concat.apply(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())


# ---------------------------------------------------------------------------
# Leitura / escrita (equivalente a readers.ipynb / writers.ipynb)
# ---------------------------------------------------------------------------
def read_raw_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_PATH / "DADOS" / filename, **CSV_OPTIONS)


def read_raw_csv_all_years(filename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Lê o mesmo arquivo (ex: TS_ALUNO.csv) em data/raw/DADOS/{ano}/ para
    cada ano em YEARS e concatena, replicando o `recursive_by_year=True`
    do commons/readers.ipynb original.

    `usecols`, quando informado, evita carregar colunas que não usamos
    (ex: respostas brutas item a item de TS_ALUNO, presentes só em alguns
    anos) — reduz bastante o pico de memória para TS_ALUNO (~6M linhas)."""
    frames = []
    for year in YEARS:
        path = RAW_PATH / "DADOS" / str(year) / filename
        if not path.exists():
            continue
        cols = None
        if usecols is not None:
            header = pd.read_csv(path, nrows=0, **CSV_OPTIONS).columns
            cols = [c for c in usecols if c in header]
        df = pd.read_csv(path, usecols=cols, **CSV_OPTIONS)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"Nenhum arquivo {filename} encontrado em data/raw/DADOS/<ano>/")
    return pd.concat(frames, ignore_index=True)


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


def surrogate_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Chave técnica leve (concatenação vetorizada, sem hash criptográfico).
    Usada em tabelas muito grandes (TS_ALUNO, milhões de linhas) onde gerar
    um SHA-256 por linha é caro demais em tempo/memória sem ganho real,
    já que aqui não fazemos MERGE incremental como no Delta Lake original —
    a unicidade da concatenação já basta."""
    parts = [df[col].astype(str) for col in cols]
    concat = parts[0]
    for p in parts[1:]:
        concat = concat + "|" + p
    return concat


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
