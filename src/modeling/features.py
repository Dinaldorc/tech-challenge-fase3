"""
Seleção de features para a FT_MACHINE_LEARNING, formalizando as decisões
tomadas na EDA (notebooks/01_EDA_Alfabetizacao.ipynb, Seções 6, 7 e 9).

Cada lista abaixo documenta o motivo da exclusão -- ver a EDA para o
detalhamento de cada achado.
"""
from __future__ import annotations
import pandas as pd

TARGET_COL = "TARGET"

# Agrupador para split treino/teste: ~70% dos alunos (ID_ALUNO) aparecem em
# mais de um ANO_REFERENCIA na base (ver investigação da Seção 8 -- 1,09M em
# 3 anos, 962k em 2 anos, de 2,95M alunos únicos). Um split aleatório por
# linha vazaria o mesmo aluno entre treino e teste; o split precisa ser
# agrupado por ID_ALUNO (ex.: sklearn GroupShuffleSplit/GroupKFold).
GROUP_COL = "ID_ALUNO"

# Leakage direto: determinístico ou quase-determinístico em relação ao
# TARGET (derivado de VL_PROFICIENCIA_LP ou das flags de participação na
# prova -- Seção 7 da EDA). Nunca deve entrar no modelo.
LEAKAGE_DIRETO = [
    "VL_PROFICIENCIA_LP",
    "IN_ALFABETIZADO",       # e' o proprio TARGET, sem a renomeacao
    "DS_ALFABETIZADO",       # versao string do TARGET
    "IN_PRESENCA_LP",
    "IN_PREENCHIMENTO_LP",
    "VL_PESO_ALUNO_LP",
    "FAIXA_PROFICIENCIA",
    "IN_PARTICIPOU_AVALIACAO",
    "DS_PARTICIPACAO",
    "DS_SITUACAO_AVALIACAO",
    "IN_PROVA_VALIDA",
    "IN_POSSUI_PROFICIENCIA",
    "DIF_MEDIA_MUNICIPIO",
    "DIF_MEDIA_ESTADO",
    "IN_ACIMA_MEDIA_MUNICIPIO",
    "IN_ACIMA_MEDIA_ESTADO",
    "DESEMPENHO_RELATIVO",   # derivado de DIF_MEDIA_ESTADO
]

# Leakage parcial: agregados por município/estado que incluem o próprio
# aluno no cálculo (Seção 9 da EDA). Tratar com cautela -- fora do baseline
# por padrão; poderiam voltar como leave-one-out se o ganho compensar.
LEAKAGE_PARCIAL = [
    "PC_ALUNO_ALFABETIZADO",
    "PC_ALUNO_ALFABETIZADO_ESTADO",
    "VL_MEDIA_LP",
    "VL_MEDIA_LP_ESTADO",
    "FAIXA_ALFABETIZACAO",
    "FAIXA_MEDIA_LP",
    "DIF_ALFABETIZACAO_MUNICIPIO",
]

# Identificadores e metadados de processamento: não têm poder preditivo
# generalizável (ou são constantes na base atual -- só 2º ano avaliado).
IDENTIFICADORES_METADADOS = [
    "SK_ALUNO",
    "ID_ESCOLA",         # alta cardinalidade (~43,6 mil escolas); fora do baseline
    "DT_PROCESSAMENTO",
    "TS_PROCESSAMENTO",
    "ANO_CARGA",          # constante na base atual
    "MES_CARGA",          # constante na base atual
    "TP_SERIE",           # constante na base atual (só 2º ano)
    "DS_SERIE",           # idem, versão texto
]

# Redundantes: mesma informação já coberta por outra coluna mantida.
REDUNDANTES = [
    "CO_UF",              # bijetivo com SG_UF
    "NO_MUNICIPIO",       # texto livre; CO_MUNICIPIO_IBGE ja identifica o municipio
    "NO_MUNICIPIO_UF",    # idem
    "CO_MUNICIPIO_IBGE",  # versao string zero-padded de CO_MUNICIPIO
    "DS_DEPENDENCIA",     # versao texto de TP_DEPENDENCIA
]

# CO_MUNICIPIO fica de fora das features "prontas pro modelo": tem ~5.567
# valores únicos, cardinalidade alta demais pra one-hot direto. Serve como
# chave de junção (já usado para trazer REGIAO/SG_UF e o enriquecimento
# socioeconômico) e pode ser usado depois via encoding específico (ex.:
# target encoding) se fizer diferença no SHAP.
CHAVE_MUNICIPIO = ["CO_MUNICIPIO"]

# Candidatas do enriquecimento externo (Seção 8 da EDA): correlação fraca a
# nível de aluno (|r| entre 0,006 e 0,073) -- entram no baseline como
# candidatas, mas a permanência final é decidida por importância de
# feature / SHAP, não pela correlação isolada.
FEATURES_SOCIOECONOMICAS = [
    "PC_FAMILIAS_POBREZA",
    "RENDA_PER_CAPITA_MEDIA",
    "MEDIA_INSE",
]

# Features seguras (conhecidas antes da prova, sem relação matemática com o
# resultado) -- base do modelo independente do enriquecimento externo.
FEATURES_SEGURAS = [
    "ANO_REFERENCIA",
    "REGIAO",
    "SG_UF",
    "TP_DEPENDENCIA",
]


def get_feature_columns(include_socioeconomico: bool = True) -> list[str]:
    """Lista de colunas seguras para treinar o modelo.

    `include_socioeconomico=False` reproduz o baseline sem o enriquecimento
    externo (pobreza/renda/INSE), para comparação -- ver Seção 8 da EDA.
    """
    cols = list(FEATURES_SEGURAS)
    if include_socioeconomico:
        cols += FEATURES_SOCIOECONOMICAS
    return cols


def select_features(
    df: pd.DataFrame, include_socioeconomico: bool = True
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Retorna (X, y, groups) prontos para split/treino.

    `groups` é o `ID_ALUNO` -- use com GroupShuffleSplit/GroupKFold pra
    evitar que o mesmo aluno apareça em treino e teste (ver GROUP_COL).
    """
    cols = get_feature_columns(include_socioeconomico=include_socioeconomico)
    X = df[cols].copy()
    y = df[TARGET_COL].copy()
    groups = df[GROUP_COL].copy()
    return X, y, groups
