# Documentação Técnica — Tech Challenge Fase 3

## 1. Propósito deste documento

O `README.md` conta a história do projeto para quem quer entender o
problema, os resultados e as decisões analíticas (contexto, perguntas de
negócio, insights, limitações). Este documento tem outro público: alguém
que vai **rodar, manter ou estender o código** — a especificação técnica
reprodutível por trás do README. Cobre arquitetura, schema de dados,
mapa de módulos, espaços de busca de hiperparâmetros e o grafo de
dependência entre os scripts. Onde os dois se sobrepõem, este documento
aponta para a seção correspondente do README em vez de duplicar o texto.

## 2. Arquitetura geral

```
data/raw/ (microdados INEP, CSV/XLSX brutos)
        │
        ▼
   src/preprocessing/bronze.py    -- leitura bruta, tipagem inicial
        │
        ▼
   src/preprocessing/silver.py    -- limpeza, joins, colunas derivadas
        │
        ▼
   src/preprocessing/gold.py      -- 4 tabelas Gold (ver Seção 5.1)
        │
        ├──────────────┬─────────────────────┐
        ▼              ▼                     ▼
  modelo de aluno  modelo municipal    clustering municipal
  (src/modeling/   (src/modeling/      (src/modeling/
   features.py,     municipal_metas.py, municipal_clustering.py,
   pipeline.py,     run_municipal_*.py) run_municipal_clustering.py)
   split.py,
   run_baseline*.py,
   run_shap.py)
        │              │                     │
        └──────────────┴─────────────────────┘
                        ▼
              reports/*.csv (métricas, SHAP, rankings, clusters)
                        │
                        ▼
              src/visualization/*.py → images/*.png
```

Toda a camada `src/evaluation/` (`evaluate.py`, `explain.py`) é
transversal: usada tanto pelo pipeline de aluno quanto pelo municipal,
nunca treina nada sozinha.

## 3. Ambiente técnico

`requirements.txt`:

| Pacote | Uso no projeto |
|---|---|
| `pandas` | toda a manipulação tabular (bronze/silver/gold, features) |
| `pyarrow` | leitura/escrita em Parquet (`commons.write_table`/`read_table`) -- sem ele, fallback automático pra CSV, mais lento e pesado em memória pra `FT_MACHINE_LEARNING` (~6M linhas) |
| `openpyxl` | leitura de `.xlsx` (INSE, metas UF/município) |
| `numpy`, `scipy` | álgebra vetorizada e `scipy.stats.randint` (espaço de busca do `RandomizedSearchCV`) |
| `scikit-learn` | pipelines, modelos, validação cruzada, métricas |
| `matplotlib`, `seaborn` | `src/visualization/charts.py` |
| `shap` | interpretabilidade (`src/evaluation/explain.py`) |
| `jupyter` | `notebooks/01_EDA_Alfabetizacao.ipynb` |
| `python-dotenv`, `pyyaml` | utilitários de configuração |

**Hardware**: a etapa mais pesada é a reconstrução de `TS_ALUNO` →
`FT_MACHINE_LEARNING` (~6,09 milhões de linhas, 3 anos) — recomendado
4-8 GB de RAM livres sem `pyarrow` instalado; com `pyarrow`, o uso de
memória é sensivelmente menor. Todos os scripts de `src/modeling/`
individualmente rodam confortavelmente em qualquer máquina com 8 GB de
RAM.

**Determinismo**: `random_state=42` é fixado em todo ponto que envolve
aleatoriedade (splits, `RandomForestClassifier`, `KMeans`,
`RandomizedSearchCV`) -- reexecutar qualquer script deve reproduzir os
mesmos números documentados no README, byte a byte no caso das tabelas
CSV. Verificado durante este projeto: reexecuções de `run_baseline.py` e
`run_municipal_metas.py` em sessões diferentes reproduziram exatamente
os mesmos valores de matriz de confusão.

## 4. Mapa do código-fonte

### `src/preprocessing/`

| Arquivo | Responsabilidade |
|---|---|
| `commons.py` | paths (`BASE_DIR`, `RAW_PATH`, `BRONZE_PATH`, `SILVER_PATH`, `GOLD_PATH`), nomes de tabela, mapas de negócio (`REGIAO_MAP`, `UF_TO_CO_UF`, `DEPENDENCIA_MAP`, `TIPO_REDE_MAP`), leitura/escrita Parquet-com-fallback-CSV (`read_table`/`write_table`), funções de validação (`validate_primary_key`, `validate_not_null`, `validate_schema`, `validate_foreign_key`) |
| `bronze.py` | leitura bruta dos CSVs do INEP (`TS_ALUNO`, `TS_MUNICIPIO`, `TS_ESTADO`, metas), concatenando os 3 anos (`YEARS = [2023, 2024, 2025]`) |
| `silver.py` | limpeza, tipagem, joins intermediários, colunas derivadas antes da agregação final |
| `gold.py` | monta as 4 tabelas Gold finais (ver Seção 5.1) — inclui `_build_dim_municipio_socioeconomico()`, a dimensão de enriquecimento externo (CadÚnico/Censo/INSE/Censo Escolar) reutilizada por todos os modelos municipais |
| `run_pipeline.py` | orquestra `bronze.run_all() → silver.run_all() → gold.run_all()` |

### `src/modeling/`

| Arquivo | Responsabilidade |
|---|---|
| `features.py` | categorização completa das 50 colunas de `FT_MACHINE_LEARNING` (ver Seção 5.2); `select_features()`, `select_weights()` |
| `pipeline.py` | `ColumnTransformer` (numéricas: `SimpleImputer(median)` + `StandardScaler`; categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`) + `build_pipeline(classifier, ...)` |
| `split.py` | `split_2025()` -- restringe a 2025, split aleatório 70/30 estratificado por `TARGET` |
| `run_baseline.py` | treina/compara os 2 modelos de aluno (com/sem enriquecimento), salva matriz de confusão, quebra por região/UF |
| `run_baseline_tuning.py` | `RandomizedSearchCV` (aluno) — ver Seção 6.1 |
| `run_shap.py` | SHAP do melhor modelo de aluno |
| `municipal_metas.py` | dataset com lag de 1 ano (`build_dataset`), `split_by_year`, `select_features`, `build_pipeline` (município) |
| `run_municipal_metas.py` | treina/compara os modelos municipais, matriz de confusão, calibração de limiar, SHAP |
| `run_municipal_metas_tuning.py` | `GridSearchCV` (município) — ver Seção 6.2 |
| `run_municipal_risco.py` | modelo "de produção" (treinado com todo o histórico) que pontua os 5.500 municípios |
| `municipal_clustering.py` | `build_perfil_municipios()` -- features de perfil por município |
| `run_municipal_clustering.py` | `KMeans` + seleção de `k` via silhouette score |

### `src/evaluation/`

| Arquivo | Responsabilidade |
|---|---|
| `evaluate.py` | `evaluate_model()` (accuracy/precision/recall/F1/AUC/matriz de confusão, com `sample_weight` opcional), `evaluate_by_group()` (quebra por região/UF), `calibrar_limiar_decisao()` (curva precision-recall), `confusion_matrix_df()`, `majority_class_baseline()` |
| `explain.py` | wrapper de `shap.TreeExplainer`: `compute_shap_values()`, `get_expanded_feature_names()` (mapeia nomes pós-`OneHotEncoder`), `importancia_por_variavel_original()` (agrega de volta pra variável original), `efeito_medio_por_categoria()` |
| `check_ft_ml.py` | script de checagem ad-hoc da `FT_MACHINE_LEARNING` |

### `src/visualization/`

| Arquivo | Responsabilidade |
|---|---|
| `charts.py` | funções puras `df/Series → matplotlib.Figure` (`bar_importancia_shap`, `bar_metricas_por_grupo`, `bar_comparacao_modelos`, `bar_perfil_clusters`, `matriz_confusao`) — nunca lê arquivo nem treina modelo |
| `run_visualizations.py` | lê `reports/*.csv` e chama `charts.py`, salva `images/*.png` |

## 5. Especificação de dados

### 5.1 Camada Gold

Reconstruída localmente em pandas puro (ver README, "Reconstrução da
camada Gold"), replicando a lógica de negócio do projeto original em
PySpark/Databricks da Fase 2. Quatro tabelas:

| Tabela | Granularidade | Linhas (3 anos) | Papel no projeto |
|---|---|---|---|
| `FT_MACHINE_LEARNING` | aluno | ~6.090.791 | base do modelo de aluno |
| `FT_INDICADOR_MUNICIPIO` | município/ano | 19.700 | indicador com metas, tendência, classificação por nível |
| `FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO` | município/ano | 16.396 | resultado observado vs. meta oficial -- base do modelo municipal e do clustering |
| `ANALISE_NIVEIS_MUNICIPIO` | município/ano | 16.396 | distribuição por nível de proficiência, `INDICE_RISCO_ESTRUTURAL` |

Validadas linha a linha contra um export real da Fase 2 (ver README,
seção "Validação contra a base original da AWS") -- contagem de linhas
idêntica e >99% de correspondência exata nos indicadores.

### 5.2 `FT_MACHINE_LEARNING` — schema completo (50 colunas)

Toda coluna é categorizada em `src/modeling/features.py`, sem sobra e
sem sobreposição (verificado: 17+7+10+5+1+3+3+3+1 = 50). Categoria e
critério de exclusão/inclusão:

| Categoria | Colunas | Critério |
|---|---|---|
| `LEAKAGE_DIRETO` (17) | `VL_PROFICIENCIA_LP`, `IN_ALFABETIZADO`, `DS_ALFABETIZADO`, `IN_PRESENCA_LP`, `IN_PREENCHIMENTO_LP`, `VL_PESO_ALUNO_LP`, `FAIXA_PROFICIENCIA`, `IN_PARTICIPOU_AVALIACAO`, `DS_PARTICIPACAO`, `DS_SITUACAO_AVALIACAO`, `IN_PROVA_VALIDA`, `IN_POSSUI_PROFICIENCIA`, `DIF_MEDIA_MUNICIPIO`, `DIF_MEDIA_ESTADO`, `IN_ACIMA_MEDIA_MUNICIPIO`, `IN_ACIMA_MEDIA_ESTADO`, `DESEMPENHO_RELATIVO` | determinístico ou quase-determinístico em relação ao `TARGET` -- nunca entra no modelo |
| `LEAKAGE_PARCIAL` (7) | `PC_ALUNO_ALFABETIZADO`, `PC_ALUNO_ALFABETIZADO_ESTADO`, `VL_MEDIA_LP`, `VL_MEDIA_LP_ESTADO`, `FAIXA_ALFABETIZACAO`, `FAIXA_MEDIA_LP`, `DIF_ALFABETIZACAO_MUNICIPIO` | agregados que incluem o próprio aluno no cálculo -- fora do baseline por padrão |
| `IDENTIFICADORES_METADADOS` (10) | `SK_ALUNO`, `ID_ALUNO`, `ID_ESCOLA`, `DT_PROCESSAMENTO`, `TS_PROCESSAMENTO`, `ANO_CARGA`, `MES_CARGA`, `TP_SERIE`, `DS_SERIE`, `ANO_REFERENCIA` | sem poder preditivo generalizável (`ID_ALUNO`/`ID_ESCOLA` são máscaras re-sorteadas a cada ano pelo INEP) ou constantes na base atual |
| `REDUNDANTES` (5) | `CO_UF`, `NO_MUNICIPIO`, `NO_MUNICIPIO_UF`, `CO_MUNICIPIO_IBGE`, `DS_DEPENDENCIA` | mesma informação já coberta por outra coluna mantida |
| `CHAVE_MUNICIPIO` (1) | `CO_MUNICIPIO` | alta cardinalidade (~5.567 valores) -- usado como chave de junção, não como feature direta |
| `FEATURES_SOCIOECONOMICAS` (3) | `PC_FAMILIAS_POBREZA`, `RENDA_PER_CAPITA_MEDIA`, `MEDIA_INSE` | enriquecimento externo, opcional via `include_socioeconomico` |
| `FEATURES_INFRAESTRUTURA_ESCOLAR` (3) | `PC_ESCOLAS_BIBLIOTECA`, `PC_ESCOLAS_LAB_INFORMATICA`, `PC_ESCOLAS_INTERNET_ALUNOS` | Censo Escolar, opcional via `include_infraestrutura` |
| `FEATURES_SEGURAS` (3) | `REGIAO`, `SG_UF`, `TP_DEPENDENCIA` | base do modelo, sempre incluída |
| `TARGET_COL` (1) | `TARGET` (= `IN_ALFABETIZADO`) | variável-alvo |

`WEIGHT_COL = VL_PESO_ALUNO_LP` não é feature (está em `LEAKAGE_DIRETO`)
-- usado exclusivamente como `sample_weight` em `.fit()` e nas métricas,
via `features.select_weights()`.

### 5.3 Enriquecimento externo por município

Quatro fontes, unidas por `CO_MUNICIPIO` em
`_build_dim_municipio_socioeconomico()` (`src/preprocessing/gold.py`):
CadÚnico (pobreza), Censo Demográfico 2022 do IBGE (renda), INSE (nível
socioeconômico escolar), Censo Escolar (infraestrutura). Datas de
referência exatas, cuidados de join (conversão de código IBGE de 6 para
7 dígitos) e a limitação de "fotos únicas" repetidas entre anos estão
documentados no README, seção "Enriquecimento externo por município" --
não duplicado aqui.

## 6. Pipeline de modelagem — detalhamento técnico

### 6.1 Modelo de aluno

1. **Split** (`split.py::split_2025`): filtra `ANO_REFERENCIA == 2025`,
   `train_test_split(test_size=0.30, random_state=42, stratify=TARGET)`.
2. **Pré-processamento** (`pipeline.py::build_preprocessor`):
   `ColumnTransformer` com dois ramos -- numéricas (`FEATURES_SOCIOECONOMICAS`
   + opcionalmente `FEATURES_INFRAESTRUTURA_ESCOLAR`) via mediana+padronização,
   categóricas (`FEATURES_SEGURAS`) via moda+one-hot.
3. **Modelo**: `RandomForestClassifier`, treinado com
   `classifier__sample_weight=w_train` (peso amostral do INEP).
4. **Tuning** (`run_baseline_tuning.py`): `RandomizedSearchCV`, `n_iter=10`,
   `StratifiedKFold(3)`, `scoring="roc_auc"`, busca numa amostra
   estratificada de 300 mil linhas do treino (treino completo tem 1,56M --
   caro demais pra grid exaustivo). Espaço de busca:
   ```python
   PARAM_DIST = {
       "classifier__n_estimators": randint(100, 400),
       "classifier__max_depth": [10, 15, 20, 30, None],
       "classifier__min_samples_leaf": randint(10, 200),
   }
   ```
   Hiperparâmetros vencedores re-treinados no treino **completo** (não só
   a amostra da busca) antes da avaliação final no teste.
5. **Avaliação**: `evaluate.evaluate_model()` (com `sample_weight`) +
   `evaluate.evaluate_by_group()` por `REGIAO` e `SG_UF` +
   `evaluate.confusion_matrix_df()`.

### 6.2 Modelo municipal (metas)

1. **Dataset** (`municipal_metas.py::build_dataset`): junta, por
   `(CO_MUNICIPIO, ANO_REFERENCIA)`, o resultado do ano N (`TARGET =
   IN_META_ATINGIDA`) com os indicadores do ano N-1 (evita vazar o
   resultado que está sendo previsto) e o enriquecimento estático por
   município.
2. **Split temporal genuíno** (`split_by_year`): treino = prevendo 2024
   (indicadores de 2023), teste = prevendo 2025 (indicadores de 2024).
   Válido porque `CO_MUNICIPIO` (código IBGE) é estável entre anos --
   diferente de `ID_ALUNO`/`ID_ESCOLA`.
3. **Pré-processamento**: mesmo padrão do modelo de aluno, mas sobre
   `NUMERIC_COLS`/`CATEGORICAL_COLS` de `municipal_metas.py` (inclui
   `PC_ALUNO_ALFABETIZADO_ANTERIOR`, `DIF_META_ALFABETIZACAO_ANTERIOR`,
   `IN_META_ATINGIDA_ANTERIOR`).
4. **Tuning** (`run_municipal_metas_tuning.py`): `GridSearchCV` completo
   (treino tem só ~5.400 linhas, cabe fácil), `StratifiedKFold(5,
   shuffle=True)`, `scoring="roc_auc"`. Espaço de busca:
   ```python
   PARAM_GRID = {
       "classifier__n_estimators": [100, 200, 300],
       "classifier__max_depth": [4, 6, 8, None],
       "classifier__min_samples_leaf": [5, 20, 50],
   }
   ```
   36 combinações x 5 folds. Achado documentado no README: AUC de CV
   (0,772) bem acima do AUC no teste real (0,663) -- gap grande entre
   validação dentro do mesmo ano e generalização pra um ano seguinte.
5. **Calibração de limiar** (`evaluate.calibrar_limiar_decisao`): via
   `sklearn.metrics.precision_recall_curve` sobre `predict_proba`,
   calcula o limiar de F1 máximo e o limiar de maior precisão que ainda
   garante `recall >= 0.80`, ambos pra classe 0 (não atingiu meta) --
   ver README, "Escolha do algoritmo", pra números e discussão do
   trade-off.
6. **XGBoost testado e descartado**: mesma metodologia de busca aplicada
   a `XGBClassifier` -- AUC de teste pior que o RandomForest (0,613 vs.
   0,663). Não faz parte do código de produção; testado num script
   descartável fora do repositório, não versionado.

### 6.3 Clustering municipal

`municipal_clustering.py::build_perfil_municipios` monta o perfil 2025
de cada município (`FEATURES_PERFIL`: alfabetização, distância da meta
2030, 3 socioeconômicas, 3 de infraestrutura). `run_municipal_clustering.py`
padroniza (`StandardScaler`) e roda `KMeans` pra `k` de 3 a 8,
escolhendo o `k` de maior `silhouette_score` (resultado: `k=3`).
Cruzamento posterior com `REGIAO` via `pd.crosstab` normalizado.

## 7. Interpretabilidade (SHAP) — implementação técnica

`src/evaluation/explain.py` usa `shap.TreeExplainer` (específico pra
modelos em árvore, muito mais rápido que o `KernelExplainer` genérico)
sobre uma amostra do teste (não a base inteira -- custo computacional).
Como o pré-processamento inclui `OneHotEncoder`, cada categoria vira uma
coluna própria (`SG_UF=CE`, `SG_UF=RN`, etc.) antes de chegar no
`TreeExplainer`; `get_expanded_feature_names()` reconstrói esses nomes na
ordem exata em que saem do `ColumnTransformer`, e
`importancia_por_variavel_original()` soma de volta pra variável
original quando a leitura por variável agregada é o que interessa (ex.:
"quanto `SG_UF` pesa no total", não "quanto `SG_UF=CE` pesa sozinho" --
distinção discutida no README, seção "Interpretação dos resultados").

Detalhe técnico que já causou um bug real neste projeto: a saída do
`ColumnTransformer` pode vir como matriz esparsa (`scipy.sparse`), que o
`TreeExplainer` não trata corretamente (quebra em `np.isnan` interno) --
`compute_shap_values()` densifica (`.toarray()`) antes de passar pro
explainer, viável porque a amostra é pequena.

## 8. Ordem de execução (grafo de dependências)

```
1. python -m src.preprocessing.run_pipeline
   → gera data/gold/*.parquet (as 4 tabelas da Seção 5.1)

2a. python -m src.modeling.run_baseline_tuning
    → reports/baseline_tuning_cv_results.csv, baseline_metricas_por_uf_tuned.csv
    (informa os hiperparâmetros hardcoded em run_baseline.py -- não há
    leitura automática do CSV de volta pro código, é decisão humana)

2b. python -m src.modeling.run_baseline
    → reports/baseline_comparison.csv, baseline_confusion_matrix.csv,
      baseline_metricas_por_{regiao,uf,uf_com_infraestrutura}.csv

2c. python -m src.modeling.run_shap
    → reports/shap_importancia.csv, shap_efeito_regiao.csv

3a. python -m src.modeling.run_municipal_metas_tuning
    → reports/municipal_metas_tuning_cv_results.csv
    (mesma relação humana-no-loop que 2a/2b)

3b. python -m src.modeling.run_municipal_metas
    → reports/municipal_metas_comparison.csv, municipal_metas_confusion_matrix.csv,
      municipal_metas_calibracao_limiar.csv, municipal_metas_shap_importancia.csv

3c. python -m src.modeling.run_municipal_risco
    → reports/municipal_ranking_risco.csv
    (usa os limiares calibrados em 3b, hardcoded como constantes --
    mesma relação humana-no-loop)

3d. python -m src.modeling.run_municipal_clustering
    → reports/municipal_clusters_{perfil,municipios,x_regiao}.csv

4. python -m src.visualization.run_visualizations
   → images/*.png (lê todos os reports/*.csv gerados acima -- precisa
     rodar por último)
```

Passos `2*` e `3*` são independentes entre si (podem rodar em qualquer
ordem ou em paralelo); o passo `1` é pré-requisito de tudo, e o passo `4`
depende de todos os `reports/*.csv` já existirem.

## 9. Reprodutibilidade

- `random_state=42` fixo em todo componente estocástico.
- `sample_weight` (peso amostral oficial do INEP) aplicado de forma
  consistente em treino e métricas do modelo de aluno -- nunca só num
  dos dois lados.
- Validação cruzada sempre com `StratifiedKFold` (preserva a proporção
  de classes em cada fold) e `shuffle=True` com seed fixa.
- Avaliação final sempre num conjunto nunca visto durante a busca de
  hiperparâmetros (amostra de tuning ≠ teste final, no caso do aluno;
  ano de teste nunca entra na CV, no caso do município).
- Camada Gold validada linha a linha contra um export real da base
  original da Fase 2 (AWS/Databricks) -- ver README.

## 10. Referências técnicas

Fontes de dados oficiais (URLs de download, campos de referência
temporal verificados internamente em cada arquivo) e notas técnicas do
INEP/Todos Pela Educação citadas na validação do peso amostral estão
documentadas no README, seções "Reconstrução da camada Gold" e
"Enriquecimento externo por município" -- não duplicadas aqui para
evitar desatualização em duas fontes simultâneas.
