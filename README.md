# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

## Contexto do problema

Alfabetizar toda criança até o final do 2º ano do Ensino Fundamental é meta
do Compromisso Nacional Criança Alfabetizada, mas o resultado varia muito
pelo território: nos dados usados aqui, a diferença entre a melhor e a pior
região chega a ~15 pontos percentuais, e o achado do SHAP (ver "Interpretação
dos resultados") mostra que essa desigualdade está concentrada em poucos
estados específicos, não distribuída de forma difusa entre regiões.

Hoje o INEP só identifica quem não se alfabetizou **depois** da avaliação
anual — quando já é tarde para intervir naquele ano letivo. Um modelo que
aponte, com base em variáveis territoriais e socioeconômicas conhecidas
*antes* da prova, quais municípios/perfis têm maior risco de baixa
alfabetização permite que gestores públicos direcionem reforço escolar,
merenda, transporte ou material didático de forma proativa — não reativa.

## Objetivo analítico
Desenvolver um modelo supervisionado que preveja se um aluno será considerado alfabetizado ou não alfabetizado, com base em variáveis educacionais, territoriais e socioeconômicas.

## Descrição da base utilizada

Base a nível de aluno (`FT_MACHINE_LEARNING`, ~6,09 milhões de linhas,
2023-2025), reconstruída localmente a partir dos microdados públicos do
INEP (ver "Reconstrução da camada Gold" abaixo), enriquecida com 3 fontes
externas por município (ver "Enriquecimento externo por município"
abaixo): pobreza (CadÚnico), renda per capita (Censo 2022) e nível
socioeconômico escolar (INSE 2023). Não obtivemos acesso a tempo ao Atlas
do Desenvolvimento Humano (IDHM) — essas 3 fontes o substituem no projeto.

## Etapas de modelagem
- [x] Análise exploratória — `notebooks/01_EDA_Alfabetizacao.ipynb`
- [x] Tratamento de valores faltantes e leakage — `src/modeling/features.py`
- [x] Feature engineering e encoding — `src/modeling/pipeline.py`
- [x] Pipeline sklearn (pré-processamento + modelo) — `src/modeling/pipeline.py`
- [x] Treinamento, validação e otimização — `src/modeling/run_baseline.py` (split temporal em `src/modeling/split.py`)

## Escolha do algoritmo

Comparamos `LogisticRegression` e `RandomForestClassifier`, com e sem o
enriquecimento socioeconômico, no split temporal (treino 2023-2024, teste
2025 — ver `src/modeling/split.py`). O **RandomForest com enriquecimento**
venceu em todas as métricas (ver tabela abaixo) — é o único cenário que
supera a acurácia de "sempre prever a classe majoritária" no teste.
Escolhido por ser o melhor resultado e por permitir a interpretação via
SHAP (Seção "Interpretação dos resultados").

## Métricas de avaliação

Split temporal: treino em 2023-2024 (3.867.999 linhas, 51,3% alfabetizados),
teste em 2025 (2.222.792 linhas, 58,6% alfabetizados — note a mudança na
taxa real entre os recortes, ver "Limitações do projeto").

| Modelo | Enriquecimento | Acurácia | Precisão | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression | não | 0,5767 | 0,6286 | 0,6792 | 0,6529 | 0,5947 |
| LogisticRegression | sim | 0,5858 | 0,6286 | 0,7170 | 0,6699 | 0,6000 |
| RandomForest | não | 0,5819 | 0,6345 | 0,6765 | 0,6548 | 0,5954 |
| **RandomForest** | **sim** | **0,5992** | **0,6547** | 0,6695 | 0,6620 | **0,6184** |

*(baseline de "sempre prever a classe majoritária" no teste = 0,5862 de
acurácia — ver `reports/baseline_comparison.csv`)*

**Quebra por região** (melhor modelo — RandomForest + enriquecimento,
`reports/baseline_metricas_por_regiao.csv`):

| Região | Taxa real de alfabetização | Acurácia | Precisão | Recall |
|---|---|---|---|---|
| Norte | 52,2% | 0,5117 | 0,6931 | **0,1158** |
| Sudeste | 57,2% | 0,6007 | 0,6115 | 0,8288 |
| Sul | 58,6% | 0,6199 | 0,6851 | 0,6508 |
| Nordeste | 60,7% | 0,6087 | 0,7030 | 0,6159 |
| Centro-Oeste | 66,7% | 0,6382 | 0,7127 | 0,7667 |

O recall varia de 11,6% a 82,9% — uma disparidade desproporcional à
diferença real na taxa de alfabetização (~15pp). Ver "Interpretação dos
resultados" e "Limitações do projeto".

## Interpretação dos resultados

SHAP (`TreeExplainer`, amostra de 20 mil linhas do teste — ver
`src/modeling/explain.py` e `reports/shap_importancia.csv`) no RandomForest
com enriquecimento:

| Variável | Importância média (\|SHAP\|) |
|---|---|
| `SG_UF` | 0,085 |
| `REGIAO` | 0,019 |
| `PC_FAMILIAS_POBREZA` | 0,016 |
| `MEDIA_INSE` | 0,010 |
| `RENDA_PER_CAPITA_MEDIA` | 0,006 |
| `ANO_REFERENCIA` | 0,005 |
| `TP_DEPENDENCIA` | 0,005 |

- **`SG_UF` domina a decisão** (4,5x mais que `REGIAO` agregada), puxado
  principalmente por `SG_UF=CE` e `SG_UF=BA` — o modelo capturou um padrão
  específico de estado, não um padrão regional difuso. Faz sentido dado que
  o Ceará tem um programa estadual de alfabetização amplamente reconhecido
  nacionalmente.
- **A disparidade regional tem causa identificada**: o efeito médio (com
  sinal) de `REGIAO=Norte` é -0,037 — o mais negativo entre as 5 regiões
  por larga margem (a segunda pior, Sul, fica em -0,005; ver
  `reports/shap_efeito_regiao.csv`). O modelo aprendeu a associar "Norte"
  fortemente a "não alfabetizado", o que explica o recall de 11,6% na
  região.
- **O enriquecimento socioeconômico se justificou**: `PC_FAMILIAS_POBREZA`
  sozinha (0,016) vale mais que toda a `REGIAO` agregada (0,019 somando as
  5 categorias) e fica atrás só de `SG_UF` — confirma manter essas 3
  features mesmo com correlação linear fraca a nível de aluno (Seção 8 da
  EDA), decisão validada pelo SHAP em vez da correlação isolada.

## Insights encontrados

- **Target balanceado**: 52,2% alfabetizados vs. 47,8% não alfabetizados — não requer balanceamento artificial (SMOTE/undersampling).
- **Desigualdade regional**: ~15 pontos percentuais de diferença entre a melhor região (Centro-Oeste, 57,5%) e a pior (Norte, 42,2%).
- **Rede**: base majoritariamente municipal (87%); amostra da rede privada é irrisória (25 alunos) e não deve ser usada para conclusões.
- **Data leakage crítico identificado na EDA**: `TARGET` é 100% determinístico a partir de `VL_PROFICIENCIA_LP >= 743` (sem exceções) e de flags de participação na prova. Essas variáveis (e derivadas) foram excluídas do conjunto de features de modelagem — ver `notebooks/01_EDA_Alfabetizacao.ipynb`, seção 7.
- Agregados municipais/estaduais (`PC_ALUNO_ALFABETIZADO*`, `VL_MEDIA_LP*`) têm vazamento parcial (incluem o próprio aluno no cálculo) e devem ser usados com cautela ou recalculados como *leave-one-out*.
- **`DESEMPENHO_RELATIVO` também é leakage** (derivado de `DIF_MEDIA_ESTADO`, que vem de `VL_PROFICIENCIA_LP`) e não estava na lista original da EDA — encontrado ao formalizar a seleção de features em código (`src/modeling/features.py`).
- **`ID_ALUNO` não é um identificador persistente entre anos**: a faixa numérica se repete quase idêntica em 2023/2024/2025 (todas começam em 11.000.001) e, dos "mesmos" IDs que aparecem em anos diferentes, 0% correspondem à mesma escola — é um ID gerado por ano, não um registro nacional do aluno. Não há vazamento de aluno entre treino e teste a evitar.
- **Sem o enriquecimento externo, o modelo mal supera prever a classe majoritária** (`REGIAO`/`SG_UF`/`TP_DEPENDENCIA`/`ANO_REFERENCIA` sozinhos: acurácia 0,577-0,582 vs. baseline de 0,586) — quase todo o sinal individual forte foi removido como leakage, então o que resta é fraco por natureza.
- **O modelo reproduz e amplia a desigualdade regional**: o recall varia de 11,6% (Norte) a 82,9% (Sudeste), desproporcional à diferença real na taxa de alfabetização (~15pp) — ver "Interpretação dos resultados".

## Limitações do projeto

- O indicador do INEP trata "aluno não avaliado" como "não alfabetizado" por definição — mistura duas populações conceitualmente diferentes (quem não aprendeu vs. quem não fez a prova).
- Amostra da rede privada é pequena demais para generalizar.
- Reconstrução da camada Gold feita localmente (fora do Databricks/AWS original da Fase 2) — ver seção "Reconstrução da camada Gold" abaixo para detalhes e possíveis pequenas diferenças de metodologia.
- Renda (Censo 2022) e INSE (SAEB 2023) são fotos únicas, repetidas nos 3 anos do painel (2023-2025) — ver "Enriquecimento externo por município" abaixo.
- **Disparidade regional no modelo (achado central da modelagem)**: o RandomForest escolhido tem recall de apenas 11,6% no Norte contra 82,9% no Sudeste — muito além do que a diferença real na taxa de alfabetização (~15pp) justificaria. O SHAP aponta que o modelo aprendeu um padrão específico por `SG_UF` (principalmente Ceará e Bahia) e penaliza fortemente `REGIAO=Norte`. **Este modelo não deve ser usado para decisões de política pública sem antes mitigar essa disparidade** (ver "Aplicação prática" e "Evoluções futuras").
- O split treino/teste é temporal (2023-2024 → 2025), e a taxa real de alfabetização mudou entre os recortes (51,3% no treino vs. 58,6% no teste) — o modelo é avaliado sob um *dataset shift* real, não uma amostra idêntica reembaralhada.

## Aplicação prática para políticas públicas

Na forma atual, o modelo serve melhor como **ferramenta exploratória** —
ex.: cruzar `PC_FAMILIAS_POBREZA`/`MEDIA_INSE`/`RENDA_PER_CAPITA_MEDIA` por
município pra priorizar visitas técnicas ou repasse de material — do que
como critério automático de decisão. A disparidade regional encontrada
(recall de 11,6% no Norte) significa que, se usado para sinalizar "alunos
em risco" hoje, o modelo **erraria mais justamente nos alunos do Norte** —
a região que mais precisa de atenção segundo os próprios dados. Qualquer
uso em política pública exige primeiro mitigar essa disparidade (ver
"Evoluções futuras") e, enquanto isso não acontece, avaliar as previsões
sempre segmentadas por região/UF, nunca por uma métrica agregada nacional.

## Possíveis evoluções futuras

- **Mitigar a disparidade regional** antes de qualquer uso prático: testar
  recalibração por região, remoção/ponderação de `SG_UF` como feature, ou
  métricas de otimização sensíveis a fairness (ex.: equalized odds).
- Revisitar a EDA (`notebooks/01_EDA_Alfabetizacao.ipynb`) trazendo a
  desigualdade regional/por UF **excluindo a rede privada** (amostra
  irrisória, ~25 alunos) para um retrato mais limpo da desigualdade real
  entre redes municipal e estadual.
- Obter acesso ao Atlas do Desenvolvimento Humano (IDHM) — não conseguimos
  a tempo (Atlas Brasil indisponível) e usamos CadÚnico/Censo/INSE como
  substituto (ver "Enriquecimento externo por município").
- Testar `ID_ESCOLA` via encoding específico (ex.: target encoding) — ficou
  fora do baseline por alta cardinalidade (~43,6 mil escolas), mas pode
  capturar efeito de escola além do efeito de estado.
- Tuning de hiperparâmetros do RandomForest e teste de outros algoritmos em
  árvore (XGBoost, LightGBM) agora que o pipeline (`src/modeling/pipeline.py`)
  já injeta qualquer classificador sklearn sem mudança de código.

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

## Reconstrução da camada Gold (sem depender da AWS da Fase 2)

Como o pipeline original (Fase 2) roda em Databricks + PySpark + Delta Lake
sobre um bucket S3 privado (`tech-challenge-etl-153372322872-us-east-2-an`,
repo: https://github.com/Macedim/tech-challenge-ETL), e nem todo integrante
do grupo tem acesso a essa conta AWS, a camada Gold foi **reconstruída
localmente em pandas puro**, replicando a mesma lógica de negócio dos
notebooks originais (bronze → silver → gold), a partir dos microdados
públicos do INEP.

**Escopo temporal: 2023, 2024 e 2025** (mesmo escopo da base original na AWS — confirmado por comparação direta com um export da camada Gold da Fase 2, ver seção "Validação" abaixo).

**Fonte dos dados (INEP, download direto):**
- Microdados 2023: https://download.inep.gov.br/dados_abertos/microdados_avaliacao_da_alfabetizacao_2023.zip
- Microdados 2024: https://download.inep.gov.br/dados_abertos/microdados_avaliacao_da_alfabetizacao_2024.zip
- Microdados 2025: https://download.inep.gov.br/dados_abertos/microdados_AEEB_2025.zip
- Metas/resultados Brasil e UF (histórico 2023-2025 em um único arquivo): https://download.inep.gov.br/avaliacao_da_alfabetizacao/resultados/resultados_e_metas_ufs_2025_v1.xlsx
- Metas/resultados Município (histórico 2023-2025 em um único arquivo): https://download.inep.gov.br/avaliacao_da_alfabetizacao/resultados/resultados_e_metas_municipios_2025_v2.xlsx

**Como rodar a reconstrução:**

1. Baixe os 5 arquivos acima e organize em `data/raw/`:
   ```
   data/raw/DADOS/2023/TS_ALUNO.csv        (dentro do zip de microdados 2023)
   data/raw/DADOS/2023/TS_MUNICIPIO.csv
   data/raw/DADOS/2023/TS_ESTADO.csv
   data/raw/DADOS/2024/TS_ALUNO.csv        (dentro do zip de microdados 2024)
   data/raw/DADOS/2024/TS_MUNICIPIO.csv
   data/raw/DADOS/2024/TS_ESTADO.csv
   data/raw/DADOS/2025/TS_ALUNO.csv        (dentro do zip de microdados 2025)
   data/raw/DADOS/2025/TS_MUNICIPIO.csv
   data/raw/DADOS/2025/TS_ESTADO.csv
   data/raw/resultados_e_metas_ufs_2025_v1.xlsx
   data/raw/resultados_e_metas_municipios_2025_v2.xlsx
   ```
2. Instale as dependências (`pip install -r requirements.txt` — inclui `pyarrow`, necessário para salvar as tabelas em Parquet; sem ele a pipeline usa CSV como fallback, bem mais lento/pesado em memória para a `FT_MACHINE_LEARNING`, que tem ~6 milhões de linhas nos 3 anos).
3. Rode a pipeline completa:
   ```bash
   python -m src.preprocessing.run_pipeline
   ```
   A etapa de `TS_ALUNO` (bronze + silver + `FT_MACHINE_LEARNING`) processa ~6 milhões de linhas — pode levar alguns minutos e exige uma máquina com pelo menos 4-8 GB de RAM livres se estiver sem `pyarrow`.

### Validação contra a base original da AWS

Um integrante do grupo exportou parte da camada Gold real (S3 da Fase 2) e comparamos linha a linha com esta reconstrução, nos 3 anos:

- `FT_INDICADOR_MUNICIPIO`, `FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO` e `ANALISE_NIVEIS_MUNICIPIO`: contagem de linhas idêntica (19.700 / 16.396 / 16.396) e mais de 99% de correspondência exata nos indicadores numéricos e categóricos, nos 3 anos.
- **Bug real encontrado na `FT_MACHINE_LEARNING` original**: o notebook `gd_machine_learning.ipynb` do repositório da Fase 2 tem um filtro `.filter(col('CO_MUNICIPIO').isNull())` que mantém só os poucos alunos com dados cadastrais incompletos (628 registros de 2025, todos preliminares), em vez da base completa de alunos (~6 milhões nos 3 anos). Vale reportar ao grupo — a tabela publicada na AWS hoje não é utilizável para treinar modelo.
- Duas pequenas correções feitas durante a validação: (1) o valor da meta nacional 2024 usado por nós vem exato do INEP (59,9%) contra um arredondamento de 60% usado na base original — decisão consciente de manter o valor exato; (2) `DISTANCIA_META_2030` foi corrigida para seguir a mesma convenção (resultado − meta) das demais colunas `DIF_META_*`, e passou a assumir 80% como meta padrão quando a planilha do INEP não traz um valor explícito para 2030 (municípios que já superaram a meta).

Isso gera, em `data/bronze/`, `data/silver/` e `data/gold/`, as mesmas 4
tabelas Gold do projeto original:

- **`FT_MACHINE_LEARNING`** — base a nível de aluno (~2,1 milhões de linhas), com `TARGET = IN_ALFABETIZADO` e features de posição relativa ao município/estado. **Esta é a tabela usada para treinar o modelo da Fase 3.**
- `FT_INDICADOR_MUNICIPIO` — indicador por município com metas, tendência (`Melhorou`/`Piorou`/`Estável`) e classificação por nível.
- `FT_INDICADOR_MUNICIPIO_META_VS_RESULTADO` — resultado observado vs. meta oficial por município.
- `ANALISE_NIVEIS_MUNICIPIO` — perfil de distribuição dos alunos entre níveis de proficiência, com índices de polarização e risco estrutural.

Uma cópia pronta das 3 tabelas menores (agregadas por município/UF) já vem
incluída em `data/gold_sample/` para consulta rápida sem precisar rodar a
pipeline. A `FT_MACHINE_LEARNING` não foi incluída por ser grande demais
(>700 MB em CSV) — gere-a localmente com o comando acima.

Os dados brutos e as camadas intermediárias (bronze/silver/gold) **não são
versionados no Git** (ver `.gitignore`) — apenas o código que as gera.

### Enriquecimento externo por município (pobreza, renda, infraestrutura)

Testamos a hipótese de que municípios mais pobres e com pior infraestrutura
educacional têm menor taxa de alfabetização (ver `notebooks/01_EDA_Alfabetizacao.ipynb`,
Seção 8). Não conseguimos acesso ao Atlas do Desenvolvimento Humano (IDHM)
a tempo (ferramenta do Atlas Brasil indisponível) — as 3 fontes abaixo o
substituem para este projeto:

- **Pobreza** — CadÚnico (VIS Data 3), % de famílias na faixa de pobreza do
  PBF por município (08/2026): `data/gold_sample/cadastro_unico_pobreza/CADUNICO_FAMILIAS_POBREZA_MUNICIPIO.csv`
  (já incluído no repositório).
- **Renda** — Censo 2022 (SIDRA, tabela 10295), renda per capita média por
  município, divulgado em out/2025. Baixe e salve em
  `data/raw/censo_renda/censo2022_renda_per_capita_municipio.csv`.
- **Infraestrutura socioeconômica escolar** — INSE 2023 (INEP/SAEB), por
  escola. Baixe e salve em `data/raw/INSE/INSE_2023_escolas.xlsx`.

**Atenção ao join:** o CadÚnico usa o código IBGE **sem dígito verificador**
(6 dígitos, ex.: `120001` para Acrelândia), enquanto `CO_MUNICIPIO` no
restante do projeto usa o código completo (7 dígitos, ex.: `1200013`). A
pipeline (`_build_dim_municipio_socioeconomico` em `src/preprocessing/gold.py`)
já faz essa conversão (`CO_MUNICIPIO // 10`) antes do merge.

**Resultado:** as 3 variáveis (`PC_FAMILIAS_POBREZA`, `RENDA_PER_CAPITA_MEDIA`,
`MEDIA_INSE`) correlacionam na direção esperada com a taxa de alfabetização
a nível de **município** (|r| entre 0,18 e 0,29), mas a correlação cai bastante
a nível de **aluno** — a granularidade real de treino da `FT_MACHINE_LEARNING`
(|r| entre 0,006 e 0,073, praticamente ruído para renda). Isso é esperado
(correlação agregada por município é sempre inflada em relação à correlação
por indivíduo) e não significa que as features sejam inúteis num modelo
multivariado — a decisão de manter ou descartar cada uma fica para a etapa
de modelagem (importância de feature / SHAP), não para a correlação isolada.

**Limitação:** renda (Censo 2022) e INSE (SAEB 2023) são fotos únicas — o
mesmo valor se repete para um dado município nos 3 anos do painel
(2023-2025), diferente do CadÚnico que já reflete a referência mais recente.
