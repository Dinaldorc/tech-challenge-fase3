# Tech Challenge – Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

## Contexto do problema
_(descrever o problema da alfabetização infantil no Brasil e por que antecipar risco importa para gestores públicos)_

## Objetivo analítico
Desenvolver um modelo supervisionado que preveja se um aluno será considerado alfabetizado ou não alfabetizado, com base em variáveis educacionais, territoriais e socioeconômicas.

## Descrição da base utilizada
_(fontes: camada Gold da Fase 2 + enriquecimentos externos — IBGE, Censo Escolar, FUNDEB, PNAD, Atlas do Desenvolvimento Humano, Cadastro Único)_

## Etapas de modelagem
- Análise exploratória
- Tratamento de valores faltantes e leakage
- Feature engineering e encoding
- Pipeline sklearn (pré-processamento + modelo)
- Treinamento, validação e otimização

## Escolha do algoritmo
_(preencher após experimentação)_

## Métricas de avaliação
_(preencher)_

## Interpretação dos resultados
_(Feature Importance / SHAP)_

## Insights encontrados

- **Target balanceado**: 52,2% alfabetizados vs. 47,8% não alfabetizados — não requer balanceamento artificial (SMOTE/undersampling).
- **Desigualdade regional**: ~15 pontos percentuais de diferença entre a melhor região (Centro-Oeste, 57,5%) e a pior (Norte, 42,2%).
- **Rede**: base majoritariamente municipal (87%); amostra da rede privada é irrisória (25 alunos) e não deve ser usada para conclusões.
- **Data leakage crítico identificado na EDA**: `TARGET` é 100% determinístico a partir de `VL_PROFICIENCIA_LP >= 743` (sem exceções) e de flags de participação na prova. Essas variáveis (e derivadas) foram excluídas do conjunto de features de modelagem — ver `notebooks/01_EDA_Alfabetizacao.ipynb`, seção 7.
- Agregados municipais/estaduais (`PC_ALUNO_ALFABETIZADO*`, `VL_MEDIA_LP*`) têm vazamento parcial (incluem o próprio aluno no cálculo) e devem ser usados com cautela ou recalculados como *leave-one-out*.

## Limitações do projeto

- O indicador do INEP trata "aluno não avaliado" como "não alfabetizado" por definição — mistura duas populações conceitualmente diferentes (quem não aprendeu vs. quem não fez a prova).
- Amostra da rede privada é pequena demais para generalizar.
- Reconstrução da camada Gold feita localmente (fora do Databricks/AWS original da Fase 2) — ver seção "Reconstrução da camada Gold" abaixo para detalhes e possíveis pequenas diferenças de metodologia.

## Aplicação prática para políticas públicas
_(preencher)_

## Possíveis evoluções futuras
_(preencher)_

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
