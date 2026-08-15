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

**Fonte dos dados (INEP, download direto):**
- Microdados: https://download.inep.gov.br/dados_abertos/microdados_avaliacao_da_alfabetizacao_2024.zip
- Metas Brasil/UF: https://download.inep.gov.br/alfabetiza_brasil/resultados_e_metas_ufs_2024_2.xlsx
- Metas Município: https://download.inep.gov.br/alfabetiza_brasil/resultados_e_metas_municipios_2024.xlsx

**Como rodar a reconstrução:**

1. Baixe os 3 arquivos acima e organize em `data/raw/`:
   ```
   data/raw/DADOS/TS_ALUNO.csv        (dentro do zip de microdados)
   data/raw/DADOS/TS_MUNICIPIO.csv    (dentro do zip de microdados)
   data/raw/DADOS/TS_ESTADO.csv       (dentro do zip de microdados)
   data/raw/resultados_e_metas_ufs_2024_2.xlsx
   data/raw/resultados_e_metas_municipios_2024.xlsx
   ```
2. Instale as dependências (`pip install -r requirements.txt` — inclui `pyarrow`, necessário para salvar as tabelas em Parquet).
3. Rode a pipeline completa:
   ```bash
   python -m src.preprocessing.run_pipeline
   ```

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
