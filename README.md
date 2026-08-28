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

## Estratégia do projeto: perguntas de negócio

O Tech Challenge pede que o projeto responda perguntas de negócio, não só
produza métricas técnicas altas -- o foco é gerar inteligência aplicável
ao contexto educacional brasileiro. Isso levou a reposicionar o projeto:
o modelo de **aluno** (`FT_MACHINE_LEARNING`, seções acima) virou um
resultado secundário documentado -- achado legítimo em si: não há sinal
individual suficiente pra prever por aluno com os dados públicos
disponíveis (ver "Limitações do projeto"). O entregável principal passou
a ser um conjunto de análises a nível de **município**
(`src/modeling/municipal_metas.py`, `municipal_clustering.py`,
`run_municipal_*.py`), o grão onde as features realmente variam de
município pra município e onde a EDA já mostrava sinal real (Seção 8).

**1. Quais fatores mais impactam a alfabetização?** SHAP nos dois modelos
(aluno e município) aponta o mesmo padrão: `SG_UF` domina, à frente de
`REGIAO` -- não é uma diferença regional difusa, é específica de estado.
No modelo municipal (mais interpretável pra essa pergunta), depois do
estado o que mais pesa é o **desempenho do ano anterior**
(`DIF_META_ALFABETIZACAO_ANTERIOR`, `PC_ALUNO_ALFABETIZADO_ANTERIOR`) e
`PC_FAMILIAS_POBREZA` -- ver `reports/municipal_metas_shap_importancia.csv`.

**2. Quais municípios apresentam maior risco educacional?**
`src/modeling/run_municipal_risco.py` pontua cada um dos 5.500 municípios
com a probabilidade de não bater a meta de alfabetização no próximo
ciclo (RandomForest treinado com todo o histórico 2023-2025), usando os
indicadores mais recentes (2025). Ranking completo em
`reports/municipal_ranking_risco.csv`. Achado: os municípios de maior
risco pontuado são majoritariamente do **Rio Grande do Sul** -- não por
desempenho absoluto ruim (52-73% de alfabetização), mas por ter metas
muito exigentes em relação à própria trajetória histórica (só 23,7% dos
anos com meta batida, 2ª pior taxa do país). "Risco de não bater a meta"
e "baixo desempenho absoluto" são coisas diferentes. A pontuação foi
validada contra o `INDICE_RISCO_ESTRUTURAL` já existente (calculado de
forma independente, só pela distribuição de níveis de proficiência):
correlação de 0,341, coerente com duas métricas relacionadas mas distintas.

**3. Quais regiões possuem padrões semelhantes?**
`src/modeling/run_municipal_clustering.py` agrupa os municípios (K-Means,
k=3 escolhido por silhouette score) por perfil socioeconômico +
educacional, sem usar região/UF como feature. Resultado: os agrupamentos
naturais **não coincidem com as 5 regiões oficiais do IBGE**. Sul +
Sudeste + Centro-Oeste formam um cluster único (perfil de renda/
infraestrutura alta). O Nordeste se divide em dois clusters opostos com
pobreza e renda quase idênticas (~34% pobreza, ~R$800-900 de renda) mas
30 pontos percentuais de diferença em alfabetização -- um puxado por
Piauí/Ceará/Paraíba/Maranhão (82,6% de alfabetização, meta 2030 já
batida), outro por Bahia/Rio Grande do Norte (52,7%, 27pp longe da meta).
Ver `reports/municipal_clusters_perfil.csv` e
`reports/municipal_clusters_x_regiao.csv`.

**4. Como prever municípios que podem não atingir metas futuras?**
`src/modeling/municipal_metas.py` + `run_municipal_metas.py`:
RandomForest treinado em 2024 (usando indicadores de 2023), testado em
2025 (usando indicadores de 2024) -- split temporal genuíno, válido
porque `CO_MUNICIPIO` é o código IBGE oficial (estável entre anos, ao
contrário de `ID_ALUNO`/`ID_ESCOLA` -- ver "Limitações do projeto").
AUC = 0,660. Na classe que importa pra política pública (município que
**não** vai bater a meta, 28,7% da base de teste): precisão 40% (quase o
dobro da taxa-base de 28,7%) e recall 51% -- modesto, mas real e
diretamente aplicável a priorização.

**5. Quais variáveis possuem maior influência nos modelos?** Mesma
resposta da pergunta 1 -- o SHAP dos dois modelos é a ferramenta usada;
ver `reports/shap_importancia.csv` (aluno) e
`reports/municipal_metas_shap_importancia.csv` (município).

## Descrição da base utilizada

Base a nível de aluno (`FT_MACHINE_LEARNING`, ~6,09 milhões de linhas,
2023-2025), reconstruída localmente a partir dos microdados públicos do
INEP (ver "Reconstrução da camada Gold" abaixo), enriquecida com 3 fontes
externas por município (ver "Enriquecimento externo por município"
abaixo): pobreza (CadÚnico), renda per capita (Censo 2022) e nível
socioeconômico escolar (INSE 2023). Não obtivemos acesso a tempo ao Atlas
do Desenvolvimento Humano (IDHM) — essas 3 fontes o substituem no projeto.

**A etapa de modelagem (treino/teste) usa só o ano de 2025** (2.222.792
linhas) — ver "Limitações do projeto" para o motivo de restringir a
apenas um ano. A EDA (`notebooks/01_EDA_Alfabetizacao.ipynb`) continua
cobrindo os 3 anos.

## Etapas de modelagem
- [x] Análise exploratória — `notebooks/01_EDA_Alfabetizacao.ipynb`
- [x] Tratamento de valores faltantes e leakage — `src/modeling/features.py`
- [x] Feature engineering e encoding — `src/modeling/pipeline.py`
- [x] Pipeline sklearn (pré-processamento + modelo) — `src/modeling/pipeline.py`
- [x] Treinamento, validação e otimização — `src/modeling/run_baseline.py` (split em `src/modeling/split.py`: só ano de 2025, 70/30 aleatório estratificado por `TARGET`)

## Escolha do algoritmo

Comparamos `LogisticRegression` e `RandomForestClassifier`, com e sem o
enriquecimento socioeconômico, restringindo a modelagem ao ano de 2025 com
split aleatório 70/30 estratificado por `TARGET` (ver `src/modeling/split.py`
e "Limitações do projeto" sobre por que abandonamos o split temporal
2023-2024→2025 usado numa primeira versão). O **RandomForest com
enriquecimento** venceu em todas as métricas (ver tabela abaixo) e foi o
único cenário que superou a acurácia de "sempre prever a classe
majoritária" no teste. Escolhido por ser o melhor resultado e por permitir
a interpretação via SHAP (Seção "Interpretação dos resultados").

## Métricas de avaliação

Split: só ano de 2025 (2.222.792 linhas), 70/30 aleatório estratificado por
`TARGET` — treino com 1.555.954 linhas, teste com 666.838, **58,6% de
alfabetizados nos dois lados** (por construção, elimina o *dataset shift*
que um split temporal por ano introduziria).

| Modelo | Enriquecimento | Acurácia | Precisão | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression | não | 0,5962 | 0,6086 | 0,8717 | 0,7168 | 0,6085 |
| LogisticRegression | sim | 0,5960 | 0,6117 | 0,8506 | 0,7117 | 0,6142 |
| RandomForest | não | 0,5920 | 0,5952 | 0,9508 | 0,7321 | 0,6086 |
| **RandomForest** | **sim** | **0,6126** | **0,6134** | 0,9172 | 0,7351 | **0,6402** |

*(baseline de "sempre prever a classe majoritária" no teste = 0,5862 de
acurácia — ver `reports/baseline_comparison.csv`)*

**Quebra por região** (melhor modelo — RandomForest + enriquecimento,
`reports/baseline_metricas_por_regiao.csv`):

| Região | Taxa real de alfabetização | Acurácia | Precisão | Recall |
|---|---|---|---|---|
| Norte | 52,0% | 0,5385 | 0,5314 | 0,9447 |
| Sudeste | 57,3% | 0,5975 | 0,5983 | 0,9056 |
| Sul | 58,7% | 0,6374 | 0,6433 | 0,8569 |
| Nordeste | 60,8% | 0,6348 | 0,6371 | 0,9275 |
| Centro-Oeste | 66,5% | 0,6655 | 0,6654 | 0,9999 |

Diferente do split temporal (versão anterior), o recall não colapsa mais
numa região específica — mas fica alto (85-100%) em todas, sinal de que o
modelo está enviesado para prever "alfabetizado" com mais frequência que
deveria (ver quebra por UF abaixo, mais reveladora).

**Quebra por UF** (mesmo modelo, `reports/baseline_metricas_por_uf.csv`)
continua degenerada, e mudou de padrão: **16 das 27 UFs (59%)** têm recall
**exatamente 1,0** (TO, DF, AC, RO, MS, AL, PE, MA, PB, MG, PR, MT, ES, GO,
PI, CE) — o modelo *sempre* prevê "alfabetizado" nessas UFs — contra
nenhuma UF com recall 0 desta vez. Ou seja, a mudança de split (temporal →
aleatório 2025) **não resolveu a degeneração — ela piorou** (59% vs. 44%
das UFs antes), só trocou de direção (antes tinha UFs travadas em "nunca
alfabetizado" e outras em "sempre alfabetizado"; agora é quase só "sempre
alfabetizado"). Confirma que o problema é estrutural (granularidade das
features), não do desenho do split — ver "Interpretação dos resultados".

**Teste adicional: infraestrutura escolar (Censo Escolar 2025).** Testamos
somar `PC_ESCOLAS_BIBLIOTECA`, `PC_ESCOLAS_LAB_INFORMATICA` e
`PC_ESCOLAS_INTERNET_ALUNOS` (% de escolas em atividade no município com
cada recurso) ao melhor cenário. Resultado: acurácia 0,6112, F1 0,7365,
AUC 0,6425 (leve melhora de AUC) mas **19/27 UFs degeneradas** (pior que
sem essa adição) — mais uma confirmação de que, sendo uma feature no mesmo
nível de granularidade municipal das outras 3, não ataca a causa raiz.
Ver `reports/baseline_metricas_por_uf_com_infraestrutura.csv`.

## Interpretação dos resultados

SHAP (`TreeExplainer`, amostra de 20 mil linhas do teste — ver
`src/modeling/explain.py` e `reports/shap_importancia.csv`) no RandomForest
com enriquecimento, split 2025:

| Variável | Importância média (\|SHAP\|) |
|---|---|
| `SG_UF` | 0,069 |
| `PC_FAMILIAS_POBREZA` | 0,024 |
| `REGIAO` | 0,018 |
| `RENDA_PER_CAPITA_MEDIA` | 0,010 |
| `MEDIA_INSE` | 0,007 |
| `TP_DEPENDENCIA` | 0,003 |

- **`SG_UF` ainda domina, mas com uma margem bem menor** que no split
  temporal (2,9x `REGIAO` agregada, contra 4,5x antes) — puxado
  principalmente por `SG_UF=CE`, o único estado que sozinho (0,013) chega
  perto de `PC_FAMILIAS_POBREZA` (0,024).
- **`PC_FAMILIAS_POBREZA` agora é a 2ª variável mais importante do modelo
  isoladamente** — na frente até da `REGIAO` inteira somada (0,024 vs.
  0,018) e de qualquer UF individual exceto `CE`. Com o split anterior ela
  ficava atrás de `SG_UF=CE` e `SG_UF=BA` juntas; aqui é a que mais pesa
  depois do estado. Reforça a decisão de manter as features
  socioeconômicas mesmo com correlação linear fraca a nível de aluno
  (Seção 8 da EDA).
- **A disparidade regional continua identificável, mas mais amena**: o
  efeito médio (com sinal) de `REGIAO=Norte` é -0,017 (era -0,037 no split
  anterior) — segue sendo o mais negativo, mas a distância pras demais
  regiões caiu. Ver `reports/shap_efeito_regiao.csv`.
- **A degeneração por UF piorou apesar da importância de `SG_UF` cair
  relativamente** (passo de 44% pra 59% das UFs) — evidência de que a
  causa não é só "o modelo dá peso demais a `SG_UF`", é a combinação de
  poucas features com granularidade municipal/estadual e um limiar de
  decisão fixo (0,5) que, nesse recorte, acaba quase sempre acima de 0,5
  na maioria dos estados (ver "Métricas de avaliação" — recall de 85-100%
  em todas as regiões).

## Insights encontrados

- **Target balanceado**: 52,2% alfabetizados vs. 47,8% não alfabetizados — não requer balanceamento artificial (SMOTE/undersampling).
- **Desigualdade regional**: ~15 pontos percentuais de diferença entre a melhor região (Centro-Oeste, 57,5%) e a pior (Norte, 42,2%).
- **Rede**: base majoritariamente municipal (87%); amostra da rede privada é irrisória (25 alunos) e não deve ser usada para conclusões.
- **Data leakage crítico identificado na EDA**: `TARGET` é 100% determinístico a partir de `VL_PROFICIENCIA_LP >= 743` (sem exceções) e de flags de participação na prova. Essas variáveis (e derivadas) foram excluídas do conjunto de features de modelagem — ver `notebooks/01_EDA_Alfabetizacao.ipynb`, seção 7.
- Agregados municipais/estaduais (`PC_ALUNO_ALFABETIZADO*`, `VL_MEDIA_LP*`) têm vazamento parcial (incluem o próprio aluno no cálculo) e devem ser usados com cautela ou recalculados como *leave-one-out*.
- **`DESEMPENHO_RELATIVO` também é leakage** (derivado de `DIF_MEDIA_ESTADO`, que vem de `VL_PROFICIENCIA_LP`) e não estava na lista original da EDA — encontrado ao formalizar a seleção de features em código (`src/modeling/features.py`).
- **`ID_ALUNO` e `ID_ESCOLA` não são identificadores persistentes entre anos**: as faixas numéricas se repetem quase idênticas em 2023/2024/2025 e, dos "mesmos" códigos que aparecem em anos diferentes, praticamente 0% correspondem à mesma escola/mesmo município — são IDs re-sorteados a cada ano (o dicionário oficial do INEP confirma: `ID_ESCOLA` é "máscara do código da escola, códigos fictícios"), não registros nacionais estáveis. Isso fecha a porta pra usar `ID_ESCOLA` via encoding como fonte de sinal individual (`CO_ENTIDADE` do Censo Escolar, que é o código real, também não bate: 0% de correspondência testada).
- **Nenhuma feature disponível varia por aluno**: com as 7 features atuais (`REGIAO`, `SG_UF`, `TP_DEPENDENCIA` + 3 socioeconômicas por município), os 2.222.792 alunos de 2025 colapsam em só ~6.500 combinações únicas de valores — em média, **~340 alunos compartilham exatamente a mesma linha de entrada**. O modelo não pode, por construção, diferenciar esses alunos entre si; só pode prever por grupo.
- **Sem o enriquecimento externo, o modelo mal supera prever a classe majoritária** (`REGIAO`/`SG_UF`/`TP_DEPENDENCIA` sozinhos: acurácia 0,592-0,596 vs. baseline de 0,586) — quase todo o sinal individual forte foi removido como leakage, então o que resta é fraco por natureza.
- **O modelo tem recall degenerado por UF** (sempre prevê a mesma classe pra 59% dos estados) — ver "Interpretação dos resultados".
- **Agregar infraestrutura escolar (Censo Escolar) por município não resolveu a degeneração** — é a mesma granularidade municipal das outras 3 features, então não ataca a causa raiz (falta de variação por aluno).

## Limitações do projeto

- O indicador do INEP trata "aluno não avaliado" como "não alfabetizado" por definição — mistura duas populações conceitualmente diferentes (quem não aprendeu vs. quem não fez a prova).
- Amostra da rede privada é pequena demais para generalizar.
- Reconstrução da camada Gold feita localmente (fora do Databricks/AWS original da Fase 2) — ver seção "Reconstrução da camada Gold" abaixo para detalhes e possíveis pequenas diferenças de metodologia.
- Renda (Censo 2022) e INSE (SAEB 2023) são fotos únicas, repetidas nos 3 anos do painel (2023-2025) — ver "Enriquecimento externo por município" abaixo.
- **Degeneração de recall por UF (achado central da modelagem, persiste em toda tentativa de correção)**: em 59% das UFs (16 de 27) o modelo sempre prevê "alfabetizado", independente do aluno. Testamos 3 correções (split aleatório em vez de temporal, adicionar infraestrutura escolar por município, ambas documentadas em "Métricas de avaliação") e nenhuma resolveu — a causa é estrutural: nenhuma feature disponível varia por aluno dentro do mesmo município/rede, então ~340 alunos em média compartilham a mesma previsão. **Este modelo não deve ser usado para decisões de política pública individual** (ver "Aplicação prática" e "Evoluções futuras").
- **Modelagem restrita ao ano de 2025** (split aleatório 70/30, ver `src/modeling/split.py`): descartamos o split temporal (2023-2024 → 2025) usado numa primeira versão porque a taxa real de alfabetização mudava entre os recortes (51,3% vs. 58,6%), misturando "o modelo generaliza mal" com "o mundo mudou entre os anos". O trade-off é não testar a capacidade do modelo de prever um ano futuro nunca visto — só validamos generalização dentro do mesmo ano.
- **`ID_ESCOLA` não pode ser usado para trazer sinal por escola**: é uma máscara re-sorteada a cada ano pelo INEP (não é o `CO_ENTIDADE` real usado no Censo Escolar, e nem é estável entre 2023-2025) — fecha a porta pra qualquer enriquecimento por escola individual com os dados públicos disponíveis.

## Aplicação prática para políticas públicas

Na forma atual, o modelo serve melhor como **ferramenta exploratória** —
ex.: cruzar `PC_FAMILIAS_POBREZA`/`MEDIA_INSE`/`RENDA_PER_CAPITA_MEDIA` por
município pra priorizar visitas técnicas ou repasse de material — do que
como critério automático de decisão. O comportamento degenerado por UF
(recall ≈100% em 16 de 27 estados) significa que, se usado para sinalizar
"alunos em risco" hoje, o modelo simplesmente **não identificaria quase
ninguém em risco** nesses estados — prevê "alfabetizado" quase sempre,
mesmo para os ~40% que não estão. Qualquer uso em política pública exige
primeiro corrigir esse comportamento (ver "Evoluções futuras") e, enquanto
isso não acontece, avaliar as previsões sempre segmentadas por UF — a
região é granularidade grossa demais até pra diagnosticar o problema (ver
EDA, Seção 5).

## Possíveis evoluções futuras

- **Corrigir o comportamento degenerado por UF** continua o item mais
  urgente. Já testamos e descartamos 3 caminhos: reduzir `max_depth`
  (piora tudo — 5/4/3 tiveram mais UFs degeneradas e AUC pior que
  `max_depth=10`), trocar o split temporal por aleatório dentro de 2025
  (piorou, de 44% pra 59% das UFs) e somar infraestrutura escolar por
  município (piorou levemente, 19/27). O padrão que emerge: como nenhuma
  feature varia por aluno, ajustar hiperparâmetro ou split só desloca
  *onde* a degeneração aparece, não a remove. Caminhos ainda não testados:
  calibração do limiar de decisão por UF (em vez do 0,5 fixo), ou métricas
  de otimização sensíveis a fairness por grupo (ex.: equalized odds)
  usando `SG_UF` como grupo protegido — mas o teto real só sobe com uma
  fonte de dado que varie dentro do município (não temos uma disponível,
  ver limitação sobre `ID_ESCOLA`).
- [x] ~~Revisitar a EDA trazendo a desigualdade regional/por UF excluindo
  a rede privada~~ — feito em `notebooks/01_EDA_Alfabetizacao.ipynb`,
  Seção 5 (branch `feature/eda-desigualdade-regional-sem-privada`): achado
  de 47pp de disparidade por UF (vs. ~13pp por região) foi o que motivou
  adicionar a quebra por UF nesta seção.
- [x] ~~Testar `ID_ESCOLA` via encoding específico~~ — testado e
  descartado: `ID_ESCOLA` é uma máscara re-sorteada a cada ano pelo INEP
  (não corresponde a uma escola real estável), então não carrega
  informação de treino (2025) pra nenhum outro ano, e mesmo dentro de
  2025 não existe um `CO_ENTIDADE` real pra cruzar com o Censo Escolar.
- [x] ~~Agregar infraestrutura escolar (Censo Escolar) por município~~ —
  testado (`PC_ESCOLAS_BIBLIOTECA`, `PC_ESCOLAS_LAB_INFORMATICA`,
  `PC_ESCOLAS_INTERNET_ALUNOS`) e o efeito foi nulo/levemente negativo —
  ver "Métricas de avaliação".
- Obter acesso ao Atlas do Desenvolvimento Humano (IDHM) — não conseguimos
  a tempo (Atlas Brasil indisponível) e usamos CadÚnico/Censo/INSE como
  substituto (ver "Enriquecimento externo por município").
- Tuning de hiperparâmetros do RandomForest (além de `max_depth`, já
  testado) e teste de outros algoritmos em árvore (XGBoost, LightGBM)
  agora que o pipeline (`src/modeling/pipeline.py`) já injeta qualquer
  classificador sklearn sem mudança de código.

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
