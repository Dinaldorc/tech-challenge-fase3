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
_(preencher)_

## Limitações do projeto
_(preencher)_

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
