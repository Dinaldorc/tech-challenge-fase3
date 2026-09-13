# Roteiro — Vídeo Executivo (Tech Challenge Fase 3)

Vídeo de **até 5 minutos**, simulando uma reunião executiva com gestores
públicos, lideranças ou stakeholders (exigência do enunciado). Roteiro
alinhado slide a slide com `Apresentacao_tech_challenge_fase3.pptx` (11
slides), na divisão que o grupo já definiu. Texto de apoio completo por
slide — cada pessoa pode adaptar o tom pra sua própria voz, mas os
números e afirmações técnicas devem ser ditos como estão (batem com o
que aparece no próprio slide e com o README).

**Duração-alvo do roteiro: ~4min05 a 4min45**, deixando folga sobre o
limite de 5 minutos — gravação ao vivo sempre roda mais devagar que o
texto lido silenciosamente. Contagem de palavras validada por script
(538 palavras no total).

## Quadro-resumo

| Apresentador | Slides | Conteúdo | Tempo-alvo |
|---|---|---|---|
| **Luís Gustavo** | 1–4 | Abertura, contexto/objetivo, os dois modelos, base de dados | ~90-100s |
| **Dinaldo** | 5–6 | Principais achados da EDA, pipeline de modelagem | ~45-50s |
| **Afonso** | 7–8 | Escolha do algoritmo, otimização de hiperparâmetros | ~45-55s |
| **Bruno** | 9–11 | Resultados finais, interpretação/SHAP, conclusões e fechamento | ~65-75s |

---

## Luís Gustavo — Slides 1 a 4

### Slide 1 — Abertura

> Bom dia. Somos o grupo 28 do Pós-Tech em Data Analytics, e vamos
> apresentar nosso modelo supervisionado de predição de alfabetização no
> Brasil, construído sobre os microdados do INEP: 5.570 municípios, 6,09
> milhões de alunos, entre 2023 e 2025.

### Slide 2 — Contexto e Objetivo

> O Compromisso Nacional Criança Alfabetizada tem meta de alfabetizar
> toda criança até o fim do 2º ano. Mas o INEP só identifica quem não se
> alfabetizou **depois** da prova — tarde demais pra agir. Nosso
> objetivo: prever com antecedência o risco de não alfabetização, a
> partir de variáveis territoriais e socioeconômicas já conhecidas, pra
> ação proativa dos gestores.

### Slide 3 — Abordagem: os dois modelos

> Construímos dois modelos complementares. O de **aluno** cumpre o
> objetivo técnico central: pipeline completo, tratamento de vazamento e
> validação cruzada. Mas o SHAP revelou um limite real: 2,2 milhões de
> alunos colapsam em só 6.500 combinações de características, sem sinal
> individual suficiente. Isso motivou o modelo **municipal**, que
> responde às perguntas de negócio do desafio.

### Slide 4 — Base de Dados

> Nossa base principal tem 6,09 milhões de linhas, das quais 2,2 milhões
> usadas na modelagem, com foco em 2025. Enriquecemos com três fontes
> externas por município: pobreza pelo Cadastro Único, renda pelo Censo
> 2022, e nível socioeconômico escolar pelo INSE.

---

## Dinaldo — Slides 5 e 6

### Slide 5 — Análise Exploratória: Principais Achados

> Três achados guiaram nossas decisões. O target é moderadamente
> desbalanceado, 61% contra 39%. A disparidade real está por **estado**,
> não por região: Ceará com 84,5%, Sergipe com 40,1%, mesma região. E
> identificamos vazamento crítico: o target era determinístico a partir
> da proficiência — removemos essa variável e suas derivadas.

### Slide 6 — Metodologia: Pipeline de Modelagem

> Nossa metodologia usa um pipeline `scikit-learn` único, unindo
> pré-processamento e modelo. Tratamos vazamento de dados, fizemos
> engenharia de atributos territoriais e socioeconômicos, com split
> estratificado 70/30 restrito a 2025. E aplicamos o peso amostral
> oficial do INEP em treino e métricas, pra representar a população
> real, não só a amostra.

---

## Afonso — Slides 7 e 8

### Slide 7 — Escolha do Algoritmo

> Comparamos regressão logística e Random Forest, com e sem
> enriquecimento socioeconômico. O Random Forest tunado com
> enriquecimento venceu em todas as métricas — acurácia de 63%, AUC de
> 0,66 — o único cenário a superar o baseline de sempre prever a classe
> majoritária, e escolhido também por permitir interpretação via SHAP.

### Slide 8 — Otimização de Hiperparâmetros

> Otimizamos hiperparâmetros com busca sistemática: `RandomizedSearchCV`
> no aluno, `GridSearchCV` no municipal. Achado importante: a validação
> cruzada do municipal reportou AUC de 0,77 no treino, mas caiu pra 0,66
> no teste real — reforça validar sempre num recorte temporal genuíno. No
> aluno, o tuning reduziu de 67% pra 30% os estados com recall
> degenerado.

---

## Bruno — Slides 9 a 11

### Slide 9 — Resultados Finais

> No teste final, com quase 667 mil alunos nunca vistos, alcançamos
> acurácia de 63%, recall de 82% e AUC de 0,66 — contra um baseline de
> 59%.

### Slide 10 — Interpretação (SHAP) e Perguntas de Negócio

> O SHAP aponta o estado como variável de maior contribuição, com a
> pobreza familiar entre as mais relevantes. O modelo municipal pontua os
> 5.500 municípios pela probabilidade de não bater a meta, com recall de
> até 80% após calibração. O clustering por perfil, por sua vez, não
> coincide com as cinco regiões do IBGE — o Nordeste se divide em dois
> grupos opostos.

### Slide 11 — Conclusões e Próximos Passos

> Limitação clara: cerca de 340 alunos compartilham a mesma previsão por
> combinação de características — é ferramenta de perfil e priorização
> territorial, não de decisão individual. Recomendamos cruzar pobreza,
> renda e nível socioeconômico por município pra priorizar ações, sempre
> por estado. Próximos passos: calibração por UF no aluno e outros
> algoritmos. Muito obrigado.

---

## Checklist antes de gravar

- [ ] Cada apresentador decorou/adaptou seus slides pro próprio jeito de
  falar (não precisa ser decoreba literal do texto acima, mas os
  **números** devem ficar exatamente como estão — batem com os slides e
  com `reports/`).
- [ ] Cronometrar um ensaio completo em voz alta, passando o slide junto
  com a fala, antes da gravação final — o texto foi calculado pra
  ~4min05-4min45, mas ritmo de fala varia por pessoa.
- [ ] Evitar linguagem causal solta ("a pobreza causa baixa
  alfabetização") — sempre "está associada a" / "contribui para a
  previsão do modelo" (ver seção "O que os modelos entregam — e o que
  não entregam" no README).
- [ ] Evitar dizer que o modelo "decide quem vai ser alfabetizado" — ele
  estima risco/probabilidade, é instrumento de priorização.
- [ ] Prestar atenção na transição entre apresentadores (slides 4→5,
  6→7, 8→9) — combinar uma frase de passagem curta pra não ficar
  abrupto.
- [ ] Testar áudio e enquadramento de câmera antes de gravar a versão
  final — sem cortes/edição, mais fácil regravar um trecho curto do que
  a íntegra.
- [ ] Confirmar que a gravação final está **dentro de 5 minutos** (usar
  cronômetro, não estimar de ouvido).
