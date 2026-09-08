# Roteiro — Vídeo Executivo (Tech Challenge Fase 3)

Vídeo de **até 5 minutos**, simulando uma reunião executiva com gestores
públicos, lideranças ou stakeholders (exigência do enunciado). Dividido em
**4 partes, uma por integrante**, com texto de apoio completo pra cada
parte — cada pessoa pode adaptar o tom pra sua própria voz, mas os números
e afirmações técnicas devem ser ditos como estão (são os números reais do
projeto, citados no README).

**Duração-alvo do roteiro: ~4min20 a 4min30**, deixando ~30-40s de folga
sobre o limite de 5 minutos — gravação ao vivo sempre roda mais devagar que
o texto lido silenciosamente.

## Quadro-resumo

| Parte | Integrante | Conteúdo | Tempo-alvo | Acumulado |
|---|---|---|---|---|
| 1 | Integrante 1 | Contexto e problema educacional | ~65s | 0:00–1:05 |
| 2 | Integrante 2 | Abordagem: os dois modelos (aluno + município) | ~70s | 1:05–2:15 |
| 3 | Integrante 3 | Principais insights | ~70s | 2:15–3:25 |
| 4 | Integrante 4 | Valor estratégico, aplicação em políticas públicas, fechamento | ~65s | 3:25–4:30 |

Ajustem a atribuição de partes entre vocês 4 como preferirem — a divisão
acima é só uma sugestão de bloco de conteúdo, não uma atribuição fixa de
nomes.

---

## Parte 1 — Contexto e problema educacional

**Tempo-alvo: ~65s (~140 palavras)**

> Bom dia. Obrigado por nos receberem. Hoje vamos apresentar como
> transformamos os dados públicos do INEP sobre alfabetização infantil em
> inteligência aplicável à gestão educacional.
>
> O Brasil tem a meta de alfabetizar toda criança até o final do 2º ano do
> Ensino Fundamental — o Compromisso Nacional Criança Alfabetizada, com
> prazo até 2030. Mas o resultado varia enormemente pelo território:
> encontramos até 44 pontos percentuais de diferença entre o Ceará, com
> 84,5% de alfabetização, e Sergipe, com 40,1% — dois estados do mesmo
> Nordeste, uma desigualdade que se esconde quando olhamos só pela região.
>
> E hoje o INEP só identifica quem não se alfabetizou **depois** da prova
> — quando já é tarde para agir naquele ano letivo. Foi esse o problema que
> decidimos atacar: antecipar risco, não só medir resultado.

**Sugestão de slide**: título do projeto + meta 2030 + o mapa/gráfico de
disparidade CE vs. SE (pode reaproveitar `images/recall_por_uf_aluno.png`
ou um gráfico novo só com a taxa real por UF, sem métrica de modelo).

---

## Parte 2 — Abordagem: os dois modelos

**Tempo-alvo: ~70s (~150 palavras)**

> O desafio pedia, como objetivo central, um modelo que previsse se um
> aluno individual seria alfabetizado. Construímos esse modelo — com
> pipeline completo em scikit-learn, tratamento rigoroso de vazamento de
> dados e validação cruzada — e ele cumpre esse objetivo.
>
> Mas a própria interpretabilidade do modelo, via SHAP, revelou algo
> importante: as variáveis públicas disponíveis explicam muito mais o
> contexto territorial e socioeconômico do aluno do que o aluno
> individualmente. Dois milhões de alunos colapsam em pouco mais de seis
> mil combinações únicas de características — um limite real dos dados
> públicos, não do modelo.
>
> Foi esse achado que nos levou a construir um segundo modelo, agora em
> nível de **município**, para responder diretamente às perguntas de
> negócio deste desafio: quais fatores mais influenciam, quais municípios
> têm mais risco, quais têm padrões semelhantes, e como prever quem vai
> deixar de bater a meta.

**Sugestão de slide**: diagrama simples com 2 caixas — "Modelo de Aluno
(objetivo técnico central)" e "Modelo Municipal (perguntas de negócio)" —
ligadas pela seta "achado de interpretabilidade". Pode reaproveitar a
lógica da seção "Objetivo analítico" do README.

---

## Parte 3 — Principais insights

**Tempo-alvo: ~70s (~150 palavras)**

> Três achados se destacam. Primeiro: o **estado** importa mais que a
> região — a desigualdade não é difusa, está concentrada em poucos estados
> específicos, e pobreza municipal é a segunda variável mais influente nas
> previsões.
>
> Segundo: nosso ranking de risco mostrou que os municípios mais
> arriscados hoje são majoritariamente do **Rio Grande do Sul** — não
> porque o desempenho absoluto seja ruim, mas porque as metas deles são
> muito exigentes em relação à própria trajetória histórica. Risco de não
> bater meta e baixo desempenho são coisas diferentes, e essa distinção
> muda a prioridade de investimento.
>
> Terceiro: o agrupamento de municípios por perfil **não seguiu as cinco
> regiões oficiais do Brasil** — encontramos municípios de regiões
> diferentes com comportamento educacional parecido, e municípios vizinhos
> com trajetórias opostas.

**Sugestão de slide**: 3 blocos numerados (um por achado), cada um com uma
imagem de apoio — `images/shap_importancia_municipal.png` (achado 1),
recorte do `municipal_ranking_risco.csv` com o top 5 do RS (achado 2),
`images/municipal_clusters_perfil.png` (achado 3).

---

## Parte 4 — Valor estratégico, aplicação e fechamento

**Tempo-alvo: ~65s (~140 palavras)**

> Esse modelo municipal já é uma ferramenta real de priorização: consegue
> identificar até 80% dos municípios que não vão bater a meta, calibrando
> o critério de decisão pro que interessa ao gestor — sem retreinar nada.
>
> Mas somos honestos sobre os limites: o modelo não decide sobre um aluno
> específico, não prova causa e efeito, e deve **apoiar** decisão — não
> substituí-la. O uso recomendado é cruzar pobreza, renda e nível
> socioeconômico por município pra priorizar visita técnica, reforço
> escolar e material didático, sempre olhando por estado, não só por
> região.
>
> A arquitetura é replicável para qualquer unidade federativa do país.
> Nosso convite é simples: dados públicos brasileiros já têm sinal
> suficiente para orientar política educacional com mais precisão — falta
> usá-los assim. Obrigado.

**Sugestão de slide**: recall antes/depois da calibração de limiar (64% →
80%) + lista curta "o modelo apoia, não substitui" + slide final de
encerramento (nome do projeto/equipe).

---

## Checklist antes de gravar

- [ ] Cada integrante decorou/adaptou sua parte pro próprio jeito de falar
  (não precisa ser decoreba literal do texto acima, mas os **números**
  devem ficar exatamente como estão — são verificados no `reports/`).
- [ ] Cronometrar um ensaio completo em voz alta antes da gravação final —
  o texto foi calculado pra ~4min20-4min30, mas ritmo de fala varia por
  pessoa.
- [ ] Evitar linguagem causal solta ("a pobreza causa baixa
  alfabetização") — sempre "está associada a" / "influencia a previsão do
  modelo" (ver seção "O que os modelos entregam — e o que não entregam" no
  README).
- [ ] Evitar dizer que o modelo "decide quem vai ser alfabetizado" — ele
  estima risco/probabilidade, é instrumento de priorização.
- [ ] Testar áudio e enquadramento de câmera antes de gravar a versão
  final — sem cortes/edição, mais fácil regravar um trecho curto do que a
  íntegra.
- [ ] Confirmar que a gravação final está **dentro de 5 minutos** (usar
  cronômetro, não estimar de ouvido).

## Próximo passo: slides

Este roteiro já sugere o conteúdo de slide por parte. Falta desenhar os
slides de fato (pode ser Google Slides/PowerPoint/Canva — fora do escopo
deste repositório) reaproveitando as imagens já geradas em `images/` e os
números deste roteiro, que já batem com o README.
