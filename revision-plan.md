# Plano de Revisão — APEN-D-25-26009

**Artigo:** Privacy-Preserving Electricity Demand Forecasting: Key Aspects and Performance Evaluation of Federated Learning  
**Destino:** Applied Energy — Special Issue *"Artificial Intelligence-Driven Solutions for Distribution Networks"*  
**Status:** Revisão pós-rejeição com convite de resubmissão

---
## Seq. de ataque
G4 → G5 → G2 (requer código) → G3 (requer código) → G1 → G7 → G6
---

## Grupo 1 — Novelty e Posicionamento das Contribuições

### Comentários dos Revisores

> **Revisor 1:** The manuscript makes strong novelty claims (e.g., "for the first time"), yet the literature review already includes several studies on federated learning (FL) for residential load forecasting, personalization, and robustness. The authors should more clearly articulate what is fundamentally new beyond prior work—specifically, which component of the FL pipeline is being advanced (e.g., repeated client selection, overfitting behavior), and how the proposed "intermediate stages" analysis provides insights not previously reported.

> **Revisor 2:** The novelty is overstated; the authors must clarify exactly what is new compared to existing work (e.g., Liu et al. [22]).

> **Revisor 3:** While the manuscript addresses a relevant topic—privacy-preserving machine learning in energy systems—the contribution is largely incremental and lacks the novelty expected for publication in Applied Energy. The study essentially applies existing FL frameworks to a well-known dataset without introducing new methodological advancements, theoretical insights, or significant improvements over the state-of-the-art.

### Plano

- [ ] **Reescrever a seção de contribuições** para posicionar o artigo como um *estudo diagnóstico sistemático*: a contribuição central é identificar, medir e caracterizar com rigor as disfunções internas do FL (seleção injusta de clientes, overfitting por treinamento repetido) — base necessária antes de propor soluções.
- [ ] **Adicionar parágrafo comparativo com Liu et al. [22]** na revisão bibliográfica, diferenciando explicitamente: Liu foca em detecção de fraude de energia; este artigo foca no comportamento de treinamento e na equidade da seleção de clientes para previsão de demanda.
- [ ] **Remover ou qualificar** todas as frases absolutas ("for the first time") substituindo por posicionamento relativo: *"to the best of our knowledge, this is the first systematic characterization of..."*
- [ ] **Conectar ao Special Issue** na Introdução e na Conclusão: articular como o diagnóstico do FL endereça barreiras reais de implantação em redes de distribuição — o operador da rede precisa compreender os limites do FL antes de implantá-lo.
- [ ] **Adicionar 5–8 referências recentes (2023–2025)** sobre FL aplicado à previsão de energia e gestão de redes de distribuição, para atualizar a revisão bibliográfica e reforçar a lacuna identificada.

---

## Grupo 2 — Rigor Experimental: Multi-Seed e Análise de Sensibilidade

### Comentários dos Revisores

> **Revisor 1:** The experiments rely on a limited number of households (25) and assume ideal FL conditions (no communication failures, no stragglers, no client dropouts). Moreover, using 200 local epochs may unrealistically amplify local overfitting effects. Additional sensitivity analyses (e.g., varying client fractions, local epochs, communication rounds) and reporting communication/computation costs would improve the practical relevance of the results.

> **Revisor 1:** Although paired statistical tests are applied, the evaluation appears sensitive to single-seed runs and specific configurations. Repeating experiments with multiple random seeds, controlling for multiple comparisons, and reporting additional error metrics (e.g., MAE, MAPE/sMAPE, or uncertainty estimates) would improve result robustness.

> **Revisor 2:** The claim of "unfairness" is based on a single run. Multiple simulations with different seeds are needed to prove systematic exclusion.

### Plano

- [ ] **Executar múltiplas seeds** (mínimo 5, idealmente 10) para todos os experimentos. Reportar todos os resultados como **média ± desvio padrão**.
- [ ] **Atualizar Figura 4** (seleção de clientes): mostrar a distribuição de participação agregada sobre múltiplas runs com intervalo de confiança — transforma um resultado de seed única em evidência sistemática.
- [ ] **Análise de sensibilidade** (nova tabela ou figura):

  | Hiperparâmetro | Valores testados | Valor base |
  |---|---|---|
  | Fração de clientes | 0.1, **0.2**, 0.4 | 0.2 |
  | Épocas locais | 50, 100, **200** | 200 |
  | Rodadas de FL | 5, **10**, 20 | 10 |

- [ ] **Reportar custos de comunicação/computação**: número de parâmetros transmitidos por rodada e tempo médio por rodada (Figura 3b já tem dados parciais).
- [ ] *(Opcional)* Expandir de 25 para ~50 clientes usando outros registros disponíveis no Pecan Street, para fortalecer a generalização dos achados.

---

## Grupo 3 — Seleção de Clientes: de Diagnóstico a Comparação com Alternativas

### Comentários dos Revisores

> **Revisor 1:** While the paper correctly observes that random client selection leads to uneven participation and potential overfitting due to repeated selection, this issue is only diagnosed, not mitigated. Including at least one alternative client-selection strategy (e.g., participation-aware, diversity-based, or clustering-based selection) and reporting fairness metrics (e.g., participation distribution, coverage ratio, Jain's index) would substantially strengthen the contribution.

> **Revisor 2:** The claim of "unfairness" is based on a single run. Multiple simulations with different seeds are needed to prove systematic exclusion.

### Plano

- [ ] **Implementar e comparar 1–2 estratégias alternativas de seleção de clientes:**
  - **Participation-aware (fairness):** clientes menos selecionados têm prioridade nas rodadas seguintes — implementação simples, diretamente motivada pelos achados da Figura 4.
  - **Diversidade de dados:** agrupa clientes por perfil de demanda e garante representação proporcional de cada grupo por rodada — conecta com literatura já citada (Tian et al. [38], Putra et al. [36]).
- [ ] **Adicionar métricas de fairness:**
  - Histograma/boxplot da distribuição de participações por cliente
  - Taxa de cobertura: percentual de clientes que participaram em ≥1 rodada
  - Índice de Jain: `J = (Σxᵢ)² / (n · Σxᵢ²)`, onde xᵢ = número de seleções do cliente i (J=1 é perfeitamente justo)
- [ ] **Mostrar impacto no desempenho do modelo:** comparar MSE e R² dos clientes nunca selecionados vs. selecionados, com e sem a seleção por fairness.
- [ ] **Reposicionar as contribuições:** o artigo passa a diagnosticar *e* propor um caminho inicial de solução.

---

## Grupo 4 — Correções Estatísticas (CRÍTICO)

### Comentários dos Revisores

> **Revisor 2:** The results are contradictory. For "New York," the Confidence Interval crosses zero, but the hypothesis is rejected. **This is statistically impossible and must be recalculated.**

> **Revisor 2:** The paper incorrectly states there is "no difference" because the p-value is high. A large effect size with a non-significant p-value means the test lacked power (sample size too small), not that the methods are equal.

> **Revisor 1:** Repeating experiments with multiple random seeds, controlling for multiple comparisons, and reporting additional error metrics (e.g., MAE, MAPE/sMAPE, or uncertainty estimates) would improve result robustness. Furthermore, the discussion of non-IID data and statistical assumptions should be more precise and consistent.

### Plano

- [ ] **Recalcular a Tabela 3 completa:** revisar todos os intervalos de confiança e testes de hipótese. Verificar:
  - Se os testes são bilaterais ou unilaterais (e se isso está consistente com a hipótese formulada)
  - Se o cálculo de variância/erro padrão está correto
  - Corrigir todas as inconsistências entre p-value e IC — um IC 95% que cruza zero implica *não rejeitar* H₀ ao nível 5%, nunca o contrário
- [ ] **Reescrever a discussão do caso Comunidade:** não concluir "ausência de diferença". Em vez disso:
  - Reportar o effect size (Cohen's d ou equivalente)
  - Discutir a limitação de poder estatístico com n=25 clientes
  - Concluir que o estudo é *inconclusivo* para esse cenário, não que os métodos são equivalentes
- [ ] **Adicionar MAE e MAPE/sMAPE** a todas as tabelas de resultados (padrão na literatura de previsão de demanda).
- [ ] **Controle de comparações múltiplas:** com 4 arquiteturas × 25 clientes = 100 comparações, aplicar correção de Bonferroni ou Benjamini-Hochberg nos testes de hipótese.

---

## Grupo 5 — Evidência de Overfitting (Figura 7)

### Comentários dos Revisores

> **Revisor 2:** Evidence is weak. The paper shows training loss only; validation loss is required to prove overfitting.

### Plano

- [ ] **Atualizar a Figura 7** para exibir curvas de **treinamento (linha sólida) e validação (linha tracejada)** sobrepostas para o cliente 8386 em cada rodada em que foi selecionado. A divergência entre as duas curvas é a evidência visual do overfitting.
- [ ] **Quantificar o overfitting:** reportar a diferença entre o MSE de validação no início e no final do treinamento para as rodadas de seleção repetida.
- [ ] **Verificar generalização do achado:** usando os dados de multi-seed do Grupo 2, checar se outros clientes selecionados múltiplas vezes também exibem o mesmo padrão e reportar quantos.

---

## Grupo 6 — Formatação e Terminologia

### Comentários dos Revisores

> **Revisor 2:** Several figures and tables are broken or poorly formatted (e.g., Table 3 and Fig. 8).

> **Revisor 2:** Terminology needs to be more consistent along the manuscript.

> **Revisor 2:** The manuscript needs extensive language editing by a native speaker.

### Plano

- [ ] **Corrigir a Tabela 3:** reformatar completamente — alinhar colunas, corrigir valores numéricos, aplicar negrito nos melhores resultados e separar claramente as colunas de Global Model e Individual Model.
- [ ] **Melhorar a Figura 8:** aumentar resolução, padronizar legenda e garantir que os 4 modelos (CNN, RNN, LSTM, GRU) estejam claramente diferenciados por cores e estilos de linha distintos.
- [ ] **Padronizar terminologia** em todo o manuscrito:
  - Usar **"global model"** (não "aggregated model")
  - Usar **"client"** (não "participant" nem "node")
  - Expandir "FL" por extenso na primeira ocorrência de cada seção
- [ ] **Revisão completa de inglês:** priorizar Introdução, Discussão e Conclusão. Considerar ferramenta de escrita acadêmica ou revisor profissional nativo.

---

## Grupo 7 — Alinhamento com o Special Issue (sugestão nova)

### Motivação

O Special Issue *"AI-Driven Solutions for Distribution Networks"* tem foco explícito em redes de distribuição. O artigo trata de FL para previsão residencial, mas não conecta diretamente com os desafios operacionais das redes de distribuição — o que reduz o fit com o call.

### Plano

- [ ] **Adicionar parágrafo na Introdução** conectando previsão de demanda residencial com desafios das redes de distribuição: gestão de carga e flexibilidade, integração de geração distribuída (solar, armazenamento), e o operador da rede como o "servidor" natural no cenário FL.
- [ ] **Adaptar a Conclusão** para incluir implicações práticas para o operador de sistema: quais limitações do FL (identificadas no artigo) precisam ser resolvidas antes da implantação em redes reais.
- [ ] **Reescrever a Cover Letter** para o Special Issue: enfatizar como o diagnóstico do FL endereça barreiras concretas de implantação em redes de distribuição — privacidade dos consumidores, heterogeneidade de dispositivos e equidade na seleção de participantes.

---

## Sequência de Execução

| # | Grupo | Prioridade | Esforço | Depende de |
|---|---|---|---|---|
| 1 | G4 — Estatística | 🔴 Crítico | Médio | — |
| 2 | G5 — Figura 7 | 🔴 Crítico | Baixo | — |
| 3 | G2 — Multi-seed | 🟠 Alto | Alto (requer código) | — |
| 4 | G3 — Seleção alternativa | 🟠 Alto | Alto (requer código) | G2 |
| 5 | G1 — Novelty | 🟠 Alto | Médio | G3 |
| 6 | G7 — Special Issue | 🟡 Médio | Baixo | G1 |
| 7 | G6 — Formatação | 🟡 Médio | Médio | Todos |

---

## Checklist Final

- [ ] Tabela 3: IC e p-value coerentes em todos os casos
- [ ] Figura 7: curvas de treinamento E validação sobrepostas
- [ ] Todos os resultados: média ± desvio padrão de múltiplas seeds
- [ ] Figura 4: distribuição de participação sobre múltiplas runs
- [ ] Pelo menos 1 estratégia alternativa de seleção de clientes implementada e comparada
- [ ] Jain's Index e taxa de cobertura reportados
- [ ] MAE e MAPE adicionados a todas as tabelas de resultados
- [ ] Correção para comparações múltiplas aplicada
- [ ] Terminologia padronizada em todo o manuscrito ("global model", "client")
- [ ] Figura 8 e Tabela 3 corrigidas e reformatadas
- [ ] Introdução com conexão explícita ao Special Issue e às redes de distribuição
- [ ] Cover Letter reescrita para o Special Issue
