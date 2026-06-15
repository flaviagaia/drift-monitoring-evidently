# Drift Monitoring — Data, Embedding e Concept Drift

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · numpy · scipy · scikit-learn · Evidently (opcional) · 100% offline, sem API key · MIT License

---

## 🇧🇷 Português

### A tese

"O modelo está em produção" não é o fim — é o começo do problema. O mundo muda e
o modelo, parado, apodrece. Mas **drift não é uma coisa só**: existem três tipos,
e cada um precisa de uma lente diferente. Monitorar a errada dá falsa sensação de
segurança.

Este repositório implementa os detectores **do zero** (para a mecânica ficar clara)
e mostra os três tipos em cenários sintéticos determinísticos. Roda em ~2 segundos.

### Os três tipos (com números reais deste repo)

**C31 — Data drift (covariate shift): muda P(X).** A distribuição da entrada se
desloca. Detectado por PSI e teste KS, feature a feature.

| cenário | PSI | label | KS p-valor | alarme |
| ------- | --- | ----- | ---------- | ------ |
| sem drift | 0.003 | estável | 0.90 | ok |
| com drift | 1.444 | forte | 0.0000 | 🚨 |

**C32 — Embedding drift: o sinal é multivariado.** Um deslocamento sutil numa
direção do espaço de embedding move cada dimensão **pouco** (PSI por dimensão
fica estável), mas o conjunto muda. PSI feature a feature **não pega**; o
classificador de domínio sim.

| cenário | PSI médio/dim | AUC classificador de domínio | alarme |
| ------- | ------------- | ---------------------------- | ------ |
| sem drift | 0.014 | 0.521 | ok |
| com drift | 0.064 (estável!) | 0.743 | 🚨 |

> A lição do post: **embedding você não monitora dimensão a dimensão.** O drift
> mora na correlação entre dimensões — use um teste multivariado (classificador
> de domínio, MMD), não 768 PSIs independentes.

**C33 — Concept drift: P(X) igual, muda P(y|X).** A regra que liga entrada e
rótulo se altera. A entrada parece idêntica — quem denuncia é a **queda de
desempenho**.

| sinal | valor |
| ----- | ----- |
| PSI máximo entre as features de X | 0.012 (estável) |
| acurácia na referência | 1.000 |
| acurácia no período atual | 0.513 |
| queda de acurácia | 0.486 🚨 |

> A lição mais importante de monitoramento: **monitorar só a entrada não pega
> concept drift.** É preciso fechar o loop com os rótulos reais que chegam depois
> e vigiar o desempenho.

### Detectores

- **PSI** (Population Stability Index): univariado, interpretável (< 0.1 estável,
  0.1–0.25 moderado, > 0.25 forte). O padrão de risco de crédito.
- **KS** (Kolmogorov-Smirnov): teste de duas amostras com p-valor, para features
  contínuas.
- **Classificador de domínio**: treina um modelo para separar referência de atual,
  com validação cruzada. AUC ~0.5 = sem drift; AUC alto = drift. A lente
  multivariada certa para embeddings.

### Evidently (camada opcional)

Em produção você não reimplementa PSI — usa uma ferramenta. `src/evidently_report.py`
pega o cenário C31 e gera um relatório HTML interativo do Evidently:

```
python src/evidently_report.py     # gera data_drift_report.html
```

### Execução

```
pip install -r requirements.txt
pytest tests/ -v        # 5 testes
python src/demo.py      # os 3 tipos de drift com números reais (~2s)
```

### Estrutura

```
src/
├── detectors.py          # PSI, KS, classificador de domínio (do zero)
├── scenarios.py          # cenários sintéticos C31, C32, C33
├── demo.py               # roda e imprime os três
└── evidently_report.py   # relatório HTML opcional (Evidently)
tests/
└── test_drift.py         # um invariante por tipo de drift
```

### Limitações honestas

Cenários sintéticos com drift injetado para isolar cada fenômeno. Em produção:
defina janelas de referência e atual com cuidado (sazonalidade vira falso
positivo), corrija para múltiplas comparações ao testar muitas features, e lembre
que detectar drift é só o gatilho — a decisão de re-treinar depende do impacto no
negócio, não só do p-valor.

---

## 🇺🇸 English

### The thesis

Shipping the model is the start of the problem, not the end. But **drift is not one
thing** — there are three types, each needing a different lens. Watch the wrong one
and you get false confidence. This repo implements the detectors **from scratch**
and shows all three on deterministic synthetic scenarios. Runs in ~2 seconds.

### The three types (with real numbers)

- **C31 — Data drift (covariate shift), P(X) changes:** PSI 0.003 → 1.444, KS
  p-value 0.90 → 0.0000. Caught feature by feature.
- **C32 — Embedding drift, multivariate signal:** a subtle shift along one
  direction moves each dimension little (mean per-dim PSI 0.064, still "stable")
  but the joint changes — domain-classifier AUC 0.52 → 0.74. **Don't monitor
  embeddings dimension by dimension.**
- **C33 — Concept drift, P(y|X) changes while P(X) stays:** max PSI on X is 0.012
  (input looks stable) yet accuracy drops 1.000 → 0.513. **Input-only monitoring
  misses concept drift** — you must close the loop with incoming labels.

### Detectors

PSI (univariate, interpretable), KS two-sample test, and a cross-validated **domain
classifier** (the right multivariate lens for embeddings).

### Evidently (optional layer)

In production you don't reimplement PSI — you use a tool. `src/evidently_report.py`
turns the C31 scenario into an interactive Evidently HTML report.

### Running

```
pip install -r requirements.txt
pytest tests/ -v        # 5 tests
python src/demo.py      # the 3 drift types with real numbers (~2s)
```

### Honest limitations

Synthetic, injected drift to isolate each phenomenon. In production: choose
reference/current windows carefully (seasonality causes false positives), correct
for multiple comparisons across many features, and remember detection is just the
trigger — the retrain decision depends on business impact, not the p-value alone.

---

Part of my LinkedIn series on LLMOps → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
