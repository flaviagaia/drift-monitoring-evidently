"""Detectores de drift, do zero — PSI, KS e classificador de domínio.

Três lentes complementares:

- PSI (Population Stability Index): drift univariado, interpretável, o
  padrão de risco de crédito. Bom para uma feature por vez.
- KS (Kolmogorov-Smirnov): teste estatístico de duas amostras para uma
  feature contínua; devolve p-valor.
- Classificador de domínio: a lente MULTIVARIADA. Treina um modelo para
  distinguir "referência" de "atual". Se ele consegue (AUC >> 0.5), as
  distribuições diferem — mesmo quando cada feature isolada parece estável.
  É o que pega drift de EMBEDDINGS, onde o sinal está na correlação entre
  dimensões, não em nenhuma dimensão sozinha.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------- PSI
def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index entre duas amostras 1D.

    Faixas usuais: < 0.1 estável | 0.1–0.25 drift moderado | > 0.25 drift forte.
    """
    reference = np.asarray(reference, dtype=float).ravel()
    current = np.asarray(current, dtype=float).ravel()
    # bordas pelos quantis da referência (bins de igual frequência)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    eps = 1e-6
    ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_label(value: float) -> str:
    if value < 0.1:
        return "estavel"
    if value < 0.25:
        return "moderado"
    return "forte"


# ------------------------------------------------------------------ KS
def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov de duas amostras. Retorna (estatistica, p_valor)."""
    res = ks_2samp(np.asarray(reference).ravel(), np.asarray(current).ravel())
    return float(res.statistic), float(res.pvalue)


# ------------------------------------------------ classificador de domínio
def domain_classifier_auc(
    reference: np.ndarray,
    current: np.ndarray,
    seed: int = 0,
) -> float:
    """AUC de um classificador que tenta separar referência de atual.

    ~0.5 = indistinguíveis (sem drift). Próximo de 1.0 = drift claro.
    Funciona em qualquer dimensão — é a ferramenta certa para embeddings.
    """
    ref = np.atleast_2d(reference)
    cur = np.atleast_2d(current)
    if ref.shape[0] == 1:
        ref = ref.T
    if cur.shape[0] == 1:
        cur = cur.T
    X = np.vstack([ref, cur])
    y = np.concatenate([np.zeros(len(ref)), np.ones(len(cur))])
    clf = LogisticRegression(max_iter=1000)
    # validação cruzada para não superestimar com overfitting
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))
