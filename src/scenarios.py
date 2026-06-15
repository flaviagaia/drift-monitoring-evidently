"""Cenários sintéticos de drift, todos determinísticos (seed fixa).

Três tipos, que são exatamente os três posts:

- C31 data drift (covariate shift): muda P(X). A feature de entrada desloca.
- C32 embedding drift: chega um tópico NOVO; o vetor de embedding muda de
  região. Nenhuma dimensão sozinha denuncia — o sinal é multivariado.
- C33 concept drift: P(X) fica IGUAL, mas P(y|X) muda. A regra que liga
  entrada e rótulo se altera. É o drift que monitorar só a entrada NÃO pega.
"""

from __future__ import annotations

import numpy as np


# ------------------------------------------------- C31: data / covariate drift
def data_drift(n: int = 2000, shift: float = 1.5, seed: int = 0):
    """Feature contínua. 'current' tem média deslocada e variância maior."""
    rng = np.random.default_rng(seed)
    reference = rng.normal(0.0, 1.0, size=n)
    current_no_drift = rng.normal(0.0, 1.0, size=n)
    current_drift = rng.normal(shift, 1.4, size=n)
    return reference, current_no_drift, current_drift


# ----------------------------------------------------- C32: embedding drift
def embedding_drift(dim: int = 16, n: int = 1500, shift: float = 0.85, seed: int = 0):
    """Embeddings de baixa dimensão. O drift é um deslocamento SUTIL numa
    direção do espaço, espalhado por TODAS as dimensões (ex.: uma mudança de
    tom/estilo no conteúdo, ou um tópico novo que move o vetor inteiro um pouco).

    Por construção, cada dimensão isolada se desloca pouco (PSI por dimensão
    fica na faixa estável), mas o deslocamento conjunto é grande — então só a
    visão MULTIVARIADA (classificador de domínio) acusa. É a razão de não se
    monitorar embedding feature a feature."""
    rng = np.random.default_rng(seed)
    reference = rng.normal(0, 1, size=(n, dim))
    current_no_drift = rng.normal(0, 1, size=(n, dim))
    # direção unitária aleatória; o deslocamento por dimensão (~shift/sqrt(dim))
    # é pequeno, mas a separação conjunta (~shift) é detectável
    direction = rng.normal(0, 1, size=dim)
    direction /= np.linalg.norm(direction)
    current_drift = rng.normal(0, 1, size=(n, dim)) + shift * direction
    return reference, current_no_drift, current_drift


# -------------------------------------------------------- C33: concept drift
def concept_drift(n: int = 2000, seed: int = 0):
    """X igual nos dois períodos; muda a REGRA y = f(X).

    Referência: y = 1 se x0 + x1 > 0.
    Atual (concept drift): a regra gira — y = 1 se x0 - x1 > 0.
    P(X) é idêntico; só P(y|X) muda. PSI/KS em X não acusam nada.
    """
    rng = np.random.default_rng(seed)

    def make(rule, m, r):
        X = r.normal(0, 1, size=(m, 4))
        y = (rule(X) > 0).astype(int)
        return X, y

    r_ref = np.random.default_rng(seed)
    r_cur = np.random.default_rng(seed + 100)
    X_ref, y_ref = make(lambda X: X[:, 0] + X[:, 1], n, r_ref)
    X_cur, y_cur = make(lambda X: X[:, 0] - X[:, 1], n, r_cur)
    return (X_ref, y_ref), (X_cur, y_cur)
