"""Cada lição de drift vira um invariante testado."""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from detectors import domain_classifier_auc, ks_test, psi, psi_label  # noqa: E402
from scenarios import concept_drift, data_drift, embedding_drift  # noqa: E402


# ----------------------------------------------------------- detectores
def test_psi_zero_para_mesma_distribuicao():
    rng = np.random.default_rng(0)
    a = rng.normal(size=3000)
    b = rng.normal(size=3000)
    assert psi(a, b) < 0.1
    assert psi_label(psi(a, b)) == "estavel"


def test_psi_alto_para_distribuicao_deslocada():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=3000)
    b = rng.normal(2, 1, size=3000)
    assert psi(a, b) > 0.25
    assert psi_label(psi(a, b)) == "forte"


# ------------------------------------------------------------ C31 data drift
def test_c31_data_drift():
    ref, cur_ok, cur_drift = data_drift()
    assert psi(ref, cur_ok) < 0.1
    assert psi(ref, cur_drift) > 0.25
    _, p_ok = ks_test(ref, cur_ok)
    _, p_drift = ks_test(ref, cur_drift)
    assert p_ok > 0.05 and p_drift < 0.01


# ------------------------------------------------------- C32 embedding drift
def test_c32_embedding_drift_eh_multivariado():
    """A lição central: PSI por dimensão fica estável, mas o classificador
    de domínio (multivariado) acusa o drift."""
    ref, cur_ok, cur_drift = embedding_drift()
    psi_medio = np.mean([psi(ref[:, j], cur_drift[:, j]) for j in range(ref.shape[1])])
    auc_ok = domain_classifier_auc(ref, cur_ok)
    auc_drift = domain_classifier_auc(ref, cur_drift)
    assert psi_medio < 0.1            # marginais parecem estáveis
    assert auc_ok < 0.6              # sem drift -> indistinguível
    assert auc_drift > 0.65          # com drift -> detectável só no conjunto


# --------------------------------------------------------- C33 concept drift
def test_c33_concept_drift_invisivel_na_entrada():
    """P(X) estável (PSI baixo) mas a acurácia despenca: só monitorar a
    entrada não pega concept drift."""
    (X_ref, y_ref), (X_cur, y_cur) = concept_drift()
    psi_max = max(psi(X_ref[:, j], X_cur[:, j]) for j in range(X_ref.shape[1]))
    assert psi_max < 0.1             # a entrada parece estável

    clf = LogisticRegression(max_iter=1000).fit(X_ref, y_ref)
    acc_ref = accuracy_score(y_ref, clf.predict(X_ref))
    acc_cur = accuracy_score(y_cur, clf.predict(X_cur))
    assert acc_ref - acc_cur > 0.2   # desempenho cai claramente
