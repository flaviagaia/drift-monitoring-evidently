"""Demo executável — roda os três tipos de drift com números reais (~2s).

    python src/demo.py
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from detectors import domain_classifier_auc, ks_test, psi, psi_label
from scenarios import concept_drift, data_drift, embedding_drift


def secao_data_drift() -> None:
    print("\n" + "=" * 64)
    print("C31 — DATA DRIFT (covariate shift): muda P(X)")
    print("=" * 64)
    ref, cur_ok, cur_drift = data_drift()
    for nome, cur in [("sem drift", cur_ok), ("com drift", cur_drift)]:
        p = psi(ref, cur)
        stat, pval = ks_test(ref, cur)
        alarme = "🚨 DRIFT" if (p >= 0.25 or pval < 0.01) else "ok"
        print(f"  {nome:>10} | PSI={p:5.3f} ({psi_label(p)}) | "
              f"KS={stat:4.2f} p={pval:7.4f} | {alarme}")


def secao_embedding_drift() -> None:
    print("\n" + "=" * 64)
    print("C32 — EMBEDDING DRIFT: deslocamento sutil e multivariado")
    print("=" * 64)
    ref, cur_ok, cur_drift = embedding_drift()
    for nome, cur in [("sem drift", cur_ok), ("com drift", cur_drift)]:
        # PSI médio por dimensão: quase não acusa (cada dim parece estável)
        psis = [psi(ref[:, j], cur[:, j]) for j in range(ref.shape[1])]
        auc = domain_classifier_auc(ref, cur)
        alarme = "🚨 DRIFT" if auc > 0.65 else "ok"
        print(f"  {nome:>10} | PSI médio/dim={np.mean(psis):5.3f} | "
              f"AUC classificador de domínio={auc:5.3f} | {alarme}")
    print("  Lição: a média de PSI por dimensão mal se mexe; o classificador")
    print("  de domínio é quem enxerga o tópico novo.")


def secao_concept_drift() -> None:
    print("\n" + "=" * 64)
    print("C33 — CONCEPT DRIFT: P(X) igual, muda P(y|X)")
    print("=" * 64)
    (X_ref, y_ref), (X_cur, y_cur) = concept_drift()

    # 1) drift de entrada? Olhamos PSI em cada feature de X.
    psis = [psi(X_ref[:, j], X_cur[:, j]) for j in range(X_ref.shape[1])]
    print(f"  PSI máximo entre as features de X: {max(psis):.3f} "
          f"({psi_label(max(psis))}) -> a ENTRADA parece estável")

    # 2) modelo treinado na referência: a acurácia despenca no período atual
    clf = LogisticRegression(max_iter=1000).fit(X_ref, y_ref)
    acc_ref = accuracy_score(y_ref, clf.predict(X_ref))
    acc_cur = accuracy_score(y_cur, clf.predict(X_cur))
    queda = acc_ref - acc_cur
    alarme = "🚨 CONCEPT DRIFT" if queda > 0.15 else "ok"
    print(f"  Acurácia na referência: {acc_ref:5.3f}")
    print(f"  Acurácia no atual     : {acc_cur:5.3f}  (queda {queda:+.3f}) | {alarme}")
    print("  Lição: monitorar só a entrada NÃO pega concept drift — é preciso")
    print("  monitorar o DESEMPENHO contra rótulos que chegam depois.")


def main() -> None:
    secao_data_drift()
    secao_embedding_drift()
    secao_concept_drift()


if __name__ == "__main__":
    main()
