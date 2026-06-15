"""Camada opcional: o mesmo cenário de drift, mas como relatório do Evidently.

Os detectores em `detectors.py` são "do zero" para mostrar a mecânica. Em
produção você não reimplementa PSI — usa uma ferramenta como o Evidently, que
roda uma bateria de testes e gera um relatório HTML interativo. Este módulo
mostra a ponte: pega o cenário C31 e produz `data_drift_report.html`.

Requer `evidently` (veja requirements.txt). Uso:
    python src/evidently_report.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scenarios import data_drift


def gerar_relatorio(out_path: str = "data_drift_report.html") -> dict:
    """Gera o relatório de data drift do Evidently para o cenário C31."""
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Evidently não está instalado. Rode: pip install evidently"
        ) from e

    ref, _, cur = data_drift()
    # várias features correlacionadas para o relatório ficar interessante
    rng = np.random.default_rng(1)
    ref_df = pd.DataFrame({
        "score_relevancia": ref,
        "tamanho_query": rng.normal(0, 1, len(ref)),
    })
    cur_df = pd.DataFrame({
        "score_relevancia": cur,                       # esta tem drift
        "tamanho_query": rng.normal(0, 1, len(cur)),   # esta não
    })

    dd = DataDefinition(numerical_columns=list(ref_df.columns))
    rd = Dataset.from_pandas(ref_df, data_definition=dd)
    cd = Dataset.from_pandas(cur_df, data_definition=dd)

    snapshot = Report([DataDriftPreset()]).run(reference_data=rd, current_data=cd)
    snapshot.save_html(out_path)
    result = snapshot.dict()
    n_metrics = len(result.get("metrics", []))
    print(f"Relatório salvo em {out_path} ({n_metrics} métricas de drift).")
    print("Abra o HTML no navegador para a visão interativa.")
    return result


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    gerar_relatorio(str(ROOT / "data_drift_report.html"))
