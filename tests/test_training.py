import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import src.training
from src.training import GREEN_AI_KEYS, model_report, save_metrics, tune


def fake_cv(f1_by_C: dict):
    """Sustituye cross_validate_model: 5 folds con F1-macro fijo según el C del modelo.

    f1_by_C[C] = (media, desviación): los folds valen media + desviación * [-1, 1, 0, -1, 1],
    cuya media y desviación (ddof=1) son exactamente las pedidas.
    """

    def cross_validate(pipeline, X, y, folds):
        mean, std = f1_by_C[pipeline.named_steps["modelo"].C]
        f1 = mean + std * np.array([-1, 1, 0, -1, 1])
        return pd.DataFrame({"fold": range(5), "f1_macro": f1, "recall_riesgo": 0.6, "precision_riesgo": 0.5}), None

    return cross_validate


def test_tune_picks_simplest_value_within_one_standard_error(monkeypatch):
    # Mejor: C=1 con 0.70 ± 0.025 -> umbral 0.675: entra 0.1 (0.69), no 0.01 (0.65).
    f1 = {0.01: (0.65, 0.02), 0.1: (0.69, 0.02), 1.0: (0.70, 0.025)}
    monkeypatch.setattr(src.training, "cross_validate_model", fake_cv(f1))
    best, table, _ = tune(LogisticRegression(), "C", list(f1), None, None, None)
    top = table.loc[table["f1_macro_media"].idxmax()]
    assert top["C"] == 1.0
    assert best == 0.1
    assert table["dentro_1se"].tolist() == [False, True, True]


def test_tune_keeps_best_when_others_are_clearly_worse(monkeypatch):
    f1 = {0.01: (0.60, 0.001), 0.1: (0.65, 0.001), 1.0: (0.70, 0.001)}
    monkeypatch.setattr(src.training, "cross_validate_model", fake_cv(f1))
    best, _, _ = tune(LogisticRegression(), "C", list(f1), None, None, None)
    assert best == 1.0


@pytest.fixture
def report_and_y():
    """Reporte de model_report con una validación cruzada sintética de 300 filas."""
    rng = np.random.default_rng(0)
    n = 300
    y = pd.Series((rng.random(n) < 0.3).astype(int))
    proba = np.clip(0.3 + 0.4 * y + rng.normal(scale=0.2, size=n), 0, 1)
    folds = pd.DataFrame(
        {"fold": range(5), "f1_macro": 0.7, "recall_riesgo": 0.6, "fit_time_s": 0.1, "predict_time_ms_1000": 1.0}
    )
    pipeline = src.training.build_pipeline(LogisticRegression())
    return model_report("baseline", pipeline, {"folds": folds, "proba": proba}, y), y


def test_report_confusion_matrix_matches_metrics(report_and_y):
    report, y = report_and_y
    cm = report["matriz_confusion"]
    assert cm["tp"] + cm["fn"] == y.sum()
    assert sum(cm.values()) == len(y)
    assert cm["tp"] / (cm["tp"] + cm["fn"]) == pytest.approx(report["oof"]["recall_riesgo"])


def test_report_has_the_standard_format(report_and_y):
    """Contrato del formato que comparten #9, #11 y #12.

    Si esta prueba falla, el cambio en model_report debe acordarse con los
    tres modelos antes de ajustar la lista.
    """
    report, _ = report_and_y
    assert set(report) == {
        "modelo", "estimador", "hiperparametros", "fecha", "n_train", "umbral",
        "cv", "oof", "matriz_confusion", "tiempos",
    }  # fmt: skip
    assert report["umbral"] == 0.5
    assert set(report["oof"]) == {"f1_macro", "recall_riesgo", "precision_riesgo", "balanced_accuracy"}
    assert {"f1_macro_media", "f1_macro_std"} <= set(report["cv"])
    assert set(report["matriz_confusion"]) == {"tn", "fp", "fn", "tp"}
    assert set(report["tiempos"]) == {"entrenamiento_s", "inferencia_ms_1000"}
    json.dumps(report)  # serializable tal cual, sin tipos de numpy


def test_save_metrics_keeps_green_ai_values(tmp_path):
    (tmp_path / "baseline.json").write_text(
        json.dumps({"energia_kwh": 1e-5, "emisiones_gco2eq": 0.002, "umbral": 0.3}), encoding="utf-8"
    )
    path = save_metrics({"modelo": "baseline", "umbral": 0.5}, metrics_dir=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["umbral"] == 0.5  # lo nuevo reemplaza a lo viejo...
    assert saved["energia_kwh"] == 1e-5  # ...salvo lo que midió #15
    assert set(GREEN_AI_KEYS) <= saved.keys()
