import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import src.training
from src.config import MIN_F1_MACRO
from src.evaluation import classification_metrics, predict_at
from src.training import GREEN_AI_KEYS, model_report, save_metrics, tune


def fake_evaluation(oof_by_C: dict):
    """Sustituye evaluate_pipeline: devuelve métricas fijas según el C del modelo."""

    def evaluate(pipeline, X, y, folds):
        f1, recall = oof_by_C[pipeline.named_steps["modelo"].C]
        return {"umbral": 0.4, "oof": {"f1_macro": f1, "recall_riesgo": recall}}

    return evaluate


def test_tune_picks_highest_recall_above_f1_floor(monkeypatch):
    oof = {0.01: (0.70, 0.60), 0.1: (MIN_F1_MACRO, 0.80), 1.0: (MIN_F1_MACRO - 0.01, 0.95)}
    monkeypatch.setattr(src.training, "evaluate_pipeline", fake_evaluation(oof))
    best, table, _ = tune(LogisticRegression(), "C", list(oof), None, None, None)
    assert best == 0.1  # 1.0 tiene más recall pero no alcanza el piso
    assert len(table) == 3


def test_tune_ties_go_to_first_value_of_grid(monkeypatch):
    oof = {0.01: (0.65, 0.80), 0.1: (0.65, 0.80), 1.0: (0.64, 0.80)}
    monkeypatch.setattr(src.training, "evaluate_pipeline", fake_evaluation(oof))
    best, _, _ = tune(LogisticRegression(), "C", list(oof), None, None, None)
    assert best == 0.01


def test_tune_fails_if_no_value_reaches_floor(monkeypatch):
    oof = {0.1: (MIN_F1_MACRO - 0.1, 0.9)}
    monkeypatch.setattr(src.training, "evaluate_pipeline", fake_evaluation(oof))
    with pytest.raises(ValueError, match="piso|>="):
        tune(LogisticRegression(), "C", list(oof), None, None, None)


@pytest.fixture
def report_and_y():
    """Reporte de model_report con una evaluación sintética de 300 filas."""
    rng = np.random.default_rng(0)
    n = 300
    y = pd.Series((rng.random(n) < 0.3).astype(int))
    proba = np.clip(0.3 + 0.4 * y + rng.normal(scale=0.2, size=n), 0, 1)
    fold = pd.DataFrame({"fold": range(5), "f1_macro": 0.7, "fit_time_s": 0.1, "predict_time_ms_1000": 1.0})
    oof = classification_metrics(y, predict_at(proba, 0.45))
    ev = {"fold_05": fold, "fold_ajustado": fold, "umbral": 0.45, "oof": oof, "proba": proba}
    pipeline = src.training.build_pipeline(LogisticRegression())
    return model_report("baseline", pipeline, ev, y), y


def test_report_confusion_matrix_matches_metrics(report_and_y):
    report, y = report_and_y
    cm = report["matriz_confusion"]["umbral_05"]
    assert cm["tp"] + cm["fn"] == y.sum()
    assert sum(cm.values()) == len(y)
    assert cm["tp"] / (cm["tp"] + cm["fn"]) == pytest.approx(report["oof_umbral_05"]["recall_riesgo"])


def test_report_has_the_standard_format(report_and_y):
    """Contrato del formato que comparten #9, #11 y #12 y que lee #13.

    Si esta prueba falla, el cambio en model_report debe acordarse con los
    tres modelos y con src/comparacion.py antes de ajustar la lista.
    """
    report, _ = report_and_y
    assert set(report) == {
        "modelo", "estimador", "hiperparametros", "fecha", "n_train", "piso_f1_macro", "umbral",
        "cv_umbral_05", "cv_umbral_ajustado", "oof_umbral_05", "oof_umbral_ajustado",
        "matriz_confusion", "tiempos",
    }  # fmt: skip
    metricas = {"f1_macro", "recall_riesgo", "precision_riesgo", "balanced_accuracy"}
    assert set(report["oof_umbral_05"]) == set(report["oof_umbral_ajustado"]) == metricas
    assert set(report["matriz_confusion"]) == {"umbral_05", "umbral_ajustado"}
    assert set(report["matriz_confusion"]["umbral_05"]) == {"tn", "fp", "fn", "tp"}
    assert set(report["tiempos"]) == {"entrenamiento_s", "inferencia_ms_1000"}
    json.dumps(report)  # serializable tal cual, sin tipos de numpy


def test_save_metrics_keeps_green_ai_values(tmp_path):
    (tmp_path / "baseline.json").write_text(
        json.dumps({"energia_kwh": 1e-5, "emisiones_gco2eq": 0.002, "umbral": 0.3}), encoding="utf-8"
    )
    path = save_metrics({"modelo": "baseline", "umbral": 0.4}, metrics_dir=tmp_path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["umbral"] == 0.4  # lo nuevo reemplaza a lo viejo...
    assert saved["energia_kwh"] == 1e-5  # ...salvo lo que midió #15
    assert set(GREEN_AI_KEYS) <= saved.keys()
