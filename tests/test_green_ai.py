import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.green_ai import inference_cost, measure, model_size_mb, summarize, write_model_keys
from src.training import GREEN_AI_KEYS, save_metrics


def test_measure_returns_result_and_positive_cost():
    result, cost, data = measure(sum, [1, 2, 3])
    assert result == 6
    assert cost["tiempo_s"] >= 0
    assert cost["energia_kwh"] >= 0
    assert cost["emisiones_gco2eq"] >= 0
    assert data.country_iso_code == "DOM"


def test_summarize_gives_mean_and_std_per_measure():
    out = summarize([{"tiempo_s": 1.0, "energia_kwh": 2.0}, {"tiempo_s": 3.0, "energia_kwh": 4.0}])
    assert out["tiempo_s_media"] == 2.0
    assert out["energia_kwh_media"] == 3.0
    assert out["tiempo_s_std"] == pytest.approx(np.sqrt(2))


def test_write_model_keys_keeps_the_rest_of_the_report(tmp_path):
    (tmp_path / "baseline.json").write_text(json.dumps({"modelo": "baseline", "cv": {"f1_macro_media": 0.6}}))
    write_model_keys("baseline", {"energia_kwh_media": 1e-5, "emisiones_gco2eq_media": 0.006}, tmp_path)
    report = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert report["cv"] == {"f1_macro_media": 0.6}
    assert report["energia_kwh"] == 1e-5
    assert report["emisiones_gco2eq"] == 0.006


def test_retraining_keeps_green_ai_keys(tmp_path):
    write_model_keys("baseline", {"energia_kwh_media": 1e-5, "emisiones_gco2eq_media": 0.006}, tmp_path)
    save_metrics({"modelo": "baseline", "cv": {}}, tmp_path)
    report = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))
    assert {k: report[k] for k in GREEN_AI_KEYS} == {"energia_kwh": 1e-5, "emisiones_gco2eq": 0.006}


@pytest.fixture
def fitted():
    X = pd.DataFrame({"a": np.arange(50.0)})
    y = pd.Series([0, 1] * 25)
    return DummyClassifier(strategy="prior").fit(X, y), X


def test_model_size_is_positive(fitted):
    model, _ = fitted
    assert model_size_mb(model) > 0


def test_inference_cost_is_per_1000_records(fitted):
    model, X = fitted
    out = inference_cost(model, X, n_lotes=3, tamano=100)
    assert out["lotes"] == 3
    assert out["inferencia_ms_1000_media"] > 0
    assert out["energia_kwh_1000"] >= 0
