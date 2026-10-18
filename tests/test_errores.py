import numpy as np
import pandas as pd
import pytest

from src.config import ANCHO_ABSTENCION, MODELO_FINAL
from src.errores import (
    NO_CONCLUYENTE,
    abstention_summary,
    abstention_zone,
    decide,
    error_profile,
    final_params,
    final_threshold,
    gap_table,
    group_gap,
    input_ranges,
    out_of_range,
    restricciones,
    subgroup_table,
    subgroups,
)

N_BOOT = 200


@pytest.fixture
def groups():
    """Dos grupos de 500: en A el modelo detecta 45 de 50 desertores, en B 25 de 50."""
    y = np.r_[[1] * 50 + [0] * 450, [1] * 50 + [0] * 450]
    pred = np.r_[[1] * 45 + [0] * 5 + [1] * 45 + [0] * 405, [1] * 25 + [0] * 25 + [1] * 90 + [0] * 360]
    g = pd.DataFrame({"Grupo": ["A"] * 500 + ["B"] * 500})
    return y, pred, g


def test_subgroup_table_counts_and_rates(groups):
    y, pred, g = groups
    table = subgroup_table(y, pred, g, n_boot=N_BOOT).set_index("nivel")
    assert table.loc["A", "recall_riesgo"] == pytest.approx(0.9)
    assert table.loc["B", "recall_riesgo"] == pytest.approx(0.5)
    assert table.loc["A", "tasa_fp"] == pytest.approx(0.1)
    assert table.loc["B", "tasa_fp"] == pytest.approx(0.2)
    assert table.loc["B", ["tn", "fp", "fn", "tp"]].tolist() == [360, 90, 25, 25]
    assert table.loc["B", "desertores_entre_no_marcados"] == pytest.approx(25 / 385)
    for level in "AB":
        row = table.loc[level]
        assert row["recall_ic_inf"] <= row["recall_riesgo"] <= row["recall_ic_sup"]


def test_group_gap_detects_real_difference_and_not_noise(groups):
    y, pred, g = groups
    a, b = (g["Grupo"] == "A").to_numpy(), (g["Grupo"] == "B").to_numpy()
    gap = group_gap(y, pred, a, b, n_boot=N_BOOT)["recall_riesgo"]
    assert gap["diferencia"] == pytest.approx(0.4)
    assert gap["ic_inf"] > 0
    # Un grupo contra sí mismo: diferencia 0 y el intervalo la contiene.
    same = group_gap(y, pred, a, a, n_boot=N_BOOT)["recall_riesgo"]
    assert same["diferencia"] == 0
    assert same["ic_inf"] <= 0 <= same["ic_sup"]


def test_restricciones_flag_the_low_recall_group_even_if_it_is_the_reference(groups):
    y, pred, g = groups
    # B pasa a ser la referencia (más numerosa) añadiéndole filas sin desertores.
    y = np.r_[y, [0] * 10]
    pred = np.r_[pred, [0] * 10]
    g = pd.concat([g, pd.DataFrame({"Grupo": ["B"] * 10})], ignore_index=True)
    table = subgroup_table(y, pred, g, n_boot=N_BOOT)
    gaps = gap_table(y, pred, g, n_boot=N_BOOT)
    assert gaps["referencia"].iloc[0] == "B"
    rules = restricciones(table, gaps)
    assert [r["nivel"] for r in rules if "escalar" in r["accion"]] == ["B"]


def test_restricciones_flag_small_and_fully_flagged_groups(groups):
    y, pred, g = groups
    table = subgroup_table(y, np.ones_like(pred), g, n_boot=N_BOOT)
    gaps = gap_table(y, np.ones_like(pred), g, n_boot=N_BOOT)
    rules = restricciones(table, gaps, min_dropouts=60)
    conditions = {(r["nivel"], r["accion"].split(":")[0]) for r in rules}
    assert ("A", "la marca no prioriza dentro del grupo") in conditions
    assert ("B", "evidencia insuficiente") in conditions


def test_decide_uses_abstention_zone_around_threshold():
    inf, sup = abstention_zone(0.4, 0.06)
    assert (inf, sup) == pytest.approx((0.34, 0.46))
    result = decide([0.1, 0.34, 0.4, 0.459, 0.46, 0.9], 0.4, 0.06)
    assert result.tolist() == ["sin riesgo", NO_CONCLUYENTE, NO_CONCLUYENTE, NO_CONCLUYENTE, "riesgo", "riesgo"]
    assert abstention_zone(0.02, 0.06)[0] == 0.0


def test_abstention_summary_partitions_all_cases():
    rng = np.random.default_rng(0)
    proba = rng.uniform(size=500)
    y = (rng.uniform(size=500) < proba).astype(int)
    s = abstention_summary(y, proba, 0.5)
    parts = [s[k] for k in ("riesgo", NO_CONCLUYENTE, "sin riesgo")]
    assert sum(p["n"] for p in parts) == 500
    assert sum(p["desertores"] for p in parts) == y.sum()
    assert s[NO_CONCLUYENTE]["fraccion"] == pytest.approx(2 * ANCHO_ABSTENCION, abs=0.05)


def test_out_of_range_flags_new_profiles():
    X_train = pd.DataFrame(
        {"Age at enrollment": [18, 20, 30] * 10, "Course": [1] * 25 + [2] * 5, "Gender": [0, 1, 1] * 10}
    )
    ranges = input_ranges(X_train)
    assert ranges["Age at enrollment"] == {"min": 18, "max": 30}
    assert ranges["Course"] == {"categorias": [1]}  # 2 tiene menos de MIN_FREQUENCY casos

    X_new = pd.DataFrame({"Age at enrollment": [19, 70, 19], "Course": [1, 1, 2], "Gender": [0, 1, 3]})
    reasons = out_of_range(X_new, ranges)
    assert reasons.iloc[0] == []
    assert reasons.iloc[1] == ["Age at enrollment fuera de [18, 30]"]
    assert len(reasons.iloc[2]) == 2


def test_error_profile_has_one_column_per_confusion_cell():
    df = pd.DataFrame({"Age at enrollment": [18, 30, 20, 40], "Gender": [0, 1, 0, 1]})
    profile = error_profile(df, [1, 1, 0, 0], [0, 1, 1, 0], columns=("Age at enrollment", "Gender"))
    assert list(profile.columns) == ["FN", "TP", "FP", "TN"]
    assert profile.loc["Age at enrollment", "FN"] == 18
    assert profile.loc["n"].tolist() == [1, 1, 1, 1]


def test_subgroups_from_raw_columns():
    df = pd.DataFrame(
        {
            "Gender": [1, 0],
            "Age at enrollment": [19, 35],
            "Scholarship holder": [0, 1],
            "Displaced": [1, 0],
            "Nacionality": [1, 41],
            "International": [0, 1],
        }
    )
    g = subgroups(df)
    assert g.loc[0].tolist() == ["Hombre", "17–19", "No", "Sí", "Portuguesa", "No"]
    assert g.loc[1].tolist() == ["Mujer", "35+", "Sí", "No", "Otra", "Sí"]


def test_final_model_matches_selection_of_13():
    params = final_params()
    assert set(params) == {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
    assert 0 < final_threshold() < 1
    assert MODELO_FINAL == "random_forest"
