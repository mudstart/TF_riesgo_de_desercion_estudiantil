"""Análisis de errores, subgrupos y condiciones de uso del modelo final (#14).

Analiza el modelo elegido en #13 (decisión 008) con sus predicciones
out-of-fold de train: el test ya se usó una vez y no se vuelve a usar
(decisión 003). Zona de abstención y condiciones de detención en
docs/decisiones/009_abstencion_y_condiciones.md. Desde la raíz:

    uv run python -m src.errores

Salidas: reports/metrics/analisis_errores.json, reports/metrics/subgrupos.csv
y reports/figures/14_*.png. La demo (#17) usa decide, input_ranges,
out_of_range y restricciones.
"""

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ANCHO_ABSTENCION,
    CONFIDENCE,
    FEATURES_BINARIAS,
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    FIGURES_DIR,
    MAX_BRECHA,
    MAX_TASA_MARCADOS,
    METRICS_DIR,
    MIN_DESERTORES_SUBGRUPO,
    MIN_FREQUENCY,
    MODELO_FINAL,
    N_BOOTSTRAP,
    RANDOM_STATE,
)
from src.data import build_target, cv_folds, load_raw, load_splits
from src.evaluation import choose_threshold, classification_metrics, cross_validate_model, paired_bootstrap, predict_at, recall_riesgo
from src.features import select_features
from src.models.train_rf import SEARCH_SPACE, build_rf
from src.training import confusion

RESULT_PATH = METRICS_DIR / "analisis_errores.json"
SUBGROUPS_PATH = METRICS_DIR / "subgrupos.csv"

NO_CONCLUYENTE = "no concluyente"

# Variables que describen el perfil de los errores (columnas del CSV original).
PERFIL = (
    "Age at enrollment",
    "Gender",
    "Admission grade",
    "Previous qualification (grade)",
    "Scholarship holder",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Daytime/evening attendance",
)

# Variantes del análisis de sensibilidad: columnas que el modelo deja de ver.
SIN_SENSIBLES = {
    "sin Gender": ("Gender",),
    "sin Gender ni edad": ("Gender", "Age at enrollment"),
}


def subgroups(df: pd.DataFrame) -> pd.DataFrame:
    """Subgrupos del issue #14, uno por columna, a partir del CSV original.

    Becario (Scholarship holder) no entra al modelo principal (decisión 001),
    pero sirve para describir a quién afecta el error.
    """
    yes_no = {1: "Sí", 0: "No"}
    return pd.DataFrame(
        {
            "Género": df["Gender"].map({1: "Hombre", 0: "Mujer"}),
            "Edad al ingreso": pd.cut(
                df["Age at enrollment"], [0, 19, 24, 34, np.inf], labels=["17–19", "20–24", "25–34", "35+"]
            ).astype(str),
            "Becario": df["Scholarship holder"].map(yes_no),
            "Desplazado": df["Displaced"].map(yes_no),
            "Nacionalidad": np.where(df["Nacionality"] == 1, "Portuguesa", "Otra"),
            "Internacional": df["International"].map(yes_no),
        },
        index=df.index,
    )


def _boot_means(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Medias bootstrap de un vector 0/1 (nan si está vacío)."""
    if len(values) == 0:
        return np.full(n_boot, np.nan)
    return values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)


def _quantiles(boot: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = (1 - confidence) / 2
    return float(np.nanquantile(boot, alpha)), float(np.nanquantile(boot, 1 - alpha))


def subgroup_table(
    y, pred, groups: pd.DataFrame, n_boot: int = N_BOOTSTRAP, confidence: float = CONFIDENCE, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Métricas por subgrupo: una fila por (variable, nivel).

    recall = desertores detectados / desertores del grupo; tasa_fp = no
    desertores marcados / no desertores del grupo; desertores_entre_no_marcados
    = fracción que deserta entre los que el modelo no marca. Intervalos
    bootstrap dentro del grupo.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    rows = []
    for variable in groups:
        levels = groups[variable].to_numpy()
        for level in sorted(pd.unique(levels)):
            mask = levels == level
            yy, pp = y[mask], pred[mask]
            hit, alarm = pp[yy == 1], pp[yy == 0]
            r_inf, r_sup = _quantiles(_boot_means(hit, rng, n_boot), confidence)
            f_inf, f_sup = _quantiles(_boot_means(alarm, rng, n_boot), confidence)
            rows.append(
                {
                    "variable": variable,
                    "nivel": level,
                    "n": int(mask.sum()),
                    "desertores": int(yy.sum()),
                    "tasa_desercion": float(yy.mean()),
                    "tasa_marcados": float(pp.mean()),
                    "recall_riesgo": float(hit.mean()) if len(hit) else np.nan,
                    "recall_ic_inf": r_inf,
                    "recall_ic_sup": r_sup,
                    "tasa_fp": float(alarm.mean()) if len(alarm) else np.nan,
                    "tasa_fp_ic_inf": f_inf,
                    "tasa_fp_ic_sup": f_sup,
                    "precision_riesgo": float(yy[pp == 1].mean()) if pp.any() else np.nan,
                    "desertores_entre_no_marcados": float(yy[pp == 0].mean()) if (pp == 0).any() else np.nan,
                    **confusion(yy, pp),
                }
            )
    return pd.DataFrame(rows)


def group_gap(
    y, pred, mask_a, mask_b, n_boot: int = N_BOOTSTRAP, confidence: float = CONFIDENCE, seed: int = RANDOM_STATE
) -> dict[str, dict[str, float]]:
    """Brecha de recall de riesgo y de tasa de falsos positivos (a - b).

    Bootstrap independiente en cada grupo: si el intervalo contiene 0, la
    brecha no se distingue del ruido.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    mask_a, mask_b = np.asarray(mask_a), np.asarray(mask_b)
    rng = np.random.default_rng(seed)
    result = {}
    for name, cls in (("recall_riesgo", 1), ("tasa_fp", 0)):
        a = pred[mask_a & (y == cls)]
        b = pred[mask_b & (y == cls)]
        diffs = _boot_means(a, rng, n_boot) - _boot_means(b, rng, n_boot)
        inf, sup = _quantiles(diffs, confidence)
        result[name] = {"diferencia": float(a.mean() - b.mean()), "ic_inf": inf, "ic_sup": sup}
    return result


def gap_table(y, pred, groups: pd.DataFrame, max_gap: float = MAX_BRECHA, **bootstrap_kwargs) -> pd.DataFrame:
    """Brechas de cada nivel contra el nivel más numeroso de su variable (referencia).

    Una brecha es relevante si supera max_gap en valor absoluto y su
    intervalo no contiene 0.
    """
    y = np.asarray(y)
    rows = []
    for variable in groups:
        levels = groups[variable].to_numpy()
        reference = groups[variable].value_counts().index[0]
        for level in sorted(pd.unique(levels)):
            if level == reference:
                continue
            gaps = group_gap(y, pred, levels == level, levels == reference, **bootstrap_kwargs)
            for metric, ic in gaps.items():
                excludes_zero = ic["ic_inf"] > 0 or ic["ic_sup"] < 0
                rows.append(
                    {
                        "variable": variable,
                        "nivel": level,
                        "referencia": reference,
                        "metrica": metric,
                        **ic,
                        "ic_excluye_0": excludes_zero,
                        "relevante": bool(excludes_zero and abs(ic["diferencia"]) > max_gap),
                    }
                )
    return pd.DataFrame(rows)


def error_profile(df: pd.DataFrame, y, pred, columns=PERFIL) -> pd.DataFrame:
    """Media de cada variable en cada celda de la matriz de confusión.

    Comparar FN con TP dice qué desertores se le escapan al modelo; FP con TN,
    a qué no desertores marca. En las binarias, la media es una proporción.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    cells = {"FN": (y == 1) & (pred == 0), "TP": (y == 1) & (pred == 1), "FP": (y == 0) & (pred == 1), "TN": (y == 0) & (pred == 0)}
    table = pd.DataFrame({cell: df.loc[mask, list(columns)].mean() for cell, mask in cells.items()})
    table.loc["n"] = [int(m.sum()) for m in cells.values()]
    return table


def abstention_zone(threshold: float, width: float = ANCHO_ABSTENCION) -> tuple[float, float]:
    """Intervalo [inf, sup) de probabilidades cuyo resultado es "no concluyente"."""
    return max(threshold - width, 0.0), min(threshold + width, 1.0)


def decide(proba, threshold: float, width: float = ANCHO_ABSTENCION) -> np.ndarray:
    """Resultado que se reporta: "riesgo", "no concluyente" o "sin riesgo"."""
    proba = np.asarray(proba)
    inf, sup = abstention_zone(threshold, width)
    return np.where(proba >= sup, "riesgo", np.where(proba < inf, "sin riesgo", NO_CONCLUYENTE))


def abstention_summary(y, proba, threshold: float, width: float = ANCHO_ABSTENCION) -> dict:
    """Cuántos casos caen en cada resultado y qué error tendrían."""
    y, proba = np.asarray(y), np.asarray(proba)
    pred = predict_at(proba, threshold)
    result = decide(proba, threshold, width)
    inf, sup = abstention_zone(threshold, width)
    summary = {"zona": [inf, sup], "ancho": width, "umbral": threshold}
    for label in ("riesgo", NO_CONCLUYENTE, "sin riesgo"):
        mask = result == label
        summary[label] = {
            "n": int(mask.sum()),
            "fraccion": float(mask.mean()),
            "desertores": int(y[mask].sum()),
            "fraccion_desertores": float(y[mask].sum() / y.sum()),
            "tasa_desercion": float(y[mask].mean()) if mask.any() else None,
            "error_al_umbral": float((pred[mask] != y[mask]).mean()) if mask.any() else None,
        }
    return summary


def input_ranges(X_train: pd.DataFrame, min_frequency: int = MIN_FREQUENCY) -> dict:
    """Contrato de entrada aprendido de train.

    Numéricas: mínimo y máximo. Nominales: categorías con al menos
    min_frequency casos (las demás el modelo las agrupa como infrecuentes).
    Binarias: {0, 1}.
    """
    ranges = {}
    for c in X_train:
        if c in FEATURES_NUMERICAS:
            ranges[c] = {"min": float(X_train[c].min()), "max": float(X_train[c].max())}
        elif c in FEATURES_NOMINALES:
            counts = X_train[c].value_counts()
            ranges[c] = {"categorias": sorted(int(v) for v in counts[counts >= min_frequency].index)}
        elif c in FEATURES_BINARIAS:
            ranges[c] = {"categorias": [0, 1]}
    return ranges


def out_of_range(X: pd.DataFrame, ranges: dict) -> pd.Series:
    """Motivos por fila para abstenerse por perfil fuera del contrato de entrada.

    Devuelve una lista (vacía si la fila está dentro de lo visto en train).
    """
    reasons = pd.Series([[] for _ in range(len(X))], index=X.index, dtype=object)
    for c, r in ranges.items():
        if c not in X:
            continue
        values = X[c]
        if "min" in r:
            bad = values.isna() | (values < r["min"]) | (values > r["max"])
            message = f"{c} fuera de [{r['min']:g}, {r['max']:g}]"
        else:
            bad = ~values.isin(r["categorias"])
            message = f"{c}: categoría poco frecuente o desconocida"
        for i in X.index[bad]:
            reasons[i] = reasons[i] + [message]
    return reasons


def restricciones(
    table: pd.DataFrame,
    gaps: pd.DataFrame,
    min_dropouts: int = MIN_DESERTORES_SUBGRUPO,
    max_flagged: float = MAX_TASA_MARCADOS,
) -> list[dict]:
    """Subgrupos en los que el uso del modelo queda restringido (decisión 009).

    - Recall de riesgo relevantemente menor que el de otro nivel de la misma
      variable: un "sin riesgo" no basta para descartar; se escala a revisión
      humana.
    - Tasa de marcados > max_flagged: la marca no distingue a nadie del grupo.
    - Menos de min_dropouts desertores: no hay evidencia para evaluarlo.
    """
    rules = []
    # El grupo desfavorecido es el de menor recall, sea o no la referencia;
    # si aparece en varias brechas, se reporta la mayor.
    worst = {}
    for _, g in gaps[(gaps["metrica"] == "recall_riesgo") & gaps["relevante"]].iterrows():
        low, high = (g["nivel"], g["referencia"]) if g["diferencia"] < 0 else (g["referencia"], g["nivel"])
        key = (g["variable"], low)
        if key not in worst or abs(g["diferencia"]) > abs(worst[key][1]):
            worst[key] = (high, g["diferencia"])
    for (variable, level), (other, diff) in worst.items():
        rules.append(
            {
                "variable": variable,
                "nivel": level,
                "condicion": f"recall {abs(diff):.2f} menor que en {other}",
                "accion": "no descartar con 'sin riesgo': escalar a revisión humana",
            }
        )
    for _, row in table.iterrows():
        if row["tasa_marcados"] > max_flagged:
            rules.append(
                {
                    "variable": row["variable"],
                    "nivel": row["nivel"],
                    "condicion": f"marca al {row['tasa_marcados']:.0%} del grupo",
                    "accion": "la marca no prioriza dentro del grupo: no usarla para ordenar",
                }
            )
        if row["desertores"] < min_dropouts:
            rules.append(
                {
                    "variable": row["variable"],
                    "nivel": row["nivel"],
                    "condicion": f"solo {row['desertores']} desertores",
                    "accion": "evidencia insuficiente: tratar sus resultados como no concluyentes",
                }
            )
    return rules


def final_params() -> dict:
    """Hiperparámetros del modelo final guardados por #11 en reports/metrics."""
    path = METRICS_DIR / f"{MODELO_FINAL}.json"
    hp = json.loads(path.read_text(encoding="utf-8"))["hiperparametros"]
    return {p.removeprefix("modelo__"): hp[p.removeprefix("modelo__")] for p in SEARCH_SPACE}


def final_threshold() -> float:
    cv = json.loads((METRICS_DIR / "comparacion_cv.json").read_text(encoding="utf-8"))
    if cv["seleccion"]["modelo"] != MODELO_FINAL:
        raise ValueError(f"MODELO_FINAL ({MODELO_FINAL}) no es el elegido en #13 ({cv['seleccion']['modelo']}).")
    return cv["umbral"][MODELO_FINAL]


def oof_proba(X: pd.DataFrame, y: pd.Series, folds, params: dict, drop: tuple[str, ...] = ()) -> np.ndarray:
    """Probabilidades out-of-fold del modelo final; las columnas de drop se vuelven constantes."""
    X = X.copy()
    for c in drop:
        X[c] = X[c].iloc[0]
    return cross_validate_model(build_rf(**params), X, y, folds)[1]


def plot_subgroups(table: pd.DataFrame, overall: dict[str, float], path):
    """Recall de riesgo y tasa de falsos positivos por subgrupo, con IC 95 %.

    overall tiene los valores globales ("recall_riesgo", "tasa_fp"), que se
    dibujan como línea de referencia.
    """
    labels = table["variable"] + ": " + table["nivel"].astype(str)
    pos = np.arange(len(table))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.35 * len(table) + 1.5), sharey=True)
    panels = (("recall_riesgo", "recall", "Recall de Dropout"), ("tasa_fp", "tasa_fp", "Tasa de falsos positivos"))
    for ax, (metric, ic, title) in zip(axes, panels):
        value = table[metric]
        err = [value - table[f"{ic}_ic_inf"], table[f"{ic}_ic_sup"] - value]
        ax.errorbar(value, pos, xerr=err, fmt="o", capsize=3)
        ax.axvline(overall[metric], color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.02)
        ax.set_title(title)
    axes[0].set_yticks(pos, labels)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_abstention(y, proba, threshold: float, path, width: float = ANCHO_ABSTENCION):
    """Tasa real de deserción por tramo de probabilidad, con la zona de abstención."""
    y, proba = np.asarray(y), np.asarray(proba)
    edges = np.quantile(proba, np.linspace(0, 1, 16))
    bins = np.clip(np.searchsorted(edges, proba, side="right") - 1, 0, len(edges) - 2)
    centers = [proba[bins == b].mean() for b in range(len(edges) - 1)]
    rates = [y[bins == b].mean() for b in range(len(edges) - 1)]
    inf, sup = abstention_zone(threshold, width)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axvspan(inf, sup, color="orange", alpha=0.25, label=f"No concluyente [{inf:.2f}, {sup:.2f})")
    ax.plot(centers, rates, "o-", label="Tasa real de Dropout")
    ax.axhline(y.mean(), color="grey", linestyle="--", linewidth=1, label=f"Tasa base ({y.mean():.2f})")
    ax.axvline(threshold, color="red", linewidth=1, label=f"Umbral {threshold:.3f}")
    ax.set_xlabel("Probabilidad de riesgo (out-of-fold)")
    ax.set_ylabel("Fracción que deserta")
    ax.set_title("Tasa de deserción por tramo de probabilidad (15 tramos de igual tamaño)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def sensitivity(X, y, folds, params, proba_full, threshold_full, groups) -> dict:
    """Modelo sin variables sensibles: desempeño global y brecha por género."""
    pred_full = predict_at(proba_full, threshold_full)
    women, men = (groups["Género"] == "Mujer").to_numpy(), (groups["Género"] == "Hombre").to_numpy()
    result = {}
    for name, drop in SIN_SENSIBLES.items():
        print(f"Sensibilidad: {name}...")
        proba = oof_proba(X, y, folds, params, drop)
        threshold = choose_threshold(y, proba)
        pred = predict_at(proba, threshold)
        result[name] = {
            "umbral": threshold,
            "metricas": classification_metrics(y, pred),
            "recall_vs_completo": paired_bootstrap(y, pred, pred_full, recall_riesgo),
            "brecha_mujer_hombre": group_gap(y, pred, women, men),
        }
    result["completo"] = {
        "umbral": threshold_full,
        "metricas": classification_metrics(y, pred_full),
        "brecha_mujer_hombre": group_gap(y, pred_full, women, men),
    }
    return result


def run() -> dict:
    df, splits = load_raw(), load_splits()
    train = df.iloc[splits["train"]].reset_index(drop=True)
    X, y = select_features(train), build_target(train)
    folds = cv_folds(splits)
    params, threshold = final_params(), final_threshold()

    print(f"Predicciones out-of-fold de {MODELO_FINAL} (umbral {threshold:.3f})...")
    proba = oof_proba(X, y, folds, params)
    pred = predict_at(proba, threshold)

    groups = subgroups(train)
    # International se deduce de Nacionality (decisión 004): si coincide con
    # "Otra" nacionalidad, se omite para no repetir el mismo grupo.
    international_redundant = bool(((groups["Internacional"] == "Sí") == (groups["Nacionalidad"] == "Otra")).all())
    if international_redundant:
        groups = groups.drop(columns="Internacional")
    table = subgroup_table(y, pred, groups)
    gaps = gap_table(y, pred, groups)
    ranges = input_ranges(X)
    reasons = out_of_range(X, ranges)
    rare = reasons.str.len().to_numpy() > 0

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    overall = {"recall_riesgo": float(pred[y == 1].mean()), "tasa_fp": float(pred[y == 0].mean())}
    plot_subgroups(table, overall, FIGURES_DIR / "14_subgrupos.png")
    plot_abstention(y, proba, threshold, FIGURES_DIR / "14_zona_abstencion.png")

    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": MODELO_FINAL,
        "umbral": threshold,
        "datos": "predicciones out-of-fold de train (3,539 filas); test no se usa",
        "parametros": {
            "ancho_abstencion": ANCHO_ABSTENCION,
            "max_brecha": MAX_BRECHA,
            "min_desertores_subgrupo": MIN_DESERTORES_SUBGRUPO,
            "max_tasa_marcados": MAX_TASA_MARCADOS,
            "n_bootstrap": N_BOOTSTRAP,
            "confidence": CONFIDENCE,
        },
        "global": {**overall, "metricas": classification_metrics(y, pred), "matriz_confusion": confusion(y, pred)},
        "perfil_errores": error_profile(train, y, pred).round(4).to_dict(),
        "internacional_igual_a_otra_nacionalidad": international_redundant,
        "brechas": gaps.to_dict(orient="records"),
        "abstencion": abstention_summary(y, proba, threshold),
        "perfil_poco_frecuente": {
            "n": int(rare.sum()),
            "error_al_umbral": float((pred[rare] != y[rare]).mean()),
            "error_resto": float((pred[~rare] != y[~rare]).mean()),
            "motivos": reasons[rare].explode().value_counts().to_dict(),
        },
        "restricciones": restricciones(table, gaps),
        "sensibilidad": sensitivity(X, y, folds, params, proba, threshold, groups),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(SUBGROUPS_PATH, index=False, float_format="%.4f", encoding="utf-8")
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"Resultados en {RESULT_PATH} y {SUBGROUPS_PATH}.")
    return result


if __name__ == "__main__":
    run()
