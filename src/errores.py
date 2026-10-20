"""Análisis de errores, subgrupos y condiciones de abstención (#14).

Criterio en docs/decisiones/009_abstencion_y_subgrupos.md. Desde la raíz:

    uv run python -m src.errores

Usa el modelo y el umbral elegidos en #13 (reports/metrics/comparacion_cv.json)
y sus probabilidades out-of-fold en train. No usa test (decisión 003): el
test ya se usó una vez en la evaluación final.

Salidas: reports/metrics/analisis_errores.json y analisis_errores.md (tablas)
y reports/figures/14_*.png.

clasificar, dominio_entrenamiento y revisar_entrada implementan las
condiciones de abstención y escalamiento que usa la demo (#17).
"""

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.comparacion import CV_PATH, load_data
from src.config import (
    ANCHO_ABSTENCION,
    CONFIDENCE,
    FEATURES_BINARIAS,
    FEATURES_NOMINALES,
    FEATURES_NUMERICAS,
    FIGURES_DIR,
    LIMITES_EDAD,
    MAX_BRECHA_RECALL,
    METRICS_DIR,
    MIN_FREQUENCY,
    MIN_RIESGO_SUBGRUPO,
    N_BOOTSTRAP,
    RANDOM_STATE,
    SUBGRUPOS,
    TARGET_COL,
)
from src.data import cv_folds, load_raw
from src.evaluation import (
    choose_threshold,
    classification_metrics,
    cross_validate_model,
    paired_bootstrap,
    predict_at,
    recall_riesgo,
)
from src.features import model_columns
from src.models import load_pipeline
from src.training import confusion

RESULT_PATH = METRICS_DIR / "analisis_errores.json"
TABLES_PATH = METRICS_DIR / "analisis_errores.md"

RIESGO, SIN_RIESGO, NO_CONCLUYENTE = "riesgo", "sin riesgo", "no concluyente"

# Diccionario de UCI (docs/ficha_datos.md, sección 3).
CURSOS = {
    33: "Biofuel Production Technologies",
    171: "Animation and Multimedia Design",
    8014: "Social Service (evening)",
    9003: "Agronomy",
    9070: "Communication Design",
    9085: "Veterinary Nursing",
    9119: "Informatics Engineering",
    9130: "Equinculture",
    9147: "Management",
    9238: "Social Service",
    9254: "Tourism",
    9500: "Nursing",
    9556: "Oral Hygiene",
    9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication",
    9853: "Basic Education",
    9991: "Management (evening)",
}

ETIQUETAS = {
    "Gender": {0: "Mujer", 1: "Hombre"},
    "Scholarship holder": {0: "No becario", 1: "Becario"},
    "Displaced": {0: "No desplazado", 1: "Desplazado"},
    "International": {0: "Nacional", 1: "Internacional"},
    "Educational special needs": {0: "Sin necesidades especiales", 1: "Con necesidades especiales"},
    "Course": CURSOS,
}

# Variables con que se describe el perfil de los errores. Debtor y Tuition fees
# up to date no entran al modelo (decisión 001): sirven para ver qué no ve.
PERFIL_NUMERICAS = ("Age at enrollment", "Admission grade", "Previous qualification (grade)", "Application order")
PERFIL_BINARIAS = (
    "Gender",
    "Displaced",
    "Daytime/evening attendance",
    "Scholarship holder",
    "Debtor",
    "Tuition fees up to date",
)

# Anchos de zona de abstención comparados (el elegido es ANCHO_ABSTENCION).
ANCHOS = (0.0, 0.025, 0.05, 0.075, 0.10)

# Variables neutralizadas en la comparación con y sin variables sensibles.
VARIANTES_SENSIBLES = {
    "modelo elegido": (),
    "sin Gender": ("Gender",),
    "sin demográficas sensibles": (
        "Gender",
        "Age at enrollment",
        "Marital Status",
        "Nacionality",
        "Displaced",
        "Educational special needs",
    ),
}


# ---------------------------------------------------------------- subgrupos


def age_band(age, limits: tuple[int, int] = LIMITES_EDAD) -> pd.Series:
    """Rango de edad al ingreso, como categoría ordenada."""
    lo, hi = limits
    labels = [f"Hasta {lo}", f"{lo + 1} a {hi}", f"Más de {hi}"]
    return pd.cut(pd.Series(age), [-np.inf, lo, hi, np.inf], labels=labels)


def subgroup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Una columna por variable de SUBGRUPOS, con etiquetas legibles."""
    out = pd.DataFrame(index=df.index)
    for col in SUBGRUPOS:
        if col == "Rango de edad":
            out[col] = age_band(df["Age at enrollment"])
        elif col == "Estado civil":
            out[col] = np.where(df["Marital Status"] == 1, "Soltero/a", "Otro estado civil")
        else:
            out[col] = df[col].map(ETIQUETAS[col]).fillna(df[col].astype(str))
    return out


def _boot_rates(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Proporción de 1 en n_boot remuestreos bootstrap de values."""
    if len(values) == 0:
        return np.full(n_boot, np.nan)
    return values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)


def _rate(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def subgroup_metrics(
    y_true,
    y_pred,
    groups,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE,
    seed: int = RANDOM_STATE,
    min_riesgo: int = MIN_RIESGO_SUBGRUPO,
    max_brecha: float = MAX_BRECHA_RECALL,
) -> pd.DataFrame:
    """Recall y tasa de falsos positivos por subgrupo, y su brecha con el resto.

    Una fila por valor de groups. Los intervalos son bootstrap al nivel
    confidence; el de cada brecha (subgrupo - resto) remuestrea por separado
    el subgrupo y el resto. Un subgrupo es evaluable si tiene al menos
    min_riesgo desertores, y queda en alerta si es evaluable, su recall está
    más de max_brecha por debajo del resto y el intervalo de la brecha excluye 0.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups = pd.Series(groups).reset_index(drop=True)
    if isinstance(groups.dtype, pd.CategoricalDtype):
        values = [v for v in groups.cat.categories if (groups == v).any()]
    else:
        values = groups.value_counts().index.tolist()
    alpha = (1 - confidence) / 2

    def ic(boots):
        return [float(np.nanquantile(boots, alpha)), float(np.nanquantile(boots, 1 - alpha))]

    rows = []
    for value in values:
        inside = (groups == value).to_numpy()
        rng = np.random.default_rng(seed)
        pos_in, pos_out = y_pred[inside & (y_true == 1)], y_pred[~inside & (y_true == 1)]
        neg_in, neg_out = y_pred[inside & (y_true == 0)], y_pred[~inside & (y_true == 0)]
        b_rec_in, b_rec_out = _boot_rates(pos_in, rng, n_boot), _boot_rates(pos_out, rng, n_boot)
        b_fpr_in, b_fpr_out = _boot_rates(neg_in, rng, n_boot), _boot_rates(neg_out, rng, n_boot)

        marcados = y_pred[inside] == 1
        recall, brecha = _rate(pos_in), _rate(pos_in) - _rate(pos_out)
        brecha_ic = ic(b_rec_in - b_rec_out)
        evaluable = len(pos_in) >= min_riesgo
        rows.append(
            {
                "subgrupo": str(value),
                "n": int(inside.sum()),
                "n_riesgo": int(len(pos_in)),
                "prevalencia": _rate(y_true[inside]),
                "tasa_marcados": _rate(marcados),
                "recall_riesgo": recall,
                "recall_ic": ic(b_rec_in),
                "fpr": _rate(neg_in),
                "fpr_ic": ic(b_fpr_in),
                "precision_riesgo": _rate(y_true[inside][marcados]),
                "brecha_recall": brecha,
                "brecha_recall_ic": brecha_ic,
                "brecha_fpr": _rate(neg_in) - _rate(neg_out),
                "brecha_fpr_ic": ic(b_fpr_in - b_fpr_out),
                "evaluable": bool(evaluable),
                "alerta": bool(evaluable and brecha < -max_brecha and brecha_ic[1] < 0),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------- perfil de errores


def error_type(y_true, y_pred) -> np.ndarray:
    """TP, FN, FP o TN de cada fila."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 1) & (y_pred == 0), (y_true == 0) & (y_pred == 1)],
        ["TP", "FN", "FP"],
        "TN",
    )


def error_profile(df: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    """Media de cada variable de perfil en TP, FN, FP y TN.

    Las binarias son proporciones (0-1). FN se compara con TP (desertores
    detectados o no) y FP con TN (no desertores marcados o no).
    """
    kind = error_type(y_true, y_pred)
    cols = list(PERFIL_NUMERICAS + PERFIL_BINARIAS)
    table = df[cols].groupby(kind).mean().T[["TP", "FN", "FP", "TN"]]
    table["FN - TP"] = table["FN"] - table["TP"]
    table["FP - TN"] = table["FP"] - table["TN"]
    return table


# --------------------------------------------- abstención y escalamiento


def clasificar(proba, threshold: float, ancho: float = ANCHO_ABSTENCION) -> np.ndarray:
    """Resultado para la coordinación: "riesgo", "sin riesgo" o "no concluyente".

    Es "no concluyente" (revisión humana) si la probabilidad está a menos de
    ancho del umbral, a uno u otro lado.
    """
    proba = np.asarray(proba, dtype=float)
    out = np.where(proba >= threshold, RIESGO, SIN_RIESGO).astype(object)
    out[np.abs(proba - threshold) < ancho] = NO_CONCLUYENTE
    return out


def abstention_table(y_true, proba, threshold: float, anchos=ANCHOS) -> pd.DataFrame:
    """Cuántos casos caen en la zona de abstención y qué pasa con los errores.

    Para cada ancho: casos no concluyentes, tasa real de deserción en la zona,
    proporción de los falsos negativos y positivos del umbral que caen en ella,
    y métricas en los casos que sí se deciden.
    """
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = predict_at(proba, threshold)
    kind = error_type(y_true, pred)
    rows = []
    for ancho in anchos:
        zona = clasificar(proba, threshold, ancho) == NO_CONCLUYENTE
        fuera = ~zona
        rows.append(
            {
                "ancho": ancho,
                "zona": [threshold - ancho, threshold + ancho],
                "n_zona": int(zona.sum()),
                "pct_zona": float(zona.mean()),
                "tasa_riesgo_zona": _rate(y_true[zona]),
                "pct_fn_en_zona": float((kind[zona] == "FN").sum() / max((kind == "FN").sum(), 1)),
                "pct_fp_en_zona": float((kind[zona] == "FP").sum() / max((kind == "FP").sum(), 1)),
                **{f"{m}_decididos": v for m, v in classification_metrics(y_true[fuera], pred[fuera]).items()},
            }
        )
    return pd.DataFrame(rows)


def dominio_entrenamiento(X: pd.DataFrame, min_frequency: int = MIN_FREQUENCY) -> dict:
    """Rango de valores visto en train para cada columna del modelo.

    Numéricas: mínimo y máximo. Nominales: códigos frecuentes (al menos
    min_frequency casos, los que el one-hot conserva) y todos los vistos.
    Binarias: valores vistos.
    """
    columns = model_columns()
    counts = {c: X[c].value_counts() for c in columns if c in FEATURES_NOMINALES}
    return {
        "numericas": {
            c: [float(X[c].min()), float(X[c].max())] for c in columns if c in FEATURES_NUMERICAS
        },
        "nominales": {
            c: {
                "frecuentes": sorted(int(v) for v in vc.index[vc >= min_frequency]),
                "vistos": sorted(int(v) for v in vc.index),
            }
            for c, vc in counts.items()
        },
        "binarias": {c: sorted(int(v) for v in X[c].unique()) for c in columns if c in FEATURES_BINARIAS},
    }


def revisar_entrada(X: pd.DataFrame, dominio: dict) -> pd.DataFrame:
    """Condiciones de entrada de la decisión 009 para cada fila de X.

    - abstenerse: falta un valor, una numérica está fuera del rango de train,
      o un código (nominal o binario) no se vio nunca en train. El modelo no
      debe dar resultado.
    - escalar: un código nominal es infrecuente en train (el modelo lo trata
      como "otros"). Se da el resultado, pero con revisión humana.

    motivos explica cada caso, separado por "; ".
    """
    abstain = np.zeros(len(X), dtype=bool)
    escalate = np.zeros(len(X), dtype=bool)
    reasons = [[] for _ in range(len(X))]

    def mark(mask, flag, text):
        for i in np.flatnonzero(mask):
            flag[i] = True
            reasons[i].append(text)

    columns = [*dominio["numericas"], *dominio["nominales"], *dominio["binarias"]]
    for c in columns:
        if c not in X.columns:
            mark(np.ones(len(X), dtype=bool), abstain, f"falta la columna {c}")
            continue
        # Un campo vacío (None) o un texto dejan la columna como object: se
        # convierten a número para poder comparar, y se marcan aparte.
        empty = X[c].isna().to_numpy()
        values = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
        mark(empty, abstain, f"{c} sin valor")
        mark(~empty & np.isnan(values), abstain, f"{c}: valor no numérico")
        missing = np.isnan(values)
        if c in dominio["numericas"]:
            lo, hi = dominio["numericas"][c]
            mark(~missing & ((values < lo) | (values > hi)), abstain, f"{c} fuera del rango de train [{lo:g}, {hi:g}]")
        elif c in dominio["nominales"]:
            seen = np.isin(values, dominio["nominales"][c]["vistos"])
            frequent = np.isin(values, dominio["nominales"][c]["frecuentes"])
            mark(~missing & ~seen, abstain, f"{c}: código no visto en train")
            mark(~missing & seen & ~frequent, escalate, f"{c}: código infrecuente en train")
        else:
            mark(~missing & ~np.isin(values, dominio["binarias"][c]), abstain, f"{c}: valor no visto en train")

    return pd.DataFrame(
        {"abstenerse": abstain, "escalar": escalate & ~abstain, "motivos": ["; ".join(r) for r in reasons]},
        index=X.index,
    )


def domain_check_cv(X: pd.DataFrame, y, pred, folds) -> dict:
    """Simula revisar_entrada en CV: cada fold contra el dominio de los otros cuatro."""
    y, pred = np.asarray(y), np.asarray(pred)
    review = pd.concat(
        [revisar_entrada(X.iloc[va], dominio_entrenamiento(X.iloc[tr])) for tr, va in folds]
    ).reindex(X.index)
    out = {}
    for flag in ("abstenerse", "escalar"):
        mask = review[flag].to_numpy()
        rest = ~review["abstenerse"].to_numpy() & ~review["escalar"].to_numpy()
        out[flag] = {
            "n": int(mask.sum()),
            "pct": float(mask.mean()),
            "recall_riesgo": _rate(pred[mask & (y == 1)]),
            "fpr": _rate(pred[mask & (y == 0)]),
            "recall_riesgo_resto": _rate(pred[rest & (y == 1)]),
            "fpr_resto": _rate(pred[rest & (y == 0)]),
            "motivos_frecuentes": (
                review.loc[mask, "motivos"].str.split("; ").explode().value_counts().head(5).to_dict()
            ),
        }
    return out


# ---------------------------------------------- con y sin variables sensibles


def neutralize(X: pd.DataFrame, columns) -> pd.DataFrame:
    """Copia de X con cada columna fija en su moda: el modelo no puede usarla."""
    X = X.copy()
    for c in columns:
        X[c] = X[c].mode().iloc[0]
    return X


def sensitivity_variants(name: str, X: pd.DataFrame, y: pd.Series, groups: pd.DataFrame, folds) -> dict:
    """CV del modelo elegido con y sin variables sensibles, cada uno con su umbral.

    La primera variante (sin neutralizar) es la referencia del bootstrap
    pareado de la diferencia de recall.
    """
    out, reference = {}, None
    for variant, columns in VARIANTES_SENSIBLES.items():
        print(f"  {variant}...")
        _, proba = cross_validate_model(load_pipeline(name), neutralize(X, columns), y, folds)
        threshold = choose_threshold(y, proba)
        pred = predict_at(proba, threshold)
        reference = pred if reference is None else reference
        out[variant] = {
            "neutralizadas": list(columns),
            "umbral": threshold,
            "metricas_oof": classification_metrics(y, pred),
            "diferencia_recall": paired_bootstrap(y, pred, reference, recall_riesgo),
            "por_subgrupo": {
                col: subgroup_metrics(y, pred, groups[col])[
                    ["subgrupo", "recall_riesgo", "fpr", "brecha_recall", "brecha_recall_ic", "alerta"]
                ].to_dict(orient="records")
                for col in ("Gender", "Rango de edad")
            },
        }
    return out


# ------------------------------------------------------------------ figuras


BLUE, ORANGE = "#2a78d6", "#eb6834"  # categóricas 1 y 2 de la paleta validada


def plot_subgroups(tables: dict[str, pd.DataFrame], recall_global: float, fpr_global: float, path):
    """Recall y tasa de falsos positivos por subgrupo, con IC 95 %."""
    rows = [(col, r) for col, t in tables.items() if col != "Course" for _, r in t.iterrows()]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
    ypos = np.arange(len(rows))[::-1]
    for (_, r), yv in zip(rows, ypos):
        for metric, color, marker, dy in (("recall_riesgo", BLUE, "o", 0.12), ("fpr", ORANGE, "s", -0.12)):
            lo, hi = r[f"{'recall' if metric == 'recall_riesgo' else 'fpr'}_ic"]
            ax.plot([lo, hi], [yv + dy] * 2, color=color, linewidth=2)
            ax.plot(r[metric], yv + dy, marker=marker, color=color, markersize=8, markeredgecolor="white")
    ax.axvline(recall_global, color=BLUE, linestyle="--", linewidth=1)
    ax.axvline(fpr_global, color=ORANGE, linestyle="--", linewidth=1)
    labels = [f"{r['subgrupo']} (n={r['n']}, desert.={r['n_riesgo']})" for _, r in rows]
    ax.set_yticks(ypos, labels, fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Proporción (IC 95 %); líneas discontinuas = valor global")
    ax.plot([], [], "o", color=BLUE, label="Recall de Dropout (desertores detectados)")
    ax.plot([], [], "s", color=ORANGE, label="Tasa de falsos positivos (no desertores marcados)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=8, frameon=False)
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_abstention(y_true, proba, threshold: float, ancho: float, path):
    """Distribución de probabilidades por clase y tasa real de deserción, con la zona."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.linspace(0, 1, 41)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(proba[y_true == 0], bins=bins, histtype="step", linewidth=2, color=ORANGE, label="No desertó")
    ax1.hist(proba[y_true == 1], bins=bins, histtype="step", linewidth=2, color=BLUE, label="Desertó")
    ax1.set_xlabel("Probabilidad de riesgo (out-of-fold)")
    ax1.set_ylabel("Estudiantes")
    ax1.legend(frameon=False)

    centers, rates, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (proba >= lo) & (proba < hi)
        if m.sum() >= 20:
            centers.append((lo + hi) / 2)
            rates.append(y_true[m].mean())
            counts.append(m.sum())
    ax2.plot(centers, rates, "o-", color=BLUE, linewidth=2, markersize=6)
    ax2.axhline(y_true.mean(), color="grey", linestyle="--", linewidth=1, label="Tasa global de deserción")
    ax2.set_xlabel("Probabilidad de riesgo (bins con ≥ 20 casos)")
    ax2.set_ylabel("Tasa real de deserción")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, loc="lower right")

    for ax in (ax1, ax2):
        ax.axvspan(threshold - ancho, threshold + ancho, color="#d9d9d9", alpha=0.6, linewidth=0)
        ax.axvline(threshold, color="black", linewidth=1)
        ax.text(threshold + ancho + 0.01, ax.get_ylim()[1] * 0.92, "zona no\nconcluyente", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------- tablas


def tables_md(result: dict) -> str:
    """Tablas en Markdown para reports/analisis_errores.md."""

    def iv(x):
        return f"[{x[0]:.2f}, {x[1]:.2f}]"

    lines = [
        f"Modelo {result['modelo']}, umbral {result['umbral']:.3f}, predicciones out-of-fold de train "
        f"(n = {result['n_train']}).",
        "",
        "## Métricas por subgrupo",
        "",
        "| Variable | Subgrupo | n | Desertores | Prevalencia | Marcados | Recall (IC 95 %) "
        "| FPR (IC 95 %) | Precisión | Brecha recall vs resto (IC 95 %) | Brecha FPR vs resto | Alerta |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for col, rows in result["subgrupos"].items():
        for r in rows:
            alerta = "**sí**" if r["alerta"] else ("sin evidencia (n < %d)" % MIN_RIESGO_SUBGRUPO if not r["evaluable"] else "no")
            lines.append(
                f"| {col} | {r['subgrupo']} | {r['n']} | {r['n_riesgo']} | {r['prevalencia']:.2f} "
                f"| {r['tasa_marcados']:.2f} | {r['recall_riesgo']:.2f} {iv(r['recall_ic'])} "
                f"| {r['fpr']:.2f} {iv(r['fpr_ic'])} | {r['precision_riesgo']:.2f} "
                f"| {r['brecha_recall']:+.2f} [{r['brecha_recall_ic'][0]:+.2f}, {r['brecha_recall_ic'][1]:+.2f}] "
                f"| {r['brecha_fpr']:+.2f} | {alerta} |"
            )

    lines += ["", "## Perfil de los errores (medias; binarias como proporción)", ""]
    lines += ["| Variable | TP | FN | FP | TN | FN − TP | FP − TN |", "|---|---|---|---|---|---|---|"]
    for r in result["perfil_errores"]:
        lines.append(
            f"| {r['variable']} | {r['TP']:.2f} | {r['FN']:.2f} | {r['FP']:.2f} | {r['TN']:.2f} "
            f"| {r['FN - TP']:+.2f} | {r['FP - TN']:+.2f} |"
        )

    lines += ["", "## Zona de abstención", ""]
    lines += [
        "| Ancho | Zona | Casos en zona | % | Tasa de deserción en zona | % FN en zona | % FP en zona "
        "| Recall decididos | Precisión decididos | F1-macro decididos |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["abstencion"]["tabla"]:
        lines.append(
            f"| ±{r['ancho']:.3f} | [{r['zona'][0]:.3f}, {r['zona'][1]:.3f}] | {r['n_zona']} | {r['pct_zona']:.1%} "
            f"| {r['tasa_riesgo_zona']:.2f} | {r['pct_fn_en_zona']:.1%} | {r['pct_fp_en_zona']:.1%} "
            f"| {r['recall_riesgo_decididos']:.3f} | {r['precision_riesgo_decididos']:.3f} "
            f"| {r['f1_macro_decididos']:.3f} |"
        )

    lines += ["", "## Revisión de la entrada (cada fold contra el dominio de los otros cuatro)", ""]
    lines += ["| Condición | Casos | % | Recall | FPR | Recall del resto | FPR del resto | Motivos más frecuentes |"]
    lines += ["|---|---|---|---|---|---|---|---|"]
    for flag, r in result["dominio_cv"].items():
        motivos = "; ".join(f"{k} ({v})" for k, v in r["motivos_frecuentes"].items())
        lines.append(
            f"| {flag} | {r['n']} | {r['pct']:.1%} | {r['recall_riesgo']:.2f} | {r['fpr']:.2f} "
            f"| {r['recall_riesgo_resto']:.2f} | {r['fpr_resto']:.2f} | {motivos} |"
        )

    lines += ["", "## Con y sin variables sensibles (cada variante con su umbral)", ""]
    lines += [
        "| Variante | Umbral | F1-macro | Recall | Δ recall vs elegido (IC 95 %) | Precisión "
        "| Recall mujer / hombre | Brecha mujer (IC 95 %) | Recall por edad | Brecha hasta 20 (IC 95 %) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for variant, r in result["sensibilidad"].items():
        m, d = r["metricas_oof"], r["diferencia_recall"]
        rows = {col: {g["subgrupo"]: g for g in rs} for col, rs in r["por_subgrupo"].items()}
        women, men = rows["Gender"]["Mujer"], rows["Gender"]["Hombre"]
        young = next(iter(rows["Rango de edad"].values()))
        ages = " / ".join(f"{k}: {g['recall_riesgo']:.2f}" for k, g in rows["Rango de edad"].items())
        lines.append(
            f"| {variant} | {r['umbral']:.3f} | {m['f1_macro']:.3f} | {m['recall_riesgo']:.3f} "
            f"| {d['diferencia']:+.3f} [{d['ic_inf']:+.3f}, {d['ic_sup']:+.3f}] | {m['precision_riesgo']:.3f} "
            f"| {women['recall_riesgo']:.2f} / {men['recall_riesgo']:.2f} | {women['brecha_recall']:+.2f} "
            f"{iv(women['brecha_recall_ic'])} | {ages} | {young['brecha_recall']:+.2f} {iv(young['brecha_recall_ic'])} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- main


def main() -> dict:
    cv = json.loads(CV_PATH.read_text(encoding="utf-8"))
    name = cv["seleccion"]["modelo"]
    threshold = cv["umbral"][name]

    df = load_raw()
    X, y, splits = load_data()
    train = splits["train"]
    raw = df.iloc[train].reset_index(drop=True)
    X_train, y_train = X.iloc[train].reset_index(drop=True), y.iloc[train].reset_index(drop=True)
    folds = cv_folds(splits)

    print(f"1. Probabilidades out-of-fold de {name} (umbral {threshold:.3f})")
    _, proba = cross_validate_model(load_pipeline(name), X_train, y_train, folds)
    pred = predict_at(proba, threshold)
    matrix = confusion(y_train, pred)
    if matrix != cv["matriz_confusion_oof"][name]:
        print(f"  Aviso: la matriz de confusión {matrix} no coincide con la de #13.")

    print("2. Métricas por subgrupo")
    groups = subgroup_columns(raw)
    tables = {col: subgroup_metrics(y_train, pred, groups[col]) for col in SUBGRUPOS}

    print("3. Perfil de los errores")
    profile = error_profile(raw, y_train, pred)
    kind = error_type(y_train, pred)
    fp_target = raw.loc[kind == "FP", TARGET_COL].value_counts().to_dict()
    neg_target = raw.loc[y_train.to_numpy() == 0, TARGET_COL].value_counts().to_dict()

    print("4. Zona de abstención y revisión de la entrada")
    abstention = abstention_table(y_train, proba, threshold)
    domain = domain_check_cv(X_train, y_train, pred, folds)

    print("5. Con y sin variables sensibles")
    sensitivity = sensitivity_variants(name, X_train, y_train, groups, folds)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    m = classification_metrics(y_train, pred)
    fpr_global = matrix["fp"] / (matrix["fp"] + matrix["tn"])
    plot_subgroups(tables, m["recall_riesgo"], fpr_global, FIGURES_DIR / "14_subgrupos.png")
    plot_abstention(y_train, proba, threshold, ANCHO_ABSTENCION, FIGURES_DIR / "14_zona_abstencion.png")

    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "modelo": name,
        "umbral": threshold,
        "n_train": int(len(y_train)),
        "parametros": {
            "ancho_abstencion": ANCHO_ABSTENCION,
            "max_brecha_recall": MAX_BRECHA_RECALL,
            "min_riesgo_subgrupo": MIN_RIESGO_SUBGRUPO,
            "limites_edad": list(LIMITES_EDAD),
            "n_bootstrap": N_BOOTSTRAP,
            "confidence": CONFIDENCE,
        },
        "metricas_oof": m,
        "fpr_oof": fpr_global,
        "matriz_confusion_oof": matrix,
        "falsos_positivos_por_clase": {"falsos_positivos": fp_target, "no_desertores": neg_target},
        "subgrupos": {col: t.to_dict(orient="records") for col, t in tables.items()},
        "alertas": [
            {"variable": col, "subgrupo": r["subgrupo"], "recall_riesgo": r["recall_riesgo"], "brecha_recall": r["brecha_recall"]}
            for col, t in tables.items()
            for _, r in t[t["alerta"]].iterrows()
        ],
        "perfil_errores": profile.round(4).rename_axis("variable").reset_index().to_dict(orient="records"),
        "abstencion": {
            "ancho": ANCHO_ABSTENCION,
            "zona": [threshold - ANCHO_ABSTENCION, threshold + ANCHO_ABSTENCION],
            "tabla": abstention.to_dict(orient="records"),
        },
        "dominio_cv": domain,
        "sensibilidad": sensitivity,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    TABLES_PATH.write_text(tables_md(result), encoding="utf-8")
    print(f"Subgrupos en alerta: {', '.join(a['subgrupo'] for a in result['alertas']) or 'ninguno'}.")
    print(f"Resultados en {RESULT_PATH} y {TABLES_PATH}.")
    return result


if __name__ == "__main__":
    main()
