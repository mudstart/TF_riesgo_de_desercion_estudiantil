"""Medición Green AI: costo computacional y energético de los tres modelos (#15).

Mide, en el mismo equipo y en el mismo proceso, el costo de entrenar cada
modelo tal como lo hacen #9, #11 y #12 (búsqueda o ajuste de hiperparámetros
+ entrenamiento final con todo train) y el de usarlo. Criterio en
docs/decisiones/011_medicion_green_ai.md. Desde la raíz:

    uv run python -m src.green_ai

Por modelo:

1. Entrenamiento con codecarbon (OfflineEmissionsTracker, intensidad de
   carbono de República Dominicana), REPETICIONES veces, alternando los
   modelos en cada repetición para que compartan las condiciones del equipo:
   tiempo (s), energía (kWh) y emisiones (gCO2eq).
2. Inferencia: N_LOTES predicciones de 1,000 filas de train con el pipeline
   entrenado: tiempo (ms) y energía por cada 1,000 registros.
3. Tamaño del pipeline serializado con joblib (MB).

Solo usa train y los 5 folds fijos; nunca test. No sobrescribe models/.

Salidas: reports/metrics/green_ai.json y green_ai.md (tablas),
reports/figures/15_desempeno_vs_costo.png, y energia_kwh y emisiones_gco2eq
en reports/metrics/<modelo>.json, que lee la comparación (#13).
"""

import argparse
import io
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from codecarbon import OfflineEmissionsTracker
from codecarbon import __version__ as codecarbon_version

from src.config import (
    BASELINE,
    FIGURES_DIR,
    GRADIENT_BOOSTING,
    METRICS_DIR,
    MODELOS_COMPARADOS,
    RANDOM_FOREST,
    RANDOM_STATE,
)
from src.data import build_target, cv_folds, load_raw, load_splits
from src.features import select_features
from src.models import train_baseline, train_gb, train_rf
from src.training import fit_full_train, tune

# República Dominicana: intensidad de carbono de su red eléctrica (codecarbon).
COUNTRY_ISO = "DOM"
# Repeticiones del entrenamiento completo de cada modelo.
REPETICIONES = 3
# Lotes de 1,000 filas para medir la inferencia.
N_LOTES = 200
TAMANO_LOTE = 1000

GREEN_AI_PATH = METRICS_DIR / "green_ai.json"
GREEN_AI_TABLES_PATH = METRICS_DIR / "green_ai.md"
FIGURE_PATH = FIGURES_DIR / "15_desempeno_vs_costo.png"
CV_PATH = METRICS_DIR / "comparacion_cv.json"

NOMBRES = {BASELINE: "Baseline", RANDOM_FOREST: "Random Forest", GRADIENT_BOOSTING: "Gradient Boosting"}


def train_baseline_full(X, y, folds):
    """Ajuste de C (#9) + entrenamiento final con todo train."""
    estimator = train_baseline.build_baseline().named_steps["modelo"]
    best_C, _, _ = tune(estimator, "C", train_baseline.C_GRID, X, y, folds)
    return fit_full_train(train_baseline.build_baseline(best_C), X, y)[0]


def train_rf_full(X, y, folds):
    """Búsqueda aleatoria (#11) + entrenamiento final con todo train."""
    best, _, _ = train_rf.search(X, y, folds)
    return fit_full_train(train_rf.build_rf(**best), X, y)[0]


def train_gb_full(X, y, folds):
    """Búsqueda aleatoria (#12) + entrenamiento final con todo train."""
    best, _, _ = train_gb.search(X, y, folds)
    return fit_full_train(train_gb.build_gb(**best), X, y)[0]


ENTRENAMIENTOS = {BASELINE: train_baseline_full, RANDOM_FOREST: train_rf_full, GRADIENT_BOOSTING: train_gb_full}


def new_tracker() -> OfflineEmissionsTracker:
    """Tracker sin conexión a internet ni archivo emissions.csv."""
    return OfflineEmissionsTracker(
        country_iso_code=COUNTRY_ISO,
        measure_power_secs=1,
        tracking_mode="process",
        output_methods=[],
        log_level="error",
    )


def measure(fn, *args) -> tuple[object, dict, object]:
    """Ejecuta fn(*args) dentro de un tracker de codecarbon.

    Devuelve el resultado de fn, la medición (tiempo en s, energía en kWh,
    emisiones en gCO2eq, energía de CPU y de RAM) y los datos completos de
    codecarbon (hardware detectado).
    """
    tracker = new_tracker()
    tracker.start()
    start = time.perf_counter()
    try:
        result = fn(*args)
    finally:
        elapsed = time.perf_counter() - start
        emissions_kg = tracker.stop()
    data = tracker.final_emissions_data
    return (
        result,
        {
            "tiempo_s": elapsed,
            "energia_kwh": float(data.energy_consumed),
            "emisiones_gco2eq": float(emissions_kg) * 1000,
            "energia_cpu_kwh": float(data.cpu_energy),
            "energia_ram_kwh": float(data.ram_energy),
        },
        data,
    )


def model_size_mb(pipeline) -> float:
    """Tamaño del pipeline serializado con joblib, como lo guarda save_pipeline (MB)."""
    buffer = io.BytesIO()
    joblib.dump(pipeline, buffer)
    return buffer.getbuffer().nbytes / 1024**2


def inference_cost(pipeline, X: pd.DataFrame, n_lotes: int = N_LOTES, tamano: int = TAMANO_LOTE) -> dict:
    """Tiempo y energía de predecir lotes de `tamano` filas, por cada 1,000 registros."""
    lote = X.sample(n=tamano, replace=len(X) < tamano, random_state=RANDOM_STATE)
    pipeline.predict_proba(lote)  # calentamiento: no se mide

    def run():
        times = []
        for _ in range(n_lotes):
            start = time.perf_counter()
            pipeline.predict_proba(lote)
            times.append(time.perf_counter() - start)
        return np.array(times) * 1000 * (1000 / tamano)

    ms, cost, _ = measure(run)
    return {
        "lotes": n_lotes,
        "tamano_lote": tamano,
        "inferencia_ms_1000_media": float(ms.mean()),
        "inferencia_ms_1000_std": float(ms.std(ddof=1)),
        "energia_kwh_1000": cost["energia_kwh"] / n_lotes * (1000 / tamano),
        "emisiones_gco2eq_1000": cost["emisiones_gco2eq"] / n_lotes * (1000 / tamano),
    }


def summarize(runs: list[dict]) -> dict:
    """Media y desviación entre repeticiones de cada medida."""
    df = pd.DataFrame(runs)
    out = {}
    for col in df.columns:
        out[f"{col}_media"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def hardware(data) -> dict:
    """Equipo y software con que se midió (codecarbon y platform)."""
    return {
        "cpu": data.cpu_model,
        "nucleos_logicos": int(data.cpu_count),
        "ram_gb": round(float(data.ram_total_size), 1),
        "gpu": "ninguna" if not data.gpu_count else f"{data.gpu_count} x {data.gpu_model}",
        "sistema_operativo": data.os,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "codecarbon": codecarbon_version,
        "pais_intensidad_carbono": data.country_name,
        "modo_codecarbon": data.tracking_mode,
        "pue": float(data.pue),
    }


def write_model_keys(name: str, training: dict, metrics_dir: Path = METRICS_DIR) -> Path:
    """Escribe energia_kwh y emisiones_gco2eq (entrenamiento) en reports/metrics/<nombre>.json.

    Son las claves que conserva save_metrics y lee la comparación (#13); el
    resto del archivo no cambia.
    """
    path = Path(metrics_dir) / f"{name}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"modelo": name}
    report["energia_kwh"] = training["energia_kwh_media"]
    report["emisiones_gco2eq"] = training["emisiones_gco2eq_media"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def performance(cv_path: Path = CV_PATH) -> dict:
    """F1-macro y recall de riesgo en CV al umbral ajustado (#13), si existen."""
    if not Path(cv_path).exists():
        return {}
    cv = json.loads(Path(cv_path).read_text(encoding="utf-8"))["cv_umbral_ajustado"]
    keys = ("f1_macro_media", "f1_macro_std", "recall_riesgo_media", "recall_riesgo_std")
    return {name: {k: row[k] for k in keys} for name, row in cv.items()}


def tables(result: dict) -> str:
    """Tablas en Markdown para reports/green_ai.md."""

    def pm(d: dict, key: str, fmt: str) -> str:
        return f"{format(d[key + '_media'], fmt)} ± {format(d[key + '_std'], fmt)}"

    def ratio(a: float, b: float) -> str:
        return f"{a / b:.1f}" if b else "n/d"

    hw = result["hardware"]
    lines = [
        f"Equipo: {hw['cpu']}, {hw['nucleos_logicos']} núcleos lógicos, {hw['ram_gb']} GB de RAM, "
        f"GPU: {hw['gpu']}. {hw['sistema_operativo']}; Python {hw['python']}, scikit-learn "
        f"{hw['scikit_learn']}, codecarbon {hw['codecarbon']} (modo {hw['modo_codecarbon']}, "
        f"intensidad de carbono de {hw['pais_intensidad_carbono']}).",
        "",
        f"Entrenamiento (búsqueda o ajuste de hiperparámetros + modelo final), "
        f"media ± desviación de {result['condiciones']['repeticiones']} repeticiones:",
        "",
        "| Modelo | Tiempo (s) | Energía (kWh) | Emisiones (gCO2eq) | Inferencia (ms/1000) "
        "| Energía inferencia (kWh/1000) | Tamaño (MB) | F1-macro | Recall riesgo |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, m in result["modelos"].items():
        e, inf, perf = m["entrenamiento"], m["inferencia"], m.get("desempeno")
        f1 = pm(perf, "f1_macro", ".3f") if perf else "pendiente (#13)"
        rec = pm(perf, "recall_riesgo", ".3f") if perf else "pendiente (#13)"
        lines.append(
            f"| {name} | {pm(e, 'tiempo_s', '.1f')} | {pm(e, 'energia_kwh', '.2e')} "
            f"| {pm(e, 'emisiones_gco2eq', '.3f')} | {pm(inf, 'inferencia_ms_1000', '.1f')} "
            f"| {inf['energia_kwh_1000']:.2e} | {m['tamano_mb']:.2f} | {f1} | {rec} |"
        )

    base = result["modelos"].get(BASELINE)
    if base:
        lines += ["", "Costo relativo al baseline (veces):", "", "| Modelo | Energía entrenamiento | Inferencia | Tamaño |", "|---|---|---|---|"]
        for name, m in result["modelos"].items():
            lines.append(
                f"| {name} | {ratio(m['entrenamiento']['energia_kwh_media'], base['entrenamiento']['energia_kwh_media'])} "
                f"| {ratio(m['inferencia']['inferencia_ms_1000_media'], base['inferencia']['inferencia_ms_1000_media'])} "
                f"| {ratio(m['tamano_mb'], base['tamano_mb'])} |"
            )
    return "\n".join(lines) + "\n"


def plot_cost(result: dict, path: Path = FIGURE_PATH) -> Path | None:
    """F1-macro y recall de riesgo (CV, umbral ajustado) frente a la energía de entrenamiento."""
    models = {n: m for n, m in result["modelos"].items() if m.get("desempeno")}
    if not models:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("f1_macro", "entrenamiento", "energia_kwh", "Energía de entrenamiento (kWh, log)", True),
        ("recall_riesgo", "entrenamiento", "energia_kwh", "Energía de entrenamiento (kWh, log)", True),
        ("f1_macro", "inferencia", "inferencia_ms_1000", "Inferencia (ms por 1,000 registros)", False),
    ]
    for ax, (metric, phase, cost, xlabel, log) in zip(axes, panels):
        for name, m in models.items():
            x, xerr = m[phase][f"{cost}_media"], m[phase][f"{cost}_std"]
            y, yerr = m["desempeno"][f"{metric}_media"], m["desempeno"][f"{metric}_std"]
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4, markersize=8, label=NOMBRES.get(name, name))
            offset = (6, -14) if name == GRADIENT_BOOSTING else (6, 6)  # evita que se solapen las etiquetas
            ax.annotate(NOMBRES.get(name, name), (x, y), textcoords="offset points", xytext=offset, fontsize=9)
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("F1-macro (CV)" if metric == "f1_macro" else "Recall Dropout (CV)")
        ax.grid(alpha=0.3)
    axes[0].set_title("F1-macro vs. energía")
    axes[1].set_title("Recall de riesgo vs. energía")
    axes[2].set_title("F1-macro vs. tiempo de inferencia")
    fig.suptitle("Desempeño (CV, umbral ajustado) frente a costo; barras = ± 1 desviación")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return Path(path)


def main(repeticiones: int = REPETICIONES, n_lotes: int = N_LOTES) -> dict:
    df, splits = load_raw(), load_splits()
    X, y = select_features(df), build_target(df)
    X_train, y_train = X.iloc[splits["train"]], y.iloc[splits["train"]]
    folds = cv_folds(splits)

    runs = {name: [] for name in MODELOS_COMPARADOS}
    pipelines, data = {}, None
    for r in range(1, repeticiones + 1):
        for name in MODELOS_COMPARADOS:  # alternados: mismas condiciones para los tres
            print(f"Repetición {r}/{repeticiones}: entrenamiento de {name}...")
            pipelines[name], cost, data = measure(ENTRENAMIENTOS[name], X_train, y_train, folds)
            runs[name].append(cost)
            print(f"  {cost['tiempo_s']:.1f} s, {cost['energia_kwh']:.2e} kWh, {cost['emisiones_gco2eq']:.3f} gCO2eq")

    perf = performance()
    modelos = {}
    for name in MODELOS_COMPARADOS:
        print(f"Inferencia y tamaño: {name}...")
        modelos[name] = {
            "entrenamiento": summarize(runs[name]),
            "repeticiones": runs[name],
            "inferencia": inference_cost(pipelines[name], X_train, n_lotes),
            "tamano_mb": model_size_mb(pipelines[name]),
            "desempeno": perf.get(name),
        }

    result = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "hardware": hardware(data),
        "condiciones": {
            "repeticiones": repeticiones,
            "orden": "modelos alternados en cada repetición, en un mismo proceso",
            "entrenamiento": "búsqueda o ajuste de hiperparámetros con los 5 folds fijos + modelo final con todo train",
            "inferencia": f"{n_lotes} lotes de {TAMANO_LOTE} filas de train (predict_proba del pipeline completo)",
            "tracker": f"OfflineEmissionsTracker(country_iso_code='{COUNTRY_ISO}', tracking_mode='process', measure_power_secs=1)",
            "n_train": int(len(y_train)),
        },
        "modelos": modelos,
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    GREEN_AI_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    GREEN_AI_TABLES_PATH.write_text(tables(result), encoding="utf-8")
    for name, m in modelos.items():
        write_model_keys(name, m["entrenamiento"])
    figure = plot_cost(result)

    print(tables(result))
    print(f"Resultados en {GREEN_AI_PATH} y {GREEN_AI_TABLES_PATH}.")
    print(f"Figura en {figure}." if figure else "Sin figura: falta comparacion_cv.json (#13).")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeticiones", type=int, default=REPETICIONES, help="repeticiones del entrenamiento")
    parser.add_argument("--lotes", type=int, default=N_LOTES, help="lotes de 1,000 filas para la inferencia")
    args = parser.parse_args()
    main(args.repeticiones, args.lotes)
