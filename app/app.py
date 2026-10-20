"""Demo de predicción del riesgo de deserción al ingreso (#17).

Desde la raíz del repositorio:

    uv run streamlit run app/app.py

Solo interfaz: la lógica está en src/predict.py y el criterio en
docs/decisiones/012_demo_prediccion.md.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # para importar src/

from src.config import FEATURES_NOMINALES, FEATURES_NUMERICAS  # noqa: E402
from src.diccionario import etiqueta  # noqa: E402
from src.predict import (  # noqa: E402
    ADVERTENCIAS_USO,
    AVISO,
    NO_PERMITE_CONCLUIR,
    PERFILES,
    cargar_modelo,
    predecir,
)

# Nombre en español de cada variable del formulario, agrupadas como se muestran.
GRUPOS = {
    "Datos personales": {
        "Gender": "Género",
        "Age at enrollment": "Edad al ingreso",
        "Marital Status": "Estado civil",
        "Nacionality": "Nacionalidad",
        "Displaced": "Desplazado (vive fuera de su residencia habitual)",
        "Educational special needs": "Necesidades educativas especiales",
    },
    "Ingreso a la carrera": {
        "Course": "Carrera",
        "Application mode": "Modo de acceso",
        "Application order": "Orden de preferencia de la carrera (0 = primera opción)",
        "Previous qualification": "Formación previa",
        "Previous qualification (grade)": "Nota de la formación previa (0–200)",
        "Admission grade": "Nota de admisión (0–200)",
    },
    "Familia": {
        "Mother's qualification": "Formación de la madre",
        "Father's qualification": "Formación del padre",
        "Mother's occupation": "Ocupación de la madre",
        "Father's occupation": "Ocupación del padre",
    },
}
NOMBRES = {c: n for grupo in GRUPOS.values() for c, n in grupo.items()}
PROPIO = "Ingresar datos propios"

st.set_page_config(page_title="Riesgo de deserción", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Entrenando el modelo con los hiperparámetros versionados...")
def modelo():
    return cargar_modelo()


def motivo_legible(motivo: str) -> str:
    """Motivo de revisar_entrada con el nombre de la variable en español."""
    for variable in sorted(NOMBRES, key=len, reverse=True):
        if motivo.startswith(variable):
            return NOMBRES[variable] + motivo[len(variable) :]
    return motivo


def valor_legible(variable: str, valor) -> str:
    if variable in FEATURES_NUMERICAS:
        return f"{float(valor):g}"
    return etiqueta(variable, valor)


def campo(m, variable: str, defecto, clave: str):
    """Widget de una variable: lista de códigos vistos en train o número libre.

    Los números no se limitan al rango de train a propósito: un valor fuera
    de rango debe llegar a la validación y producir una abstención visible.
    """
    nombre = NOMBRES[variable]
    if variable in FEATURES_NUMERICAS:
        return st.number_input(nombre, min_value=0.0, max_value=1000.0, value=float(defecto), step=0.1, key=clave)
    if variable in FEATURES_NOMINALES:
        opciones = m.dominio["nominales"][variable]["vistos"]
    else:
        opciones = m.dominio["binarias"][variable]
    indice = opciones.index(defecto) if defecto in opciones else 0
    return st.selectbox(
        nombre, opciones, index=indice, format_func=lambda c: etiqueta(variable, c), key=clave
    )


m = modelo()
t = m.trazabilidad

st.title("Riesgo de deserción al ingreso")
st.caption(
    "Demo del modelo elegido en el proyecto (Random Forest, #13) para estudiantes que acaban de ingresar. "
    "Usa solo datos disponibles al ingreso."
)
st.warning(f"**{AVISO}**", icon="⚠️")

# ------------------------------------------------------------- barra lateral

with st.sidebar:
    st.header("Perfil")
    elegido = st.radio(
        "Cargar un perfil de ejemplo (ficticio, no tomado del dataset) o ingresar datos propios:",
        [*PERFILES, PROPIO],
    )
    st.divider()
    st.header("Trazabilidad del modelo")
    st.markdown(
        f"""
- **Modelo:** {t['modelo']} · versión `{t['version']}`
- **Entrenado:** {t['fecha_entrenamiento']} (búsqueda de hiperparámetros, #11); reconstruido con todo train ({t['n_train']:,} estudiantes) al abrir la demo
- **Dataset:** UCI 697, SHA-256 `{t['dataset_sha256'][:16]}…` (#3)
- **Umbral:** {t['umbral']:.3f} (decisión 008)
- **Zona no concluyente:** [{t['zona_no_concluyente'][0]:.3f}, {t['zona_no_concluyente'][1]:.3f}] (decisión 009)
- **Evaluado en test:** {t['fecha_test']}, una sola vez (#13)
"""
    )
    st.caption("Criterios: docs/decisiones/012_demo_prediccion.md")

# --------------------------------------------------------------- formulario

base = PERFILES.get(elegido, m.tipicos)
with st.form("perfil"):
    st.subheader("Datos del estudiante al ingreso")
    if elegido == PROPIO:
        st.caption("El formulario empieza con los valores más frecuentes en train; cámbielos libremente.")
    columnas = st.columns(len(GRUPOS))
    perfil = {}
    for col, (titulo, variables) in zip(columnas, GRUPOS.items()):
        with col:
            st.markdown(f"**{titulo}**")
            for variable in variables:
                perfil[variable] = campo(m, variable, base[variable], f"{variable}|{elegido}")
    enviado = st.form_submit_button("Calcular riesgo", type="primary")

if not enviado:
    st.info("Elija un perfil o complete el formulario y pulse **Calcular riesgo**.")
    st.stop()

# ---------------------------------------------------------------- resultado

r = predecir(m, perfil)
st.subheader("Resultado")

if r["estado"] == "abstencion":
    st.error("**Abstención: el modelo no da resultado para este perfil.**", icon="⛔")
    st.markdown("El perfil está fuera de lo que el modelo vio al entrenar, así que su predicción no sería fiable:")
    for motivo in r["motivos"]:
        st.markdown(f"- {motivo_legible(motivo)}")
    st.markdown("Revise los datos. Si son correctos, el caso debe valorarse sin el modelo.")
else:
    zona = t["zona_no_concluyente"]
    c1, c2 = st.columns([1, 2])
    c1.metric("Probabilidad de deserción", f"{r['probabilidad']:.0%}")
    c1.progress(min(r["probabilidad"], 1.0))
    mensaje = f"**Nivel {r['categoria'].lower()}: {r['texto']}**"
    if r["categoria"] == "Alto":
        c2.error(mensaje, icon="🔴")
    elif r["categoria"] == "Medio":
        c2.warning(mensaje, icon="🟡")
    else:
        c2.success(mensaje, icon="🟢")
    c2.markdown(
        f"Bajo: menos de {zona[0]:.0%} · **No concluyente: de {zona[0]:.0%} a {zona[1]:.0%}** "
        f"(cerca del umbral {t['umbral']:.0%}) · Alto: {zona[1]:.0%} o más."
    )
    if r["categoria"] == "Medio":
        c2.markdown(
            "En esta franja la tasa real de deserción (24 %) queda por debajo de la global (32 %): marcar o "
            "descartar a estos estudiantes no es mejor que elegir al azar, así que el caso debe revisarlo "
            "una persona (decisión 009)."
        )
    if r["escalar"]:
        st.warning(
            "**Revisión humana:** "
            + "; ".join(motivo_legible(x) for x in r["motivos"])
            + ". El modelo trata ese código como 'otros'.",
            icon="👁️",
        )
    if r["alertas"]:
        texto = "; ".join(f"{a['subgrupo']} (detecta {a['recall_riesgo']:.0%} de los desertores)" for a in r["alertas"])
        if r["e3"]:
            st.warning(
                f"**No usar este 'sin riesgo' para excluir del acompañamiento.** El perfil pertenece a grupos donde "
                f"el modelo detecta menos desertores: {texto}.",
                icon="⚠️",
            )
        else:
            st.caption(f"Grupos del perfil donde el modelo detecta menos desertores: {texto}.")

    st.markdown("#### Qué pesa en este caso")
    f = r["factores"]
    if f.empty:
        st.markdown("El perfil coincide con los valores típicos de train: no hay factores que destacar.")
    else:
        tabla = pd.DataFrame(
            {
                "Variable": [NOMBRES[v] for v in f["variable"]],
                "Valor del perfil": [valor_legible(v, x) for v, x in zip(f["variable"], f["valor"])],
                "Valor típico en train": [valor_legible(v, x) for v, x in zip(f["variable"], f["tipico"])],
                "Efecto en la probabilidad": [f"{e:+.1%}" for e in f["efecto"]],
            }
        ).head(6)
        st.dataframe(tabla, hide_index=True, width="stretch")
        st.caption(
            "Efecto = probabilidad del perfil menos la que tendría si esa variable tomara su valor típico. "
            "Muestra a qué es sensible el modelo, **no** las causas de la deserción."
        )

    st.markdown("#### Qué NO permite concluir este resultado")
    for linea in NO_PERMITE_CONCLUIR:
        st.markdown(f"- {linea}")

# ------------------------------------------------------- contexto del modelo

st.divider()
d = m.desempeno
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Desempeño conocido (test, evaluado una vez)")
    mt, ic = d["metricas_test"], d["ic_test"]
    st.markdown(
        f"""
- Detecta **{d['detecta_de_cada_10']} de cada 10** estudiantes que desertan
  (recall {mt['recall_riesgo']:.2f}, IC 95 % [{ic['recall_riesgo'][0]:.2f}, {ic['recall_riesgo'][1]:.2f}]).
- De cada 10 marcados como riesgo, unos **{round(mt['precision_riesgo'] * 10)}** desertan
  (precisión {mt['precision_riesgo']:.2f}): el resto recibe un acompañamiento que quizá no necesitaba.
- F1-macro {mt['f1_macro']:.2f} en {d['n_test']} estudiantes de test.
"""
    )
    peores = ", ".join(f"{a['subgrupo']} ({a['recall']:.0%})" for a in d["falla_mas_en"][:5])
    st.markdown(f"**Falla más en:** {peores} (desertores detectados en cada grupo; #14).")
with c2:
    st.markdown("#### Variables más influyentes en general")
    top = m.importancia.head(5)
    st.dataframe(
        pd.DataFrame(
            {"Variable": [NOMBRES.get(v, v) for v in top["variable"]], "Importancia": top["importancia_media"].round(3)}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("Permutation importance del Random Forest (#11): cuánto cae el F1-macro al desordenar cada variable.")

st.markdown("#### Cuándo no usar el modelo")
for linea in ADVERTENCIAS_USO:
    st.markdown(f"- {linea}")
