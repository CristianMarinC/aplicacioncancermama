"""
ACA N° 1 – Aprendizaje Automático (CUN)
Detección asistida de cáncer de mama con ML clásico (SVM) – App Streamlit
Dataset: UCI Breast Cancer Wisconsin (Diagnostic) – Wolberg, Street & Mangasarian (1995)
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="Detección asistida de cáncer de mama", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------ carga de artefactos
@st.cache_resource
def cargar_modelo():
    return joblib.load(BASE / "modelo_cancer.joblib")


@st.cache_data
def cargar_info():
    with open(BASE / "info_modelo.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_ejemplos():
    ruta = BASE / "ejemplos.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


try:
    modelo = cargar_modelo()
    info = cargar_info()
except FileNotFoundError:
    st.error("Faltan archivos del modelo. Coloca `modelo_cancer.joblib` e `info_modelo.json` "
             "(generados por el notebook) en la misma carpeta que `app.py`.")
    st.stop()

FEATURES = info["features"]
RANGOS = info["rangos"]
METRICAS = info["metricas"]
ejemplos = cargar_ejemplos()

# Nombres en español para la interfaz
BASE_ES = {
    "radius": "Radio", "texture": "Textura", "perimeter": "Perímetro", "area": "Área",
    "smoothness": "Suavidad", "compactness": "Compacidad", "concavity": "Concavidad",
    "concave points": "Puntos cóncavos", "symmetry": "Simetría", "fractal dimension": "Dimensión fractal",
}
GRUPOS = {
    "mean": ("📏 Valores promedio (mean)", "mean {}"),
    "error": ("📐 Error estándar (error)", "{} error"),
    "worst": ("⚠️ Peores valores (worst)", "worst {}"),
}


def etiqueta(feature: str) -> str:
    for b, es in BASE_ES.items():
        if feature in (f"mean {b}", f"{b} error", f"worst {b}"):
            return es
    return feature


# ------------------------------------------------------------------ estado inicial de sliders
def valores_por_defecto():
    return {f: RANGOS[f]["mean"] for f in FEATURES}


def cargar_en_sliders(valores: dict):
    for f in FEATURES:
        st.session_state[f"sl_{f}"] = float(valores[f])


def cb_promedio():
    cargar_en_sliders(valores_por_defecto())


def cb_ejemplo(clase: int):
    if ejemplos is None:
        return
    fila = ejemplos[ejemplos["diagnostico"] == clase].iloc[0]
    cargar_en_sliders({f: fila[f] for f in FEATURES})


for f in FEATURES:
    st.session_state.setdefault(f"sl_{f}", float(RANGOS[f]["mean"]))

# ------------------------------------------------------------------ utilidades de predicción
def predecir(X: pd.DataFrame, umbral: float):
    proba = modelo.predict_proba(X[FEATURES])[:, 1]  # 1 = maligno
    pred = (proba >= umbral).astype(int)
    return proba, pred


# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.title("🩺 Menú")
    st.markdown("**Modelo:** SVM (Máquina de Vectores de Soporte)")
    st.markdown("**Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) · 569 casos · 30 variables")
    st.divider()
    umbral = st.slider(
        "Umbral de alerta (probabilidad de malignidad)", 0.10, 0.90, 0.50, 0.05,
        help="Si la probabilidad de malignidad supera este umbral, el caso se marca como MALIGNO. "
             "Bajarlo detecta más casos malignos (más recall) pero genera más falsas alarmas.")
    st.caption("Valor por defecto: 0.50. En medicina se suele preferir un umbral bajo para no dejar escapar casos malignos.")
    st.divider()
    st.warning("Herramienta académica de apoyo. **No reemplaza** el diagnóstico de un profesional de la salud.")

# ------------------------------------------------------------------ encabezado
st.title("🩺 Detección asistida de cáncer de mama")
st.markdown("Clasificación de tumores como **benignos** o **malignos** a partir de características del núcleo celular "
            "(biopsia por aspiración con aguja fina), usando aprendizaje automático clásico.")

tab1, tab2, tab3, tab4 = st.tabs(["🧍 Predicción individual", "📂 Predicción por CSV", "📊 Desempeño del modelo", "ℹ️ Acerca del proyecto"])

# ================================================================== TAB 1
with tab1:
    st.subheader("Ingresa las medidas del paciente")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.button("📋 Cargar promedio", on_click=cb_promedio, width="stretch")
    c2.button("🟢 Ejemplo benigno", on_click=cb_ejemplo, args=(0,), width="stretch", disabled=ejemplos is None)
    c3.button("🔴 Ejemplo maligno", on_click=cb_ejemplo, args=(1,), width="stretch", disabled=ejemplos is None)
    c4.caption("Usa los botones para precargar valores y luego ajústalos.")

    for clave, (titulo, patron) in GRUPOS.items():
        with st.expander(titulo, expanded=(clave == "worst")):
            cols = st.columns(5)
            for i, b in enumerate(BASE_ES):
                feat = patron.format(b)
                r = RANGOS[feat]
                paso = float((r["max"] - r["min"]) / 200) or 0.001
                with cols[i % 5]:
                    st.slider(BASE_ES[b], float(r["min"]), float(r["max"]), key=f"sl_{feat}",
                              step=paso, format="%.4f" if r["max"] < 1 else "%.2f")

    entrada = pd.DataFrame([{f: st.session_state[f"sl_{f}"] for f in FEATURES}])
    proba, pred = predecir(entrada, umbral)
    p = float(proba[0])

    st.divider()
    st.subheader("Resultado")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if pred[0] == 1:
            st.error("### 🔴 Posible tumor MALIGNO")
        else:
            st.success("### 🟢 Posible tumor BENIGNO")
        st.metric("Probabilidad de malignidad", f"{p:.1%}")
        st.metric("Probabilidad de benignidad", f"{1 - p:.1%}")
    with col_b:
        st.markdown("**Nivel de riesgo estimado**")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(f"Umbral de alerta actual: {umbral:.0%}. Valores por encima se clasifican como maligno.")
        if abs(p - umbral) < 0.10:
            st.info("⚠️ La probabilidad está cerca del umbral: es un caso límite que requiere especial revisión profesional.")
        with st.expander("Ver valores ingresados"):
            st.dataframe(entrada.T.rename(columns={0: "Valor"}), width="stretch")

# ================================================================== TAB 2
with tab2:
    st.subheader("Predicción de varios pacientes desde un archivo CSV")
    st.markdown("El archivo debe contener las **30 columnas** del modelo con los mismos nombres del dataset "
                "(p. ej. `mean radius`, `worst area`…). Si incluye una columna `diagnostico` (1 = maligno, 0 = benigno), "
                "la app calculará además el desempeño sobre ese archivo.")

    plantilla = (ejemplos[FEATURES + ["diagnostico"]] if ejemplos is not None
                 else pd.DataFrame([valores_por_defecto()]))
    st.download_button("⬇️ Descargar plantilla de ejemplo (CSV)", plantilla.to_csv(index=False).encode("utf-8"),
                       "plantilla_pacientes.csv", "text/csv")

    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if archivo is not None:
        try:
            df_in = pd.read_csv(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()

        faltantes = [f for f in FEATURES if f not in df_in.columns]
        if faltantes:
            st.error(f"Faltan {len(faltantes)} columnas requeridas. Ejemplos: {', '.join(faltantes[:5])}")
        elif df_in[FEATURES].isnull().any().any():
            st.error("El archivo tiene valores vacíos en las columnas del modelo. Corrígelos e inténtalo de nuevo.")
        else:
            proba, pred = predecir(df_in, umbral)
            res = df_in.copy()
            res["prob_maligno"] = proba.round(4)
            res["prediccion"] = np.where(pred == 1, "Maligno", "Benigno")

            m1, m2, m3 = st.columns(3)
            m1.metric("Pacientes analizados", len(res))
            m2.metric("Clasificados como malignos", int((pred == 1).sum()))
            m3.metric("Clasificados como benignos", int((pred == 0).sum()))

            if "diagnostico" in df_in.columns:
                from sklearn.metrics import accuracy_score, recall_score, precision_score
                y = df_in["diagnostico"].astype(int)
                st.markdown("**Desempeño sobre el archivo cargado**")
                k1, k2, k3 = st.columns(3)
                k1.metric("Accuracy", f"{accuracy_score(y, pred):.1%}")
                k2.metric("Precision", f"{precision_score(y, pred, zero_division=0):.1%}")
                k3.metric("Recall", f"{recall_score(y, pred, zero_division=0):.1%}")
                st.caption("Nota: si el archivo incluye datos usados en el entrenamiento, estas cifras serán más optimistas que en datos nuevos.")

            orden = ["prediccion", "prob_maligno"] + [c for c in res.columns if c not in ("prediccion", "prob_maligno")]
            st.dataframe(res[orden].style.apply(
                lambda fila: ["background-color: #fde2e4" if fila["prediccion"] == "Maligno" else "" for _ in fila], axis=1),
                width="stretch", height=350)
            st.bar_chart(res["prediccion"].value_counts())
            st.download_button("⬇️ Descargar resultados (CSV)", res[orden].to_csv(index=False).encode("utf-8"),
                               "resultados_predicciones.csv", "text/csv")

# ================================================================== TAB 3
with tab3:
    st.subheader("Desempeño del modelo en el conjunto de prueba (20 % de los datos)")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{METRICAS['accuracy']:.2%}")
    k[1].metric("Precision", f"{METRICAS['precision']:.2%}")
    k[2].metric("Recall", f"{METRICAS['recall']:.2%}")
    k[3].metric("F1-score", f"{METRICAS['f1']:.2%}")
    k[4].metric("ROC-AUC", f"{METRICAS['roc_auc']:.3f}")

    st.markdown(
        "- **Precision:** de los casos que el modelo marcó como malignos, qué porcentaje realmente lo era.\n"
        "- **Recall (sensibilidad):** de los tumores realmente malignos, qué porcentaje detectó el modelo. "
        "Es la métrica prioritaria en salud, porque un falso negativo es el error más grave.\n"
        "- **ROC-AUC:** capacidad de separar ambas clases (1.0 = perfecto, 0.5 = azar).")

    st.markdown("#### Comparación de algoritmos (validación cruzada 5-fold sobre entrenamiento)")
    comp = pd.DataFrame(info["comparacion"]).set_index("Modelo")
    st.dataframe(comp.style.format("{:.3f}").highlight_max(axis=0, color="#d8f3dc"), width="stretch")
    st.bar_chart(comp[["recall", "f1", "roc_auc"]])

    st.markdown("#### Mejores hiperparámetros (GridSearchCV)")
    st.json(info["mejores_parametros"])

# ================================================================== TAB 4
with tab4:
    st.subheader("Acerca del proyecto")
    st.markdown("""
**Asignatura:** Aprendizaje Automático · Especialización en Inteligencia Artificial · CUN  
**Proyecto:** ACA N° 1 – Modelado con algoritmos clásicos de Machine Learning y despliegue en Streamlit  
**Pilar:** Salud

**Pregunta orientadora:** ¿Cómo transformar datos clínicos en conocimiento predictivo mediante técnicas clásicas de
Machine Learning y presentarlo en una interfaz interactiva accesible?

**Flujo metodológico:** EDA → Limpieza y preprocesamiento → Modelado clásico (Regresión Logística, KNN, SVM, Random Forest)
→ Evaluación (accuracy, precision, recall, F1, ROC-AUC) → Despliegue en Streamlit.

**Datos:** 569 casos (357 benignos y 212 malignos) con 30 variables calculadas a partir de imágenes digitalizadas de
aspirados con aguja fina. Cada una de las 10 características (radio, textura, perímetro, área, suavidad, compacidad,
concavidad, puntos cóncavos, simetría y dimensión fractal) se resume como *promedio*, *error estándar* y *peor valor*.

**Referencia:** Wolberg, W., Street, N., & Mangasarian, O. (1995). *UCI Breast Cancer Wisconsin (Diagnostic)*
[Conjunto de datos]. UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
""")
    st.warning("**Aviso:** esta aplicación es un prototipo académico. Sus resultados no constituyen un diagnóstico médico "
               "y deben ser valorados siempre por un profesional de la salud.")
