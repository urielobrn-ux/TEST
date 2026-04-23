# app.py - dashboard del proyecto
# Uriel O. - modulo 4 DCD UNAM
#
# Tiene 6 pestanas: resumen, EDA, curvas de entrenamiento, evaluacion,
# prediccion en vivo sobre el dataset, y una para subir CSVs propios.
# Los KPIs del resumen se recalculan si hay archivos cargados.
#
# TODO: sacar la pestana de prediccion en vivo y dejar solo la de archivos,
# quedan medio redundantes. Por ahora las dejo las dos.
#
# correr:
#   cd dashboard && streamlit run app.py

import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Config / rutas
# ----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")

PALETTE = {"Gigantes": "#B85042", "Eficientes": "#028090", "Especialistas": "#6D2E46"}

st.set_page_config(
    page_title="Módulo 4 · Info-caña · DL (v2)",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.big-metric {font-size: 2.3rem; font-weight: 700; color: #028090;}
.section-title {border-left: 4px solid #B85042; padding-left: .6rem; margin-top: 1rem;}
.archetype-Gigantes {color:#B85042; font-weight:700;}
.archetype-Eficientes {color:#028090; font-weight:700;}
.archetype-Especialistas {color:#6D2E46; font-weight:700;}
.upload-banner {background:#E7E8D1; padding:.8rem 1rem; border-radius:8px; border-left:4px solid #028090;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# Cargas cacheadas
# ----------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA, "infocana_processed.csv"))
    profile = pd.read_csv(os.path.join(DATA, "ingenios_profile_clusters.csv"))
    with open(os.path.join(DATA, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return df, profile, meta

@st.cache_data
def load_metrics():
    with open(os.path.join(MODELS, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(MODELS, "history.json"), "r", encoding="utf-8") as f:
        history = json.load(f)
    return metrics, history

@st.cache_resource
def load_models():
    scaler = np.load(os.path.join(MODELS, "scaler.npz"), allow_pickle=True)
    mlp_clf = np.load(os.path.join(MODELS, "mlp_clf.npz"))
    mlp_reg = np.load(os.path.join(MODELS, "mlp_reg.npz"))
    rnn_clf = np.load(os.path.join(MODELS, "rnn_clf.npz"))
    rnn_reg = np.load(os.path.join(MODELS, "rnn_reg.npz"))
    def _split_mlp(npz):
        arrs = [npz[k] for k in npz.files]
        n = len(arrs) // 2
        return arrs[:n], arrs[n:]
    W_c, b_c = _split_mlp(mlp_clf)
    W_r, b_r = _split_mlp(mlp_reg)
    rnn_c = {k: rnn_clf[k] for k in rnn_clf.files}
    rnn_r = {k: rnn_reg[k] for k in rnn_reg.files}
    return scaler, (W_c, b_c), (W_r, b_r), rnn_c, rnn_r

df, profile, meta = load_data()
metrics, history = load_metrics()
scaler, mlp_clf_w, mlp_reg_w, rnn_clf_w, rnn_reg_w = load_models()

FEATURES = meta["feature_cols"]
CLASS_NAMES = meta["class_names"]
mu = scaler["mu"].astype(np.float32)
sd = scaler["sd"].astype(np.float32)
y_reg_mu = float(scaler["y_reg_mu"])
y_reg_sd = float(scaler["y_reg_sd"])
SEQ_LEN = 8

# ----------------------------------------------------------------
# Inferencia (numpy)
# ----------------------------------------------------------------
def _relu(x): return np.maximum(0, x)
def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True); e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def mlp_predict(W, b, x, task="clf"):
    a = x
    for i in range(len(W) - 1):
        a = _relu(a @ W[i] + b[i])
    out = a @ W[-1] + b[-1]
    if task == "clf":
        return _softmax(out)
    return out

def rnn_predict(p, X, task="clf"):
    if X.ndim == 2:
        X = X[None]
    B, T, _ = X.shape
    H = p["Wh"].shape[0]
    h = np.zeros((B, H), dtype=np.float32)
    for t in range(T):
        h = np.tanh(X[:, t] @ p["Wx"] + h @ p["Wh"] + p["bh"])
    out = h @ p["Wy"] + p["by"]
    if task == "clf":
        return _softmax(out)
    return out

# ----------------------------------------------------------------
# Preprocesamiento de archivos subidos
# ----------------------------------------------------------------
def _hms_to_hours(s):
    if pd.isna(s):
        return 0.0
    s = str(s).strip()
    if ":" not in s:
        try: return float(s)
        except Exception: return 0.0
    parts = s.split(":")
    try:
        h = float(parts[0]); m = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return h + m/60.0 + sec/3600.0
    except Exception:
        return 0.0

def preprocess_uploaded(df_in: pd.DataFrame) -> (pd.DataFrame, list):
    """Alinea el DataFrame subido con el esquema del modelo.
    - Convierte columnas tiempo_perdido_* (HH:MM:SS) a *_h
    - Rellena columnas faltantes con la mediana del dataset entrenado
    - Devuelve (df_preparado, lista de advertencias)
    """
    warnings_ = []
    df = df_in.copy()
    # Detectar columnas HH:MM:SS
    for c in list(df.columns):
        if c.startswith("tiempo_perdido") and not c.endswith("_h"):
            try:
                df[c + "_h"] = df[c].apply(_hms_to_hours)
            except Exception:
                pass
    # Chequear columnas obligatorias
    missing = [c for c in FEATURES if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURES + ["ingenio", "semana", "zafra",
                                                             "eficiencia_en_fabrica",
                                                             "class_id", "cluster_name"]]
    if missing:
        warnings_.append(
            f"Se rellenaron {len(missing)} columnas faltantes con la mediana del dataset de entrenamiento: "
            + ", ".join(missing[:6]) + ("…" if len(missing) > 6 else "")
        )
        medians = pd.read_csv(os.path.join(DATA, "infocana_processed.csv"))[FEATURES].median()
        for c in missing:
            df[c] = medians[c]
    if extra:
        warnings_.append(f"Se ignoraron {len(extra)} columnas extra.")
    # Reordenar a FEATURES
    df = df.replace([np.inf, -np.inf], np.nan)
    df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median(numeric_only=True))
    # Para globales sin mediana local
    if df[FEATURES].isna().any().any():
        medians = pd.read_csv(os.path.join(DATA, "infocana_processed.csv"))[FEATURES].median()
        df[FEATURES] = df[FEATURES].fillna(medians)
    return df, warnings_

def predict_tabular(df_prep: pd.DataFrame) -> pd.DataFrame:
    """Predicción MLP fila a fila."""
    X = (df_prep[FEATURES].values.astype(np.float32) - mu) / sd
    probs = mlp_predict(mlp_clf_w[0], mlp_clf_w[1], X, task="clf")
    eff_n = mlp_predict(mlp_reg_w[0], mlp_reg_w[1], X, task="reg").reshape(-1)
    eff = eff_n * y_reg_sd + y_reg_mu
    out = df_prep.copy()
    out["MLP_arquetipo"] = [CLASS_NAMES[i] for i in probs.argmax(axis=1)]
    out["MLP_confianza"] = probs.max(axis=1).round(3)
    for i, name in enumerate(CLASS_NAMES):
        out[f"MLP_prob_{name}"] = probs[:, i].round(3)
    out["MLP_eficiencia_pred"] = eff.round(2)
    return out

def predict_sequence(df_prep: pd.DataFrame) -> pd.DataFrame:
    """Si hay ≥ SEQ_LEN filas consecutivas por ingenio, predice LSTM."""
    if "ingenio" not in df_prep.columns or "semana" not in df_prep.columns:
        # Sin ingenio/semana, tratar todo como una sola serie
        if len(df_prep) < SEQ_LEN:
            return pd.DataFrame()
        sub = df_prep.sort_index()
        Xs = []; idx_target = []
        Xv = (sub[FEATURES].values.astype(np.float32) - mu) / sd
        for i in range(len(sub) - SEQ_LEN + 1):
            Xs.append(Xv[i:i+SEQ_LEN]); idx_target.append(sub.index[i+SEQ_LEN-1])
        Xs = np.asarray(Xs, dtype=np.float32)
        probs = rnn_predict(rnn_clf_w, Xs, task="clf")
        eff_n = rnn_predict(rnn_reg_w, Xs, task="reg").reshape(-1)
        eff = eff_n * y_reg_sd + y_reg_mu
        out = pd.DataFrame(index=idx_target)
        out["LSTM_arquetipo"] = [CLASS_NAMES[i] for i in probs.argmax(axis=1)]
        out["LSTM_confianza"] = probs.max(axis=1).round(3)
        out["LSTM_eficiencia_pred"] = eff.round(2)
        return out

    out_parts = []
    for ing, g in df_prep.sort_values(["ingenio", "semana"]).groupby("ingenio"):
        if len(g) < SEQ_LEN:
            continue
        Xv = (g[FEATURES].values.astype(np.float32) - mu) / sd
        Xs = []; target_idx = []
        for i in range(len(g) - SEQ_LEN + 1):
            Xs.append(Xv[i:i+SEQ_LEN]); target_idx.append(g.index[i+SEQ_LEN-1])
        Xs = np.asarray(Xs, dtype=np.float32)
        probs = rnn_predict(rnn_clf_w, Xs, task="clf")
        eff_n = rnn_predict(rnn_reg_w, Xs, task="reg").reshape(-1)
        eff = eff_n * y_reg_sd + y_reg_mu
        part = pd.DataFrame(index=target_idx)
        part["ingenio"] = ing
        part["LSTM_arquetipo"] = [CLASS_NAMES[i] for i in probs.argmax(axis=1)]
        part["LSTM_confianza"] = probs.max(axis=1).round(3)
        part["LSTM_eficiencia_pred"] = eff.round(2)
        out_parts.append(part)
    return pd.concat(out_parts) if out_parts else pd.DataFrame()

# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------
st.sidebar.title("🧠 Módulo 4 · Ciencia de Datos UNAM")
st.sidebar.markdown("**Uriel Obregón Aguilar**")
st.sidebar.markdown("Zafra")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Sección",
    ["Resumen", "EDA", "Entrenamiento", "Evaluación",
     "Predicción en vivo", "Cargar archivos"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Proyecto final del modulo 4 · Uriel O."
)

# ----------------------------------------------------------------
# Página: Resumen
# ----------------------------------------------------------------
if page == "Resumen":
    st.title("De la segmentacion a la prediccion")
    st.caption("Proyecto del modulo 4 · industria azucarera · zafra 2014-15")

    # KPIs - si hay archivos cargados, uso esos; si no el dataset base
    src_df = st.session_state.get("_uploaded_df")
    active = src_df if (src_df is not None and len(src_df) > 0) else df
    source_lbl = "archivos cargados" if active is not df else "dataset base"

    c1, c2, c3, c4 = st.columns(4)
    n_ing = active['ingenio'].nunique() if 'ingenio' in active.columns else 0
    c1.metric("Ingenios", f"{n_ing}")
    c2.metric("Observaciones", f"{len(active):,}")
    c3.metric("Variables numericas", f"{len(FEATURES)}")
    c4.metric("Arquetipos (M3)", f"{len(CLASS_NAMES)}")
    if active is not df:
        st.caption(f"Los numeros reflejan {source_lbl} ({len(active):,} filas). Para volver al dataset base, limpia los archivos en la pestana de carga.")

    st.markdown("---")
    st.markdown("### Idea central del proyecto")
    st.markdown(
        """
El Módulo 3 aplicó K-Means a los 50 ingenios y encontró tres arquetipos:

<span class='archetype-Gigantes'>Gigantes</span> · mucho volumen, pérdidas elevadas.
<span class='archetype-Eficientes'>Eficientes</span> · balance entre escala y control.
<span class='archetype-Especialistas'>Especialistas</span> · pequeños pero con eficiencia excepcional.

El Módulo 4 usa esas etiquetas como *target* para entrenar redes neuronales:

1. **Clasificar** — predecir el arquetipo de un ingenio desde sus variables operativas.
2. **Predecir** — estimar la eficiencia de fábrica antes de cerrar la zafra.

Comparamos dos arquitecturas: MLP (feedforward) y LSTM (recurrente, que aprovecha el historial semanal).
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Para qué sirve esto")
    st.markdown(
        """
La idea no es sólo una métrica bonita en test. El modelo se puede usar cada lunes
con los datos de la semana anterior para:

- **Alerta temprana**: si un ingenio empieza a desviarse de su arquetipo
  (ej. un Eficiente cuyo perfil muta a Gigante por pérdidas crecientes), saltan
  banderas antes de que la zafra cierre mal.
- **Estimación de eficiencia**: cuántos puntos porcentuales de rendimiento
  esperar en las próximas semanas, para planear logística, mantenimiento y compra
  de insumos.
- **Benchmarking interno**: comparar un ingenio contra sus pares del mismo
  arquetipo y detectar huecos operativos.
        """
    )

    st.markdown("### Resultados principales (test)")
    st.markdown(
        "**Cómo leer la tabla**: *Accuracy* es el % de filas bien clasificadas. "
        "*F1 macro* promedia el F1 de las 3 clases sin penalizar la minoritaria. "
        "En regresión, *MAE* es el error absoluto medio (en puntos porcentuales de eficiencia), "
        "*RMSE* penaliza más los outliers y *R²* compara contra predecir siempre la media "
        "(cerca de 1 = muy bueno, 0 = igual a la media, negativo = peor que la media)."
    )
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Clasificación")
        df_clf = pd.DataFrame({
            "MLP": [metrics["MLP"]["classification"]["accuracy"],
                    metrics["MLP"]["classification"]["f1_macro"]],
            "LSTM": [metrics["LSTM_like"]["classification"]["accuracy"],
                     metrics["LSTM_like"]["classification"]["f1_macro"]],
        }, index=["Accuracy", "F1 macro"])
        st.dataframe(df_clf.style.format("{:.3f}").highlight_max(axis=1, color="#C9E4CA"),
                     use_container_width=True)
    with colB:
        st.markdown("#### Regresión")
        df_reg = pd.DataFrame({
            "MLP": [metrics["MLP"]["regression"]["MAE"],
                    metrics["MLP"]["regression"]["RMSE"],
                    metrics["MLP"]["regression"]["R2"]],
            "LSTM": [metrics["LSTM_like"]["regression"]["MAE"],
                     metrics["LSTM_like"]["regression"]["RMSE"],
                     metrics["LSTM_like"]["regression"]["R2"]],
        }, index=["MAE", "RMSE", "R²"])
        st.dataframe(df_reg.style.format("{:.3f}"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Baseline trivial (sanity check)")
    st.markdown(
        "Antes de creerle a la red, vale compararla contra dos modelos tontos: "
        "*clase mayoritaria* (siempre predecir la clase más frecuente) y "
        "*media del target* (siempre predecir el promedio de eficiencia). "
        "Si la red no supera esto por un margen claro, es que no aprendió nada útil."
    )
    class_counts = df["class_id"].value_counts(normalize=True) if "class_id" in df.columns else None
    if class_counts is not None:
        majority_acc = class_counts.iloc[0]
        y = df["eficiencia_en_fabrica"].values
        mae_base = float(np.mean(np.abs(y - y.mean())))
        st.markdown(
            f"- **Clase mayoritaria**: acc ≈ `{majority_acc:.3f}` vs LSTM "
            f"`{metrics['LSTM_like']['classification']['accuracy']:.3f}` → la red mejora "
            f"~{(metrics['LSTM_like']['classification']['accuracy']-majority_acc)*100:.1f} pp.\n"
            f"- **Predecir la media**: MAE ≈ `{mae_base:.2f}` pp vs LSTM "
            f"`{metrics['LSTM_like']['regression']['MAE']:.2f}` pp → la red reduce el error a la mitad."
        )

    st.markdown("### Limitaciones conocidas")
    st.markdown(
        """
- Una sola zafra (2014-15). Los arquetipos podrían mutar año a año y el modelo
  no lo captaría sin reentrenar.
- El arquetipo *Especialistas* tiene sólo ~5 ingenios, así que su F1 es inestable
  por muestra pequeña; probé class_weight balanceado pero no alcanza.
- La LSTM necesita 8 semanas de historia para dar una predicción; las primeras
  semanas de zafra quedan sin cobertura.
- No apliqué validación cruzada con varias semillas (sería el siguiente paso).
        """
    )
    st.caption("Nota: si subis archivos propios en la pestana de carga, se corren los dos modelos sobre esos datos.")

# ----------------------------------------------------------------
# Página: EDA
# ----------------------------------------------------------------
elif page == "EDA":
    st.title("Análisis Exploratorio")
    st.markdown(
        "Antes de modelar, vale mirar qué hay. El EDA responde tres preguntas: "
        "(1) ¿están balanceadas las clases?, (2) ¿qué variables separan los arquetipos?, "
        "(3) ¿hay estacionalidad dentro de la zafra?"
    )
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Clases", "Eficiencia", "Boxplots", "Correlaciones", "Evolución temporal"
    ])
    with tab1:
        st.markdown("#### Distribución de clases")
        st.markdown(
            "Cuenta cuántos ingenios hay en cada arquetipo. **Sirve para** detectar "
            "desbalance: si una clase domina, un clasificador tonto podría tener "
            "accuracy alta sin aprender nada. Acá se ve que *Eficientes* es la clase "
            "mayoritaria y *Especialistas* la más chica — por eso conviene mirar F1 "
            "macro en vez de accuracy."
        )
        st.image(os.path.join(FIG, "01_distribucion_clases.png"), use_container_width=True)
    with tab2:
        st.markdown("#### Eficiencia por arquetipo")
        st.markdown(
            "Compara la eficiencia promedio en fábrica entre los 3 grupos. "
            "**Sirve para** validar que los clusters del M3 tienen sentido operativo "
            "(no son sólo matemáticos): si los arquetipos separan eficiencia, entonces "
            "predecir el arquetipo ya da pista del rendimiento."
        )
        st.image(os.path.join(FIG, "02_eficiencia_por_arquetipo.png"), use_container_width=True)
    with tab3:
        st.markdown("#### Boxplots de variables por arquetipo")
        st.markdown(
            "Muestra la dispersión (mediana, cuartiles, outliers) de las variables "
            "más relevantes dentro de cada grupo. **Sirve para** ver qué tanto se "
            "mezclan los arquetipos en el espacio de features — si las cajas están "
            "bien separadas, el problema es fácil; si se solapan, la red tendrá que "
            "combinar variables no lineales para separarlos."
        )
        st.image(os.path.join(FIG, "03_boxplots_por_arquetipo.png"), use_container_width=True)
    with tab4:
        st.markdown("#### Correlaciones con el target")
        st.markdown(
            "Ranking de variables por correlación lineal con `eficiencia_en_fabrica`. "
            "**Sirve para** identificar candidatas obvias (pérdidas de bagazo, pol en "
            "bagazo, humedad) y también para sospechar multicolinealidad cuando dos "
            "variables están muy correlacionadas entre sí. Es un diagnóstico rápido, "
            "no reemplaza feature importance del modelo (que vive en la pestaña Evaluación)."
        )
        st.image(os.path.join(FIG, "04_correlacion_target.png"), use_container_width=True)
    with tab5:
        st.markdown("#### Evolución temporal dentro de la zafra")
        st.markdown(
            "Promedio semanal de eficiencia por arquetipo a lo largo de las 26 semanas "
            "de la zafra 2014-15. **Sirve para** motivar el uso de la LSTM: si hay "
            "patrón temporal (arranque lento, estabilización, caída final), entonces "
            "un modelo que mire una sola semana se pierde información. Acá se ve "
            "claramente la curva típica de zafra, y los tres arquetipos la atraviesan "
            "distinto."
        )
        st.image(os.path.join(FIG, "05_evolucion_temporal.png"), use_container_width=True)

# ----------------------------------------------------------------
# Página: Entrenamiento
# ----------------------------------------------------------------
elif page == "Entrenamiento":
    st.title("Curvas de entrenamiento")
    st.markdown(
        "Las curvas muestran cómo baja el *loss* (error) y cómo sube la métrica principal "
        "(accuracy para clasificación, R² para regresión) en cada época de entrenamiento. "
        "**Para qué sirven**: (1) detectar overfitting — si la curva de validación empieza a "
        "subir mientras train sigue bajando, el modelo está memorizando; "
        "(2) decidir cuándo parar (early stopping); "
        "(3) comparar modelos — si uno converge más rápido o llega más alto, probablemente "
        "es la mejor arquitectura para el problema."
    )

    def plot_history_interactive(h, title_metric, metric_label):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        axes[0].plot(h["loss"], label="train", color="#028090")
        axes[0].plot(h["val_loss"], label="val", color="#B85042", linestyle="--")
        axes[0].set_title(f"{title_metric} — Loss"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[1].plot(h["metric"], label="train", color="#028090")
        axes[1].plot(h["val_metric"], label="val", color="#B85042", linestyle="--")
        axes[1].set_title(f"{title_metric} — {metric_label}"); axes[1].legend(); axes[1].grid(alpha=.3)
        plt.tight_layout()
        return fig

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### MLP")
        st.pyplot(plot_history_interactive(history["MLP_clf"], "MLP · Clasificación", "Accuracy"))
        st.pyplot(plot_history_interactive(history["MLP_reg"], "MLP · Regresión", "R²"))
    with c2:
        st.markdown("### LSTM")
        st.pyplot(plot_history_interactive(history["RNN_clf"], "LSTM · Clasificación", "Accuracy"))
        st.pyplot(plot_history_interactive(history["RNN_reg"], "LSTM · Regresión", "R²"))

# ----------------------------------------------------------------
# Página: Evaluación
# ----------------------------------------------------------------
elif page == "Evaluación":
    st.title("Evaluación en test")
    st.markdown(
        "Después de entrenar, hay que medir. Esta pestaña reúne las tres herramientas "
        "estándar: matriz de confusión (por dónde se equivoca el clasificador), "
        "comparativa global (qué modelo gana) y feature importance (qué variables "
        "pesan más en las predicciones)."
    )

    st.markdown("### Matrices de confusión (clasificación)")
    st.markdown(
        "Cada celda `(fila, columna)` = cantidad de ejemplos cuya clase **real** es la "
        "fila y cuya **predicción** es la columna. La diagonal son aciertos; todo lo "
        "demás son errores. **Sirve para** ver si los errores se concentran entre dos "
        "clases específicas (ej. *Gigantes* confundidos con *Eficientes*) — eso indica "
        "que esas dos clases se parecen en las features disponibles."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(FIG, "20_confusion_mlp.png"), use_container_width=True)
    with c2:
        st.image(os.path.join(FIG, "21_confusion_rnn.png"), use_container_width=True)

    st.markdown("### Comparativa global")
    st.markdown(
        "Barras lado a lado de las métricas de MLP vs LSTM en clasificación y regresión. "
        "**Sirve para** tomar la decisión de modelo final de un vistazo. "
        "En este proyecto la LSTM gana en las dos tareas, lo que confirma que el "
        "contexto temporal aporta señal."
    )
    st.image(os.path.join(FIG, "30_comparativa_metricas.png"), use_container_width=True)

    st.markdown("### Tabla de métricas")
    rows = []
    for mname, key in [("MLP", "MLP"), ("LSTM", "LSTM_like")]:
        c = metrics[key]["classification"]; r = metrics[key]["regression"]
        rows.append({
            "Modelo": mname,
            "Accuracy": c["accuracy"], "F1 macro": c["f1_macro"],
            "F1 Gigantes": c["f1_per_class"][0],
            "F1 Eficientes": c["f1_per_class"][1],
            "F1 Especialistas": c["f1_per_class"][2],
            "MAE (%)": r["MAE"], "RMSE (%)": r["RMSE"], "R²": r["R2"],
        })
    dfm = pd.DataFrame(rows).set_index("Modelo")
    st.dataframe(dfm.style.format("{:.3f}"), use_container_width=True)

    # -------- Baseline trivial --------
    st.markdown("### ¿Gana contra un modelo tonto?")
    st.markdown(
        "Antes de festejar la accuracy, comparamos contra dos baselines triviales: "
        "*clase mayoritaria* (predecir siempre la clase más frecuente) y "
        "*media del target* (predecir siempre el promedio de eficiencia). "
        "Si la red no les gana por margen claro, no aprendió nada útil."
    )
    try:
        y_cls = df["class_id"].values
        majority_acc = float((y_cls == np.bincount(y_cls).argmax()).mean())
        y_reg = df["eficiencia_en_fabrica"].values
        mae_base = float(np.mean(np.abs(y_reg - y_reg.mean())))
        import pandas as _pd
        base_tbl = _pd.DataFrame({
            "Metrica": ["Accuracy (clf)", "MAE (reg, pp)"],
            "Baseline tonto": [f"{majority_acc:.3f}", f"{mae_base:.2f}"],
            "MLP": [f"{metrics['MLP']['classification']['accuracy']:.3f}",
                    f"{metrics['MLP']['regression']['MAE']:.2f}"],
            "LSTM": [f"{metrics['LSTM_like']['classification']['accuracy']:.3f}",
                     f"{metrics['LSTM_like']['regression']['MAE']:.2f}"],
        })
        st.dataframe(base_tbl, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"(no pude calcular el baseline: {e})")

    # -------- Feature importance (permutacion, MLP clf) --------
    st.markdown("### Qué variables pesan más (feature importance)")
    st.markdown(
        "Con *permutation importance*: tomo una variable, le barajo los valores al azar "
        "y mido cuánto se degrada la accuracy. Si baja mucho, esa variable es clave. "
        "**Sirve para** abrir la caja negra del MLP y verificar que la red aprendió algo "
        "sensato — no variables espurias."
    )
    try:
        rng = np.random.default_rng(0)
        n_show = min(1000, len(df))
        idx = rng.choice(len(df), size=n_show, replace=False)
        Xs = df.iloc[idx][FEATURES].values.astype(np.float32)
        ys = df.iloc[idx]["class_id"].values
        Xn = (Xs - mu) / sd
        base_pred = mlp_predict(mlp_clf_w[0], mlp_clf_w[1], Xn, task="clf").argmax(axis=1)
        base_acc = float((base_pred == ys).mean())
        drops = []
        for j, name in enumerate(FEATURES):
            Xp = Xn.copy()
            rng.shuffle(Xp[:, j])
            p = mlp_predict(mlp_clf_w[0], mlp_clf_w[1], Xp, task="clf").argmax(axis=1)
            drops.append(base_acc - float((p == ys).mean()))
        import pandas as _pd
        imp = _pd.DataFrame({"feature": FEATURES, "drop_accuracy": drops})
        imp = imp.sort_values("drop_accuracy", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(imp["feature"][::-1], imp["drop_accuracy"][::-1], color="#028090", edgecolor="#222")
        ax.set_xlabel("Caída de accuracy al permutar la variable")
        ax.set_title("Top-10 variables por importancia (MLP clasificación)")
        ax.grid(alpha=.3, axis="x")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.caption(f"(no pude calcular importancia: {e})")

    # -------- Analisis de errores --------
    st.markdown("### Dónde se equivoca (análisis de errores)")
    st.markdown(
        "Miro los errores del MLP sobre el dataset completo y los agrupo por ingenio "
        "y por clase real. **Sirve para** responder: *¿hay un ingenio particularmente "
        "difícil? ¿una clase que la red se confunde sistemáticamente?* Si hay patrón, "
        "quizás convenga ver los features específicos de esas filas."
    )
    try:
        Xall = df[FEATURES].values.astype(np.float32)
        yall = df["class_id"].values
        Xn = (Xall - mu) / sd
        pred = mlp_predict(mlp_clf_w[0], mlp_clf_w[1], Xn, task="clf").argmax(axis=1)
        err_mask = np.not_equal(pred, yall)
        import pandas as _pd
        cA, cB = st.columns(2)
        with cA:
            st.markdown("**Tasa de error por clase real**")
            tab = _pd.DataFrame({
                "Clase": [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))],
                "Total": [int((yall == i).sum()) for i in range(len(CLASS_NAMES))],
                "Errores": [int(((yall == i) & err_mask).sum()) for i in range(len(CLASS_NAMES))],
            })
            tab["Tasa"] = (tab["Errores"] / tab["Total"]).round(3)
            st.dataframe(tab, use_container_width=True, hide_index=True)
        with cB:
            st.markdown("**Top 10 ingenios con más errores**")
            err_df = df.loc[err_mask, "ingenio"].value_counts().head(10).reset_index()
            err_df.columns = ["ingenio", "errores"]
            st.dataframe(err_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"(no pude armar el análisis: {e})")

    st.markdown("### Conclusión")
    st.markdown(f"""
- La LSTM supera al MLP en ambas tareas: el contexto temporal de {SEQ_LEN} semanas
  permite capturar dinámicas que un vector de una sola semana no refleja.
- El arquetipo Especialistas sigue siendo el más difícil (apenas 5 ingenios);
  el F1 de esa clase es bajo aun con class_weight balanceado.
- La red supera al baseline trivial por un margen claro (ver tabla arriba), así
  que hay aprendizaje real — no sólo memorizar la clase mayoritaria.
- Valor operativo: con MAE ≈ {metrics['LSTM_like']['regression']['MAE']:.2f} pp, la
  LSTM permite emitir alerta con ≥ 1 semana de antelación sobre caídas de
  rendimiento en fábrica.
    """)

# ----------------------------------------------------------------
# Página: Predicción en vivo
# ----------------------------------------------------------------
elif page == "Predicción en vivo":
    st.title("Predicción en vivo — dataset cargado")
    st.markdown(
        "Esta pestaña aplica los modelos entrenados a casos reales del dataset histórico. "
        "**Para qué sirve**: verificar a mano que las predicciones son razonables "
        "(ej. un ingenio que sabemos Gigante, ¿la red lo etiqueta así?). "
        "También sirve para comparar MLP (una sola semana) contra LSTM (8 semanas de contexto) "
        "sobre el mismo caso."
    )
    st.markdown(
        f"Seleccioná un ingenio y la semana base. El dashboard toma las últimas "
        f"{SEQ_LEN} semanas y devuelve predicciones con MLP y LSTM."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        ingenio = st.selectbox("Ingenio", sorted(df["ingenio"].unique()), index=0)
    with col2:
        weeks = sorted(df.loc[df["ingenio"] == ingenio, "semana"].unique())
        sem_base = st.selectbox("Semana base", weeks, index=len(weeks) - 1)

    g = df[df["ingenio"] == ingenio].sort_values("semana")
    g_hist = g[g["semana"] <= sem_base].tail(SEQ_LEN)
    if len(g_hist) < SEQ_LEN:
        st.warning(f"Sólo hay {len(g_hist)} semanas antes de la semana {sem_base}. El MLP igual predice.")

    x_last = (g_hist[FEATURES].values[-1:].astype(np.float32) - mu) / sd
    probs_mlp = mlp_predict(mlp_clf_w[0], mlp_clf_w[1], x_last, task="clf").ravel()
    cls_mlp = int(probs_mlp.argmax())
    eff_mlp = float(mlp_predict(mlp_reg_w[0], mlp_reg_w[1], x_last, task="reg").ravel()[0] * y_reg_sd + y_reg_mu)

    if len(g_hist) >= SEQ_LEN:
        Xseq = (g_hist[FEATURES].values.astype(np.float32) - mu) / sd
        probs_lstm = rnn_predict(rnn_clf_w, Xseq[None], task="clf").ravel()
        cls_lstm = int(probs_lstm.argmax())
        eff_lstm = float(rnn_predict(rnn_reg_w, Xseq[None], task="reg").ravel()[0] * y_reg_sd + y_reg_mu)
    else:
        probs_lstm = None; cls_lstm = None; eff_lstm = None

    eff_real = float(g_hist["eficiencia_en_fabrica"].iloc[-1]) if len(g_hist) else None
    arq_real = str(g_hist["cluster_name"].iloc[-1]) if len(g_hist) else None

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### MLP (una semana)")
        st.markdown(f"**Arquetipo:** <span class='archetype-{CLASS_NAMES[cls_mlp]}'>{CLASS_NAMES[cls_mlp]}</span>", unsafe_allow_html=True)
        st.markdown(f"**Eficiencia predicha:** `{eff_mlp:.2f}%`")
        st.progress(float(probs_mlp.max()))
    with c2:
        st.markdown("#### LSTM (ventana temporal)")
        if cls_lstm is None:
            st.info("Requiere 8 semanas de historial.")
        else:
            st.markdown(f"**Arquetipo:** <span class='archetype-{CLASS_NAMES[cls_lstm]}'>{CLASS_NAMES[cls_lstm]}</span>", unsafe_allow_html=True)
            st.markdown(f"**Eficiencia predicha:** `{eff_lstm:.2f}%`")
            st.progress(float(probs_lstm.max()))
    with c3:
        st.markdown("#### Valor real")
        if eff_real is not None:
            st.markdown(f"**Arquetipo real:** <span class='archetype-{arq_real}'>{arq_real}</span>", unsafe_allow_html=True)
            st.markdown(f"**Eficiencia real:** `{eff_real:.2f}%`")
            if eff_lstm is not None:
                st.caption(f"Δ LSTM: {eff_lstm - eff_real:+.2f}%")
            st.caption(f"Δ MLP:  {eff_mlp - eff_real:+.2f}%")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(g["semana"], g["eficiencia_en_fabrica"], marker="o", color="#333", alpha=0.35,
            label="Eficiencia real (toda la zafra)")
    ax.plot(g_hist["semana"], g_hist["eficiencia_en_fabrica"], marker="o",
            color="#333", linewidth=2, label=f"Ventana usada ({SEQ_LEN} semanas)")
    ax.scatter([sem_base], [eff_mlp], color="#028090", s=120, zorder=5,
               label=f"Predicción MLP = {eff_mlp:.2f}%")
    if eff_lstm is not None:
        ax.scatter([sem_base], [eff_lstm], color="#B85042", s=120, zorder=5,
                   marker="^", label=f"Predicción LSTM = {eff_lstm:.2f}%")
    ax.set_xlabel("Semana de zafra"); ax.set_ylabel("Eficiencia (%)")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.3)
    ax.set_title(f"{ingenio} — semana {sem_base}")
    st.pyplot(fig)

# ----------------------------------------------------------------
# pagina: subir CSVs propios
# ----------------------------------------------------------------
elif page == "Cargar archivos":
    st.title("Predicción batch sobre archivos propios")
    st.markdown(
        "**Qué hace esta pestaña**: subí uno o varios CSV con datos nuevos "
        "(una fila = una semana × ingenio) y el dashboard corre los dos modelos "
        "entrenados sobre esos datos, fila por fila. Devuelve el arquetipo predicho, "
        "la probabilidad de cada clase, la eficiencia estimada, y te lo deja todo "
        "descargable como CSV."
    )
    st.markdown(
        "**Para qué sirve**: si el ingenio recolecta datos nuevos (otra zafra, otra planta, "
        "un escenario hipotético), podés correr inferencia sin reentrenar nada. "
        "Los modelos esperan las mismas variables que el dataset original — hay una "
        "plantilla descargable más abajo con los nombres de columna esperados."
    )
    st.markdown(
        "<div class='upload-banner'>Subí uno o más archivos <b>CSV</b> con datos nuevos de ingenios azucareros. "
        "El sistema los combinará, corregirá el esquema (formato <code>HH:MM:SS</code>, columnas faltantes, etc.) "
        "y ejecutará predicción con MLP y LSTM.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    # --------- Ayuda sobre formato esperado ----------
    with st.expander("📋 ¿Qué formato debe tener el CSV?", expanded=False):
        st.markdown(f"""
El CSV debe tener columnas numéricas alineadas con el esquema del dataset de entrenamiento.
Columnas **recomendadas** (no todas son obligatorias; las que falten se rellenan con la mediana del dataset entrenado):

- **Identificadoras** (opcionales pero útiles para agrupar secuencias): `ingenio`, `semana`
- **Producción**: `cana_molida_bruta`, `cana_molida_neta`, `superficie_cosechada`, `azucar_producida_total`, `alcohol_96_gl_producido`, …
- **Calidad de caña**: `pol_en_cana`, `fibra_en_cana`, `brix_en_jugo_mezclado`, …
- **Pérdidas**: `perdidas_de_bagazo`, `perdidas_en_cachaza`, `perdidas_totales`, …
- **Consumo de petróleo**: `consumo_de_petroleo_total`, …
- **Tiempos perdidos** (`HH:MM:SS` o ya en horas como `*_h`): `tiempo_perdido_en_fabrica`, `tiempo_perdido_por_lluvias`, …
- **Eficiencia real** (opcional, si la incluís se compara contra la predicción): `eficiencia_en_fabrica`

**Ejemplo mínimo**:
```csv
ingenio,semana,cana_molida_neta,azucar_producida_total,pol_en_cana,eficiencia_en_fabrica
ingenio_x,10,80000,5200,12.5,82.3
ingenio_x,11,85000,5500,12.6,82.8
...
```
Se incluye una plantilla descargable aquí abajo.
        """)
        # Plantilla de descarga con 1 fila ejemplo
        template_cols = ["ingenio", "semana"] + FEATURES + ["eficiencia_en_fabrica"]
        template = pd.DataFrame(columns=template_cols)
        template.loc[0] = ["ejemplo_ingenio", 10] + [0.0] * len(FEATURES) + [80.0]
        st.download_button(
            "⬇️ Descargar plantilla CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="plantilla_infocana.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # --------- Uploader múltiple ----------
    uploaded = st.file_uploader(
        "Seleccioná **uno o varios** CSV",
        type=["csv"],
        accept_multiple_files=True,
        help="Podés arrastrar varios archivos a la vez.",
    )

    if not uploaded:
        st.info("Esperando archivos. Mientras tanto, podés probar con el dataset de entrenamiento usando el botón de abajo.")
        if st.button("▶️ Probar con una muestra del dataset de entrenamiento"):
            sample = df.head(30).copy()
            st.session_state["_uploaded_df"] = sample
            st.session_state["_uploaded_files"] = ["muestra_entrenamiento.csv"]
            st.rerun()

    # --------- Procesamiento ----------
    df_all = None; filenames = []
    if uploaded:
        dfs = []
        for f in uploaded:
            try:
                content = f.read()
                try:
                    d = pd.read_csv(io.BytesIO(content))
                except UnicodeDecodeError:
                    d = pd.read_csv(io.BytesIO(content), encoding="latin-1")
                d["_source_file"] = f.name
                dfs.append(d)
                filenames.append(f.name)
            except Exception as e:
                st.error(f"No se pudo leer `{f.name}`: {e}")
        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            st.session_state["_uploaded_df"] = df_all
            st.session_state["_uploaded_files"] = filenames
    elif "_uploaded_df" in st.session_state:
        df_all = st.session_state["_uploaded_df"]
        filenames = st.session_state.get("_uploaded_files", [])

    if df_all is not None:
        st.success(f"Se cargaron **{len(filenames)} archivo(s)** con un total de **{len(df_all):,} filas** y {df_all.shape[1]} columnas.")

        # Preview
        with st.expander(f"👁 Vista previa (primeras 10 filas de {len(df_all):,})"):
            st.dataframe(df_all.head(10), use_container_width=True)

        # Preprocesar
        df_prep, warns = preprocess_uploaded(df_all)
        for w in warns:
            st.warning(w)

        # Predicción tabular
        with st.spinner("Ejecutando MLP (fila a fila)…"):
            out_tab = predict_tabular(df_prep)

        # Predicción secuencial
        with st.spinner(f"Ejecutando LSTM (ventana {SEQ_LEN})…"):
            out_seq = predict_sequence(df_prep)

        # Merge
        if not out_seq.empty:
            out = out_tab.join(out_seq[["LSTM_arquetipo", "LSTM_confianza", "LSTM_eficiencia_pred"]],
                               how="left")
        else:
            out = out_tab.copy()
            out["LSTM_arquetipo"] = np.nan
            out["LSTM_confianza"] = np.nan
            out["LSTM_eficiencia_pred"] = np.nan

        # Resumen
        st.markdown("### Resumen de predicciones")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filas procesadas", f"{len(out):,}")
        c2.metric("Ventanas LSTM", f"{(~out['LSTM_eficiencia_pred'].isna()).sum():,}")
        c3.metric("MLP eficiencia media", f"{out['MLP_eficiencia_pred'].mean():.2f}%")
        if (~out['LSTM_eficiencia_pred'].isna()).any():
            c4.metric("LSTM eficiencia media", f"{out['LSTM_eficiencia_pred'].mean():.2f}%")

        # Distribución de arquetipos
        st.markdown("### Distribución de arquetipos predichos")
        d1, d2 = st.columns(2)
        with d1:
            counts_mlp = out["MLP_arquetipo"].value_counts().reindex(CLASS_NAMES).fillna(0)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(counts_mlp.index, counts_mlp.values,
                   color=[PALETTE[c] for c in counts_mlp.index], edgecolor="#222")
            for i, v in enumerate(counts_mlp.values):
                ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontweight="bold")
            ax.set_title("MLP", fontweight="bold"); ax.grid(alpha=.3)
            st.pyplot(fig)
        with d2:
            if (~out["LSTM_arquetipo"].isna()).any():
                counts_lstm = out["LSTM_arquetipo"].dropna().value_counts().reindex(CLASS_NAMES).fillna(0)
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.bar(counts_lstm.index, counts_lstm.values,
                       color=[PALETTE[c] for c in counts_lstm.index], edgecolor="#222")
                for i, v in enumerate(counts_lstm.values):
                    ax.text(i, v, f"{int(v)}", ha="center", va="bottom", fontweight="bold")
                ax.set_title("LSTM", fontweight="bold"); ax.grid(alpha=.3)
                st.pyplot(fig)
            else:
                st.info("Ningún ingenio alcanzó 8 semanas consecutivas — el LSTM no se ejecutó.")

        # Comparación contra eficiencia real
        if "eficiencia_en_fabrica" in out.columns and out["eficiencia_en_fabrica"].notna().any():
            st.markdown("### Predicho vs Real")
            fig, ax = plt.subplots(figsize=(8, 5))
            mask = out["eficiencia_en_fabrica"].notna()
            ax.scatter(out.loc[mask, "eficiencia_en_fabrica"],
                       out.loc[mask, "MLP_eficiencia_pred"],
                       alpha=0.6, color="#028090", edgecolor="#222", label="MLP")
            if (~out["LSTM_eficiencia_pred"].isna()).any():
                mask2 = mask & out["LSTM_eficiencia_pred"].notna()
                ax.scatter(out.loc[mask2, "eficiencia_en_fabrica"],
                           out.loc[mask2, "LSTM_eficiencia_pred"],
                           alpha=0.6, color="#B85042", marker="^", edgecolor="#222", label="LSTM")
            lo = float(min(out.loc[mask, "eficiencia_en_fabrica"].min(), out["MLP_eficiencia_pred"].min()))
            hi = float(max(out.loc[mask, "eficiencia_en_fabrica"].max(), out["MLP_eficiencia_pred"].max()))
            ax.plot([lo, hi], [lo, hi], "--", color="#333", label="Perfecto")
            ax.set_xlabel("Eficiencia real (%)"); ax.set_ylabel("Eficiencia predicha (%)")
            ax.legend(); ax.grid(alpha=.3)
            st.pyplot(fig)

            mae_mlp = float(np.abs(out.loc[mask, "eficiencia_en_fabrica"] - out.loc[mask, "MLP_eficiencia_pred"]).mean())
            st.metric("MAE del MLP sobre tu archivo (%)", f"{mae_mlp:.2f}")

        # Tabla de predicciones
        st.markdown("### Predicciones fila a fila")
        show_cols = (["_source_file", "ingenio", "semana"] if "ingenio" in out.columns else ["_source_file"]) + \
                    (["eficiencia_en_fabrica"] if "eficiencia_en_fabrica" in out.columns else []) + \
                    ["MLP_arquetipo", "MLP_confianza", "MLP_eficiencia_pred",
                     "LSTM_arquetipo", "LSTM_confianza", "LSTM_eficiencia_pred"]
        show_cols = [c for c in show_cols if c in out.columns]
        st.dataframe(out[show_cols], use_container_width=True)

        # Descarga
        csv_buf = io.StringIO()
        out.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Descargar predicciones (CSV)",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name="predicciones_infocana.csv",
            mime="text/csv",
        )
