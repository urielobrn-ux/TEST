# build_dataset.py
# Prepara los datos para el M4. Parte del CSV original del ingenio que
# usamos en el M3 (infocana 14-15). Hace tres cosas:
#   1. convierte los tiempos perdidos de HH:MM:SS a horas decimales
#   2. rehace los clusters K-Means del M3 (3 grupos: gigantes, eficientes,
#      especialistas) sobre los perfiles promedio por ingenio. Lo rehago
#      aca para que el pipeline sea autocontenido y no tengamos que volver
#      al notebook del M3.
#   3. escribe los CSVs finales en /data
#
# nota: el nombre del archivo original tiene parentesis y los acentos
# del encoding latin-1, por eso leo con encoding="latin-1". Si alguien
# quiere rehacer esto, cuidado con eso.


import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

SRC = os.path.join("data", "infocana(14-15)(resumen).csv")

df = pd.read_csv(SRC, encoding="latin-1")
print("Loaded:", df.shape)

# --- 1. Convert HH:MM:SS columns to hours (float)
time_cols = [c for c in df.columns if c.startswith("tiempo_perdido")]

def hms_to_hours(s):
    if pd.isna(s):
        return 0.0
    s = str(s).strip()
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return 0.0
    parts = s.split(":")
    try:
        h = float(parts[0]); m = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        return h + m/60.0 + sec/3600.0
    except Exception:
        return 0.0

for c in time_cols:
    df[c + "_h"] = df[c].apply(hms_to_hours)

# Drop text time cols and zafra (constant)
df = df.drop(columns=time_cols + ["zafra"])

# --- 2. Clean: drop rows where efficiency == 0 (record not reported)
df = df[df["eficiencia_en_fabrica"] > 0].reset_index(drop=True)
print("After cleaning zeros in efficiency:", df.shape)

# --- 3. Feature columns (exclude id + target + cluster label)
id_cols = ["ingenio", "semana"]
target_reg = "eficiencia_en_fabrica"

feature_cols = [c for c in df.columns if c not in id_cols + [target_reg]]
print("N features:", len(feature_cols))

# Replace any remaining NaN/Inf with column medians
X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan)
X_df = X_df.fillna(X_df.median(numeric_only=True))
df[feature_cols] = X_df

# --- 4. Create per-ingenio aggregated profile for clustering (Module 3 logic)
profile = df.groupby("ingenio")[feature_cols + [target_reg]].mean()

# K-Means from scratch (3 clusters, as per Module 3)
def kmeans(X, k=3, iters=200, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    # k-means++ init
    centers = [X[rng.integers(0, n)]]
    for _ in range(k - 1):
        d = np.min(np.sum((X[:, None] - np.array(centers)[None, :])**2, axis=2), axis=1)
        probs = d / d.sum()
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers, dtype=float)
    for _ in range(iters):
        d = np.sum((X[:, None] - C[None, :])**2, axis=2)
        labels = np.argmin(d, axis=1)
        new_C = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else C[j] for j in range(k)])
        if np.allclose(new_C, C):
            break
        C = new_C
    return labels, C

# Standardize profile
P = profile.values.astype(float)
P_mean = P.mean(axis=0); P_std = P.std(axis=0) + 1e-9
P_scaled = (P - P_mean) / P_std

labels, centers = kmeans(P_scaled, k=3, iters=300, seed=42)

# Map clusters to archetypes using domain logic:
# - "Gigantes" = highest cana_molida_neta avg
# - "Especialistas" = lowest cana_molida_neta
# - "Eficientes" = the rest, but typically highest efficiency
cana_col = feature_cols.index("cana_molida_neta")
eff_col = len(feature_cols)  # target_reg is last in profile columns

mean_cana_by_c = [profile.values[labels == c, cana_col].mean() for c in range(3)]
mean_eff_by_c  = [profile.values[labels == c, eff_col].mean() for c in range(3)]

# Gigantes = argmax of cana
gigantes = int(np.argmax(mean_cana_by_c))
# Especialistas = argmin of cana
especialistas = int(np.argmin(mean_cana_by_c))
# Eficientes = remaining
eficientes = [c for c in range(3) if c not in (gigantes, especialistas)][0]

cluster_name_map = {gigantes: "Gigantes", eficientes: "Eficientes", especialistas: "Especialistas"}
# Stable 0/1/2 mapping: 0=Gigantes, 1=Eficientes, 2=Especialistas
class_id_map = {gigantes: 0, eficientes: 1, especialistas: 2}

profile["cluster"] = labels
profile["cluster_name"] = [cluster_name_map[l] for l in labels]
profile["class_id"] = [class_id_map[l] for l in labels]

# Attach cluster label to weekly rows
ingenio_to_class = profile["class_id"].to_dict()
df["class_id"] = df["ingenio"].map(ingenio_to_class)
df["cluster_name"] = df["ingenio"].map(profile["cluster_name"].to_dict())

print("\nArquetipos (conteos por ingenio):")
print(profile["cluster_name"].value_counts())

print("\nMétricas promedio por arquetipo (muestra):")
summary = profile.groupby("cluster_name")[
    ["cana_molida_neta", "azucar_producida_total", "perdidas_totales", "eficiencia_en_fabrica"]
].mean().round(2)
print(summary)

# --- 5. Save datasets
df.to_csv(os.path.join(DATA_DIR, "infocana_processed.csv"), index=False)
profile.reset_index().to_csv(os.path.join(DATA_DIR, "ingenios_profile_clusters.csv"), index=False)

# Also save meta: feature names order, means/stds for later scaling in dashboard
meta = {
    "feature_cols": feature_cols,
    "feature_mean": df[feature_cols].mean().to_dict(),
    "feature_std": (df[feature_cols].std() + 1e-9).to_dict(),
    "target_mean": float(df[target_reg].mean()),
    "target_std": float(df[target_reg].std() + 1e-9),
    "class_names": ["Gigantes", "Eficientes", "Especialistas"],
}