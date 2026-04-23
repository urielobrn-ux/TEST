"""Genera las figuras de EDA del notebook."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(DATA, "infocana_processed.csv"))
profile = pd.read_csv(os.path.join(DATA, "ingenios_profile_clusters.csv"))

PALETTE = {"Gigantes": "#B85042", "Eficientes": "#028090", "Especialistas": "#6D2E46"}

# ======================================================================
# FIG 1 — Distribución de clases (del target supervisado = arquetipo M3)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 1a. Conteo por ingenio (único)
counts_ing = profile["cluster_name"].value_counts().reindex(["Gigantes", "Eficientes", "Especialistas"])
axes[0].bar(counts_ing.index, counts_ing.values,
            color=[PALETTE[x] for x in counts_ing.index], edgecolor="#222", linewidth=0.8)
for i, v in enumerate(counts_ing.values):
    axes[0].text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold")
axes[0].set_title("Ingenios por arquetipo (N=50)", fontweight="bold")
axes[0].set_ylabel("Nº de ingenios")
axes[0].set_ylim(0, counts_ing.max() * 1.2)

# 1b. Conteo por observación semanal
counts_obs = df["cluster_name"].value_counts().reindex(["Gigantes", "Eficientes", "Especialistas"])
axes[1].bar(counts_obs.index, counts_obs.values,
            color=[PALETTE[x] for x in counts_obs.index], edgecolor="#222", linewidth=0.8)
for i, v in enumerate(counts_obs.values):
    axes[1].text(i, v + 10, str(int(v)), ha="center", fontweight="bold")
axes[1].set_title(f"Observaciones semanales por arquetipo (N={len(df)})", fontweight="bold")
axes[1].set_ylabel("Nº de registros")

plt.suptitle("Distribución de clases — Etiquetas supervisadas del Módulo 3", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "01_distribucion_clases.png"), bbox_inches="tight")
plt.close()

# ======================================================================
# FIG 2 — Ejemplos de datos: distribución de la eficiencia por arquetipo
# ======================================================================
fig, ax = plt.subplots(figsize=(10, 5))
for name in ["Gigantes", "Eficientes", "Especialistas"]:
    vals = df.loc[df["cluster_name"] == name, "eficiencia_en_fabrica"]
    ax.hist(vals, bins=25, alpha=0.65, label=f"{name} (n={len(vals)})",
            color=PALETTE[name], edgecolor="white")
ax.axvline(df["eficiencia_en_fabrica"].mean(), color="#333", linestyle="--", lw=1.2,
           label=f"Media global = {df['eficiencia_en_fabrica'].mean():.2f}%")
ax.set_title("Distribución de eficiencia de fábrica por arquetipo", fontweight="bold")
ax.set_xlabel("Eficiencia de fábrica (%)")
ax.set_ylabel("Frecuencia (semanas)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "02_eficiencia_por_arquetipo.png"), bbox_inches="tight")
plt.close()

# ======================================================================
# FIG 3 — Estadísticas descriptivas: boxplots de las variables clave por arquetipo
# ======================================================================
key_vars = [
    ("cana_molida_neta",          "Caña molida neta (ton)"),
    ("azucar_producida_total",    "Azúcar producida total"),
    ("perdidas_totales",          "Pérdidas totales"),
    ("extraccion_de_jugo_mezclado_en_cana", "Extracción de jugo mezclado"),
    ("pol_en_cana",               "POL en caña"),
    ("eficiencia_en_fabrica",     "Eficiencia fábrica (%)"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()
for i, (col, label) in enumerate(key_vars):
    data = [df.loc[df["cluster_name"] == n, col].values for n in ["Gigantes", "Eficientes", "Especialistas"]]
    bp = axes[i].boxplot(data, labels=["Gig.", "Efic.", "Esp."], patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
    for patch, name in zip(bp["boxes"], ["Gigantes", "Eficientes", "Especialistas"]):
        patch.set_facecolor(PALETTE[name]); patch.set_alpha(0.85)
    axes[i].set_title(label, fontweight="bold", fontsize=10)
    axes[i].tick_params(axis="x", labelsize=9)
plt.suptitle("Estadísticas descriptivas de variables clave por arquetipo", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "03_boxplots_por_arquetipo.png"), bbox_inches="tight")
plt.close()

# ======================================================================
# FIG 4 — Matriz de correlación con la variable objetivo (eficiencia)
# ======================================================================
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in ["class_id", "semana"]]
corr_vs_eff = df[num_cols].corr()["eficiencia_en_fabrica"].drop("eficiencia_en_fabrica")
corr_vs_eff = corr_vs_eff.reindex(corr_vs_eff.abs().sort_values(ascending=False).index).head(15)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#028090" if v > 0 else "#B85042" for v in corr_vs_eff.values]
ax.barh(corr_vs_eff.index[::-1], corr_vs_eff.values[::-1], color=colors[::-1], edgecolor="#222", linewidth=0.6)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 15 variables correlacionadas con eficiencia_en_fabrica", fontweight="bold")
ax.set_xlabel("Coeficiente de correlación (Pearson)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "04_correlacion_target.png"), bbox_inches="tight")
plt.close()

# ======================================================================
# FIG 5 — Evolución temporal de la eficiencia promedio por arquetipo (semanas)
# ======================================================================
fig, ax = plt.subplots(figsize=(11, 4.5))
for name in ["Gigantes", "Eficientes", "Especialistas"]:
    sub = df[df["cluster_name"] == name].groupby("semana")["eficiencia_en_fabrica"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=name, color=PALETTE[name], linewidth=2)
ax.set_title("Evolución de la eficiencia media por arquetipo durante la zafra", fontweight="bold")
ax.set_xlabel("Semana de zafra"); ax.set_ylabel("Eficiencia fábrica (%)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "05_evolucion_temporal.png"), bbox_inches="tight")
plt.close()

print("Figuras EDA generadas:")
for f in sorted(os.listdir(FIG)):
    print("  -", f)
