"""
MUCars-2024 — Prédiction des Prix de Voitures Marocaines
=========================================================
Dataset réel : 101 896 annonces — Maroc 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import joblib, json, warnings
warnings.filterwarnings("ignore")

PALETTE = {
    "bg": "#0D1117", "card": "#161B22",
    "accent": "#C9A84C", "blue": "#4A90D9",
    "green": "#3FB950", "red": "#F85149",
    "text": "#C9D1D9", "muted": "#484F58", "orange": "#E06C1F",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"], "axes.facecolor": PALETTE["card"],
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["text"],
    "xtick.color": PALETTE["text"], "ytick.color": PALETTE["text"],
    "text.color": PALETTE["text"], "grid.color": PALETTE["muted"],
    "grid.alpha": 0.3, "font.family": "monospace",
})

print("=" * 65)
print("  MUCars-2024 — PRÉDICTION DES PRIX DE VOITURES MAROCAINES")
print("=" * 65)

# ── 1. CHARGEMENT
df = pd.read_csv("cars_dataframe.csv", dtype=str)
print(f"\n[1] Données brutes : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ── 2. NETTOYAGE
print("\n[2] NETTOYAGE")

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df = df.dropna(subset=["Price"])
print(f"    Après suppression NaN prix : {len(df):,}")

q01, q99 = df["Price"].quantile([0.01, 0.99])
df = df[(df["Price"] >= q01) & (df["Price"] <= q99)]
print(f"    Après filtre prix [{q01:,.0f} — {q99:,.0f} MAD] : {len(df):,}")

df["Year"] = pd.to_numeric(df["Year"].str.extract(r"(\d{4})")[0], errors="coerce")

def parse_mileage(m):
    try:
        a, b = str(m).split(" - ")
        return (int(a.replace(" ", "")) + int(b.replace(" ", ""))) / 2
    except:
        return np.nan

df = df.copy()
df["Mileage_num"]     = df["Mileage"].apply(parse_mileage)
df["FP_num"]          = pd.to_numeric(df["Fiscal Power"].str.extract(r"(\d+)")[0], errors="coerce")
df["Number of Doors"] = pd.to_numeric(df["Number of Doors"], errors="coerce")

df = df[df["Fuel"].isin(["Diesel", "Petrol", "Hybrid", "Electrique", "LPG"])]
print(f"    Après nettoyage Fuel : {len(df):,}")

# ── 3. FEATURE ENGINEERING
print("\n[3] FEATURE ENGINEERING")

cat_cols = ["Fuel", "Gearbox", "Condition", "Origin", "First Owner", "Brand"]

# ✅ encoders sauvegardé ici comme dictionnaire
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str).fillna("Unknown"))
    encoders[col] = le  # ← on garde chaque LabelEncoder

features = [
    "Year", "Mileage_num", "FP_num",
    "Fuel_enc", "Gearbox_enc", "Condition_enc",
    "Origin_enc", "First Owner_enc", "Brand_enc",
    "Number of Doors",
]

X = df[features].astype(float).values
y = df["Price"].astype(float).values

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

print(f"    Features : {len(features)}")
print(f"    Dataset final : {len(X):,} lignes")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"    Train : {len(X_train):,} | Test : {len(X_test):,}")

# ── 4. OPTIMISATION DU K
print("\n[4] OPTIMISATION DU K")
sample = min(20_000, len(X_train))
idx    = np.random.RandomState(42).choice(len(X_train), sample, replace=False)
Xs, ys = X_train_s[idx], y_train[idx]

param_grid = {"n_neighbors": list(range(1, 26, 2))}
grid = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5,
                    scoring="neg_mean_absolute_error", n_jobs=-1)
grid.fit(Xs, ys)
best_k = grid.best_params_["n_neighbors"]
cv_mae = -grid.cv_results_["mean_test_score"]
k_list = [p["n_neighbors"] for p in grid.cv_results_["params"]]
print(f"    Meilleur K : {best_k}")

# ── 5. MODÈLE FINAL
print(f"\n[5] ENTRAÎNEMENT FINAL (K={best_k})")
model = KNeighborsRegressor(n_neighbors=best_k, n_jobs=-1)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
cv_r2 = cross_val_score(
    KNeighborsRegressor(n_neighbors=best_k, n_jobs=-1),
    X_train_s[:10_000], y_train[:10_000], cv=5, scoring="r2"
).mean()

print(f"    MAE   : {mae:>12,.0f} MAD")
print(f"    RMSE  : {rmse:>12,.0f} MAD")
print(f"    R²    : {r2:>12.4f}")
print(f"    MAPE  : {mape:>12.2f}%")
print(f"    CV R² : {cv_r2:>12.4f}")

# ── 6. VISUALISATIONS
print("\n[6] VISUALISATIONS...")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor(PALETTE["bg"])
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.38)

fig.text(0.5, 0.975, "MUCars-2024 — PRÉDICTION DES PRIX — KNN",
         ha="center", fontsize=17, fontweight="bold", color=PALETTE["accent"])
fig.text(0.5, 0.950,
         f"K={best_k}  •  R²={r2:.3f}  •  MAE={mae:,.0f} MAD  •  RMSE={rmse:,.0f} MAD  •  MAPE={mape:.1f}%",
         ha="center", fontsize=10, color=PALETTE["text"])

ax1 = fig.add_subplot(gs[0, 0])
bins = np.logspace(np.log10(df["Price"].min()+1), np.log10(df["Price"].max()), 60)
ax1.hist(df["Price"], bins=bins, color=PALETTE["accent"], alpha=0.85, edgecolor=PALETTE["bg"])
ax1.set_xscale("log")
ax1.set_title("Distribution des prix (log)", color=PALETTE["accent"], fontsize=11, pad=8)
ax1.set_xlabel("Prix (MAD)"); ax1.set_ylabel("Fréquence")
med = df["Price"].median()
ax1.axvline(med, color=PALETTE["blue"], lw=2, linestyle="--", label=f"Médiane : {med:,.0f}")
ax1.legend(fontsize=8); ax1.grid(axis="y")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(k_list, cv_mae, color=PALETTE["blue"], lw=2, marker="o",
         markersize=5, markerfacecolor=PALETTE["accent"])
ax2.axvline(best_k, color=PALETTE["accent"], lw=2, linestyle="--", label=f"K optimal = {best_k}")
ax2.set_title("MAE par K (GridSearchCV)", color=PALETTE["accent"], fontsize=11, pad=8)
ax2.set_xlabel("K"); ax2.set_ylabel("MAE (MAD)")
ax2.legend(fontsize=8); ax2.grid()
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

ax3 = fig.add_subplot(gs[0, 2])
sidx = np.random.RandomState(0).choice(len(y_test), min(2000, len(y_test)), replace=False)
ax3.scatter(y_test[sidx], y_pred[sidx], alpha=0.3, s=12, color=PALETTE["blue"], edgecolors="none")
lim = [0, float(q99) * 1.05]
ax3.plot(lim, lim, "--", color=PALETTE["accent"], lw=1.5, label="Parfait")
ax3.set_title("Réel vs Prédit (échantillon)", color=PALETTE["accent"], fontsize=11, pad=8)
ax3.set_xlabel("Prix réel (MAD)"); ax3.set_ylabel("Prix prédit (MAD)")
ax3.set_xlim(lim); ax3.set_ylim(lim)
ax3.legend(fontsize=8); ax3.grid()
ax3.text(0.05, 0.92, f"R² = {r2:.3f}", transform=ax3.transAxes,
         fontsize=10, color=PALETTE["green"], fontweight="bold")
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

ax4 = fig.add_subplot(gs[1, 0])
res = y_test[sidx] - y_pred[sidx]
ax4.scatter(y_pred[sidx], res, alpha=0.3, s=12, color=PALETTE["green"], edgecolors="none")
ax4.axhline(0, color=PALETTE["accent"], lw=1.5, linestyle="--")
ax4.set_title("Résidus", color=PALETTE["accent"], fontsize=11, pad=8)
ax4.set_xlabel("Prix prédit (MAD)"); ax4.set_ylabel("Résidu (MAD)")
ax4.grid()
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

ax5 = fig.add_subplot(gs[1, 1])
top12 = df["Brand"].value_counts().head(12)
ax5.barh(top12.index[::-1], top12.values[::-1], color=PALETTE["blue"],
         alpha=0.85, edgecolor=PALETTE["bg"])
ax5.set_title("Top 12 marques (annonces)", color=PALETTE["accent"], fontsize=11, pad=8)
ax5.set_xlabel("Nombre d'annonces"); ax5.grid(axis="x")

ax6 = fig.add_subplot(gs[1, 2])
brand_price = df.groupby("Brand")["Price"].median().sort_values(ascending=False).head(12)
ax6.barh(brand_price.index[::-1], brand_price.values[::-1],
         color=[PALETTE["accent"] if v > 200_000 else PALETTE["blue"] for v in brand_price.values[::-1]],
         alpha=0.85, edgecolor=PALETTE["bg"])
ax6.set_title("Prix médian marque (top 12)", color=PALETTE["accent"], fontsize=11, pad=8)
ax6.set_xlabel("Prix médian (MAD)"); ax6.grid(axis="x")
ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

ax7 = fig.add_subplot(gs[2, 0])
yr = df.groupby("Year")["Price"].median().reset_index()
yr = yr[(yr["Year"] >= 1990) & (yr["Year"] <= 2024)]
ax7.plot(yr["Year"], yr["Price"], color=PALETTE["accent"], lw=2, marker="o", markersize=4)
ax7.fill_between(yr["Year"], yr["Price"], alpha=0.12, color=PALETTE["accent"])
ax7.set_title("Prix médian par année", color=PALETTE["accent"], fontsize=11, pad=8)
ax7.set_xlabel("Année"); ax7.set_ylabel("Prix médian (MAD)")
ax7.grid()
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

ax8 = fig.add_subplot(gs[2, 1])
ax8.set_facecolor(PALETTE["bg"]); ax8.axis("off")
metrics_list = [
    ("MAE",       f"{mae:,.0f} MAD",  PALETTE["blue"]),
    ("RMSE",      f"{rmse:,.0f} MAD", PALETTE["blue"]),
    ("R²",        f"{r2:.4f}",         PALETTE["green"]),
    ("MAPE",      f"{mape:.2f}%",      PALETTE["green"]),
    ("CV R²",     f"{cv_r2:.4f}",      PALETTE["accent"]),
    ("K optimal", str(best_k),          PALETTE["accent"]),
    ("Train",     f"{len(X_train):,}", PALETTE["muted"]),
    ("Test",      f"{len(X_test):,}",  PALETTE["muted"]),
]
ax8.text(0.5, 1.02, "MÉTRIQUES DU MODÈLE", ha="center", va="top",
         fontsize=12, color=PALETTE["accent"], fontweight="bold", transform=ax8.transAxes)
for i, (label, value, color) in enumerate(metrics_list):
    ax8.text(0.10, 0.86 - i * 0.12, label, transform=ax8.transAxes,
             fontsize=9.5, color=PALETTE["muted"])
    ax8.text(0.75, 0.86 - i * 0.12, value, transform=ax8.transAxes,
             fontsize=10, color=color, fontweight="bold", ha="right")

ax9 = fig.add_subplot(gs[2, 2])
fuel_p = df.groupby("Fuel")["Price"].median().sort_values()
fc = {"Diesel": PALETTE["blue"], "Petrol": PALETTE["orange"],
      "Hybrid": PALETTE["green"], "Electrique": PALETTE["accent"], "LPG": PALETTE["muted"]}
bars9 = ax9.barh(fuel_p.index, fuel_p.values,
                 color=[fc.get(f, PALETTE["blue"]) for f in fuel_p.index],
                 alpha=0.85, edgecolor=PALETTE["bg"])
ax9.set_title("Prix médian par carburant", color=PALETTE["accent"], fontsize=11, pad=8)
ax9.set_xlabel("Prix médian (MAD)"); ax9.grid(axis="x")
ax9.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
for bar, val in zip(bars9, fuel_p.values):
    ax9.text(val + 1000, bar.get_y() + bar.get_height()/2,
             f"{val/1000:.0f}k", va="center", fontsize=8, color=PALETTE["text"])

plt.savefig("mucars_knn_results.png", dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print("    ✓ mucars_knn_results.png")

# ── 7. EXPORT CSV
out = pd.DataFrame({
    "Prix_Reel_MAD":   y_test.astype(int),
    "Prix_Predit_MAD": y_pred.astype(int),
    "Residus_MAD":     (y_test - y_pred).astype(int),
    "Erreur_Pct":      (np.abs((y_test - y_pred) / (y_test + 1e-9)) * 100).round(2),
})
out.to_csv("mucars_predictions.csv", index=False)
print("    ✓ mucars_predictions.csv")

print("\n" + "=" * 65)
print(f"  TERMINÉ  |  R²={r2:.4f}  MAE={mae:,.0f} MAD  K={best_k}")
print("=" * 65)

stats = {"mae": round(mae), "rmse": round(rmse), "r2": round(r2, 4),
         "mape": round(mape, 2), "cv_r2": round(cv_r2, 4),
         "best_k": int(best_k), "n_train": len(X_train), "n_test": len(X_test),
         "total": len(df), "median_price": int(med)}
print("STATS_JSON:", json.dumps(stats))

# ── 8. SAUVEGARDE DU MODÈLE ✅
joblib.dump(model,    "model.pkl")
joblib.dump(scaler,   "scaler.pkl")
joblib.dump(imputer,  "imputer.pkl")
joblib.dump(encoders, "encoders.pkl")   # ← dictionnaire complet des LabelEncoders
print("\n✓ model.pkl")
print("✓ scaler.pkl")
print("✓ imputer.pkl")
print("✓ encoders.pkl")
print("\n  Modèle prêt pour le déploiement !")