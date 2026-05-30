import gradio as gr
import joblib
import numpy as np

# ── Chargement du modèle
model    = joblib.load("model.pkl")
scaler   = joblib.load("scaler.pkl")
imputer  = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")

# ── Listes des options
BRANDS = sorted([
    "Audi", "BMW", "Citroen", "Dacia", "Fiat", "Ford", "Honda",
    "Hyundai", "Kia", "Mercedes-Benz", "Nissan", "Opel", "Peugeot",
    "Renault", "Seat", "Skoda", "Suzuki", "Toyota", "Volkswagen", "Volvo"
])
FUELS        = ["Diesel", "Petrol", "Hybrid", "Electrique", "LPG"]
GEARBOXES    = ["Manual", "Automatic"]
CONDITIONS   = ["New", "Excellent", "Very Good", "Good", "Fair", "Damaged"]
ORIGINS      = ["WW in Morocco", "Customs-cleared car", "Imported New", "Car not yet customs-cleared"]
FIRST_OWNERS = ["Yes", "No"]

def encode(col, val):
    le = encoders[col]
    if val in le.classes_:
        return le.transform([val])[0]
    return 0

def predict_price(brand, year, mileage, fiscal_power,
                  fuel, gearbox, condition, origin, first_owner, doors):
    try:
        X = np.array([[
            float(year),
            float(mileage),
            float(fiscal_power),
            encode("Fuel",        fuel),
            encode("Gearbox",     gearbox),
            encode("Condition",   condition),
            encode("Origin",      origin),
            encode("First Owner", first_owner),
            encode("Brand",       brand),
            float(doors),
        ]])

        X = imputer.transform(X)
        X = scaler.transform(X)
        price = int(model.predict(X)[0])

        low  = int(price * 0.90)
        high = int(price * 1.10)

        return (
            f"## 💰 Prix estimé : **{price:,} MAD**",
            f"📊 Fourchette : **{low:,} MAD** — **{high:,} MAD**",
            f"ℹ️ Modèle KNN (K=5) — R²=0.76 — entraîné sur 74 900 annonces marocaines"
        )
    except Exception as e:
        return f"Erreur : {str(e)}", "", ""

# ── Interface Gradio
with gr.Blocks(title="MUCars — Prédiction Prix Voitures Maroc", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🚗 MUCars — Prédiction des Prix de Voitures au Maroc
    ### Estimez le prix d'une voiture d'occasion basé sur 74 900 annonces réelles (2024)
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Informations du véhicule")
            brand  = gr.Dropdown(BRANDS,          label="Marque",                value="Renault")
            year   = gr.Slider(1990, 2024,         label="Année",                 value=2018, step=1)
            mileage= gr.Slider(0, 500000,          label="Kilométrage (km)",      value=80000, step=1000)
            fp     = gr.Slider(4, 20,              label="Puissance fiscale (CV)", value=7, step=1)
            doors  = gr.Slider(2, 5,               label="Nombre de portes",      value=5, step=1)

        with gr.Column():
            gr.Markdown("### Caractéristiques")
            fuel        = gr.Dropdown(FUELS,        label="Carburant",              value="Diesel")
            gearbox     = gr.Dropdown(GEARBOXES,    label="Boîte de vitesses",      value="Manual")
            condition   = gr.Dropdown(CONDITIONS,   label="État du véhicule",       value="Good")
            origin      = gr.Dropdown(ORIGINS,      label="Origine",                value="WW in Morocco")
            first_owner = gr.Dropdown(FIRST_OWNERS, label="Premier propriétaire ?", value="No")

    btn = gr.Button("🔍 Estimer le prix", variant="primary", size="lg")

    with gr.Row():
        out_price = gr.Markdown()
    with gr.Row():
        out_range = gr.Markdown()
    with gr.Row():
        out_info  = gr.Markdown()

    btn.click(
        fn=predict_price,
        inputs=[brand, year, mileage, fp, fuel, gearbox, condition, origin, first_owner, doors],
        outputs=[out_price, out_range, out_info]
    )

    gr.Markdown("""
    ---
    **Dataset** : MUCars-2024 — 101 896 annonces de voitures d'occasion au Maroc  
    **Modèle** : K-Nearest Neighbors (KNN) — Scikit-learn  
    **Métriques** : R²=0.76 · MAE=27 489 MAD · RMSE=52 809 MAD
    """)

demo.launch()