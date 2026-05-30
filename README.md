# 🚗 MUCars — Prédiction des Prix de Voitures au Maroc

Plateforme de prédiction des prix de voitures d'occasion au Maroc
basée sur le dataset MUCars-2024 (101 896 annonces réelles).

## 📊 Dataset
- **Source** : MUCars-2024 — annonces en ligne au Maroc (2024)
- **Taille** : 101 896 annonces → 74 900 après nettoyage
- **Features** : Marque, Année, Kilométrage, Carburant, Boîte, Condition, Origine...

## 🤖 Modèle
- **Algorithme** : K-Nearest Neighbors (KNN) — Scikit-learn
- **K optimal** : 5 (GridSearchCV)
- **R²** : 0.76
- **MAE** : 27 489 MAD
- **RMSE** : 52 809 MAD

## 🚀 Démo en ligne
👉 [HuggingFace Spaces](https://huggingface.co/spaces/GhizlaneAitelhaj/mucars-predict)

## 🛠️ Stack technique
- Python · Scikit-learn · Pandas · NumPy
- FastAPI · Gradio · Joblib
- GitHub · HuggingFace Spaces