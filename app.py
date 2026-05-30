from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, numpy as np

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

model    = joblib.load("model.pkl")
scaler   = joblib.load("scaler.pkl")
imputer  = joblib.load("imputer.pkl")
encoders = joblib.load("encoders.pkl")

class CarInput(BaseModel):
    brand: str; year: int; mileage: float; fiscal_power: float
    fuel: str; gearbox: str; condition: str

@app.post("/predict")
def predict(car: CarInput):
    enc = lambda col, val: encoders[col].transform([val])[0]
    X = np.array([[car.year, car.mileage, car.fiscal_power,
                   enc("Fuel", car.fuel), enc("Gearbox", car.gearbox),
                   enc("Condition", car.condition), 0, 0, enc("Brand", car.brand), 4]])
    X = scaler.transform(imputer.transform(X))
    price = int(model.predict(X)[0])
    return {"price": price, "currency": "MAD"}