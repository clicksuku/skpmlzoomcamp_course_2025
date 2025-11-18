import pickle
import pandas as pd
from fastapi import FastAPI
from FilmFeature import FilmFeature

app = FastAPI()

with open('_models/classification_model.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    model_classification, dv_classification = pickle.load(f_in)

with open('_models/regression_model.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    model_regression = pickle.load(f_in)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict_revenue")
async def predict_revenue(request: FilmFeature):
    x = pd.DataFrame([request.dict()])
    y_pred = model_regression.predict(x)
    return {"revenue": float(y_pred)}


@app.post("/predict_hit")
async def predict_hit(request: FilmFeature):
    x = dv_classification.transform([request.dict()])
    y_pred = model_classification.predict_proba(x)[0, 1]
    #y_pred_status = model_classification.predict(x)
    return {"probability": float(y_pred)}