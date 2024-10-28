import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = FastAPI()

sample_data = {
    "gamma": np.array([ 21.3846,  10.917 ,   2.6161,   0.5857,   0.3934,  15.2618,
        11.5245,   2.8766,   2.4229, 106.8258]),
    "hedron": np.array([ 1.205135e+02,  7.690180e+01,  3.993900e+00,  9.440000e-02,
        6.830000e-02,  5.804300e+00, -9.352240e+01, -6.383890e+01,
        8.468740e+01,  4.083166e+02])
}

class InputData(BaseModel):
    features: list[float] 

@app.post("/predict")
async def predict(input: InputData):
    features = np.array(input.features).reshape(1, -1)
    features = scaler.transform(features)
    
    prediction = model.predict(features)
    prediction = prediction.tolist()[0]

    if prediction == 0:
        prediction = "Hedron"
    else:
        prediction = "Gamma"

    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)