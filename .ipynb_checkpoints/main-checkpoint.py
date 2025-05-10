import json

import dill

import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

with open('c:/Users/A315-23-R7CZ/ds-intro/ds-intro/31_model_as_api/module31_homework/model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)

class Form(BaseModel):
    fuel: str
    id: int
    image_url: str
    lat: float
    long: float
    manufacturer: str
    model: str
    odometer: int
    posting_date: str
    price: int
    region: str
    region_url: str
    state: str
    title_status: str
    transmission: str
    url: str
    year: int

class Prediction(BaseModel):
    id: int
    price_category: str


@app.get('/status')

def status():
    return "I am Ok!"

@app.get('/version')

def version():
    return model['metadata']
#
#
#
@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'id': form.id,
        'price_category': y[0]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, debug=True)
