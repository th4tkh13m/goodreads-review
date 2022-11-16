from fastai.text.all import *
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Reviews to Ratings")

class Review(BaseModel):
    review: str

@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file

    global clf
    clf = load_learner("model/model.pkl")


@app.post("/predict")
def predict(review: Review):
    print(review)
    

    pred = clf.predict(review.review)
    print(clf)
    print(pred)
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}