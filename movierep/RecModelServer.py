from movierep.IndependentLogisticModel import IndependentLogisticModel
import sys
from os import path, environ
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
from pydantic import BaseModel

# Add parent to path so that we can import modules
sys.path.append("..")


app = FastAPI()

origins = ["https://localhost:3000", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = IndependentLogisticModel()
model.load(environ["MODEL_CHECKPOINT"])


class SuggestionRequest(BaseModel):
    user_id: str
    liked_movie_ids: List[int]
    num_suggestions: int = 5


@app.get("/")
def read_root():
    return {"documentation": "TODO return basic functionality"}


@app.post("/suggestmovies/")
def suggest_movies(request: SuggestionRequest):
    # TODO model must see all movies len(model.movie_index))
    movie_ids = np.zeros(4500)
    for movie_id in request.liked_movie_ids:
        movie_ids[movie_id] = 1
    suggestions = model(movie_ids, top_k=request.num_suggestions)
    print(suggestions)
    return [[x, y.item()] for (x, y) in suggestions]
