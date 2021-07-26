"""
A class for interacting with a series of 
"""
from sklearn.dummy import DummyClassifier

from joblib import load
import numpy as np

from movierep.HostedModel import HostedModel


class IndependentLogisticModel(HostedModel):
    def __init__(self):
        super().__init__("Independent Logistic Models")

    def load(self, saved_checkpoint_path: str):
        saved_data = load(saved_checkpoint_path)
        self.creation_date = saved_data["creation_timestamp"]
        self.models = saved_data["models"]
        self.movie_index = saved_data["index"]
        self.num_models = len(self.models)

    def predict(self, x: np.array, top_k=-1) -> np.array:
        scores = []
        for i, model in enumerate(self.models):
            if type(model) == DummyClassifier:
                continue
            temp_x = np.concatenate((x[:i], x[i + 1 :])).reshape(1, -1)
            prediction = model.decision_function(temp_x)
            scores.append((i + 1, prediction))
        scores.sort(key=lambda x: x[1], reverse=True)

        if top_k > 0:
            return scores[:top_k]
        else:
            return scores[0]

    def __call__(self, x: np.array, top_k=-1) -> np.array:
        return self.predict(x, top_k)
