import numpy as np

from random import randint
import sys

from movierep.IndependentLogisticModel import IndependentLogisticModel


def main():
    models = IndependentLogisticModel()
    models.load(sys.argv[1])
    n = models.num_models
    x = np.zeros(n)
    for i in range(100):
        rand = randint(1, 4499)
        x[rand] = 1
        print(f"User likes: {models.model_index[rand]}")

    result = models.predict(x, top_k=5)

    print(f"Recommending:")
    for i, res in enumerate(result):
        print(f"{i+1}: {models.model_index[res[0]]}")


if __name__ == "__main__":
    main()
