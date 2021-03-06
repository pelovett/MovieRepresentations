from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from joblib import dump
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import logging
import sys
import os
import yaml

RANDOM_SEED = None


def convert_to_int_list(dataframe: pd.Series) -> "list[list[int]]":
    """
    Takes a dataframe with a string representation of a list of ints
    and converts that into a list of lists of ints
    """
    result_list = []
    for row in dataframe:
        result_list.append([int(x) for x in row[1:-1].split(", ")])
    return result_list


def create_predictor(main_matrix, index):
    # Convert to sparse matrix to save memory
    x = csr_matrix(
        np.concatenate([main_matrix[:, :index], main_matrix[:, index + 1 :]], axis=1)
    )
    y = main_matrix[:, index]
    if y.sum() < 15:
        model = DummyClassifier(strategy="most_frequent").fit(x, y)
        model_name = "dummy classifier"
    else:
        model = SGDClassifier(loss="log", random_state=RANDOM_SEED).fit(x, y)
        model_name = "logistic regression"
    score = f1_score(y, model.predict(x), zero_division=1)
    logging.info(f"Model {index} using {model_name} | f1 score: {score:.4}")
    return (index, model)


def load_dataframe(data_dir):
    global RANDOM_SEED
    # Parse training data
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"), sep="\t")
    train_cust = train_df["cust_id"].to_numpy()
    temp_train_x = convert_to_int_list(train_df["movies"])
    max_movie_id = max([max(_) for _ in temp_train_x])
    logging.info(f"Data set size: {(len(temp_train_x), max_movie_id+1)}")
    train_x = np.zeros((len(temp_train_x), max_movie_id + 1), dtype=np.int16)
    for i, row in enumerate(temp_train_x):
        for movie_id in row:
            train_x[i][movie_id] = 1
    logging.info("Finished instantiating matrix")
    return train_x


if __name__ == "__main__":
    with open("params.yaml", "r") as raw_yaml:
        params = yaml.safe_load(raw_yaml)
    RANDOM_SEED = params["training"]["seed"]
    logging.basicConfig(
        format="%(asctime)-15s | %(message)s",
        level=logging.DEBUG,
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    data_frame = load_dataframe(sys.argv[1])

    models = []
    thread_func = lambda x: create_predictor(data_frame, x)
    with ThreadPoolExecutor() as thread_pool:
        logging.info("Beginning training process")
        models = thread_pool.map(
            thread_func,
            range(data_frame.shape[1]),
        )
        models = list(models)
        models.sort(key=lambda x: x[0])

    # Save models to checkpoint file
    save_dir = sys.argv[2]
    vers_num = 1
    save_file_name = f"logistic_regression_models_{vers_num}"
    while os.path.isfile(os.path.join(save_dir, save_file_name)):
        vers_num += 1
        save_file_name = f"logistic_regression_models_{vers_num}"

    logging.info(f"Saving models to file: {os.path.join(save_dir, save_file_name)}")
    model_dict = dict()
    model_dict["creation_timestamp"] = str(datetime.now())
    model_dict["models"] = [x for (_, x) in models]
    model_dict["index"] = dict()
    with open("data/raw/movie_titles.csv", "r", encoding="ISO-8859-1") as in_file:
        for line in in_file:
            cur = line.strip().split(",")
            model_dict["index"][int(cur[0]) + 1] = (cur[1], cur[2])
    dump(model_dict, os.path.join(save_dir, save_file_name))
    logging.info("Saving complete. Exiting...")
