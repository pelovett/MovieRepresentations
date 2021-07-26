from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from joblib import dump
import numpy as np
import pandas as pd

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


def calc_projection_matrix(main_matrix, k):
    matrix = PCA(n_components=k).fit(main_matrix)
    explained = matrix.explained_variance_ratio_
    logging.info(f"Explained variance ratios for first 50: {explained[:50]}...")


def load_dataframe(data_dir):
    global RANDOM_SEED
    # Parse training data
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"), sep="\t")
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
    projection_matrix = calc_projection_matrix(
        data_frame["cust_id"].to_numpy(), k=params["training"]["pca"]["k"]
    )

    # Save models to checkpoint file
    save_dir = sys.argv[2]
    vers_num = 1
    save_file_name = f"pca_matrix_{vers_num}"
    while os.path.isfile(os.path.join(save_dir, save_file_name)):
        vers_num += 1
        save_file_name = f"pca_matrix__{vers_num}"

    logging.info(f"Saving matrix to file: {os.path.join(save_dir, save_file_name)}")
    model_dict = dict()
    model_dict["creation_timestamp"] = str(datetime.now())
    model_dict["pca_matrix"] = projection_matrix
    model_dict["index"] = dict()
    with open("data/raw/movie_titles.csv", "r", encoding="ISO-8859-1") as in_file:
        for line in in_file:
            cur = line.strip().split(",")
            model_dict["index"][int(cur[0]) + 1] = (cur[1], cur[2])
    dump(model_dict, os.path.join(save_dir, save_file_name))
    logging.info("Saving complete. Exiting...")
