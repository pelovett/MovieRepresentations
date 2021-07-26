# Take in a tsv file formatted as:
#   movie_id, cust_id, rating, datestring
#
# split the data according to the passed in split sizes. Then group by users
# creating lines formatted as:
#   cust_id, movie_id_0, movie_id_1, ..., movie_id_n
#
# Where movies are only included if their score is above the passed
# in threshold.
from numpy.lib.utils import source
import pandas as pd
from sklearn.model_selection import train_test_split

import glob
import sys
import yaml


def split_transform_file(path_to_file: str, prepare_params: dict):
    train_size = prepare_params["train_split"]
    val_size = prepare_params["val_split"]
    random_seed = prepare_params["seed"]
    min_rating = 2
    assert train_size > 0.0 and val_size > 0.0
    assert train_size + val_size < 1.0
    assert min_rating > 0 and min_rating < 5

    collection = []
    for data_file_name in glob.glob(path_to_file + "/data_*.csv"):
        source_df = pd.read_csv(
            data_file_name,
            delimiter="\t",
            names=["movie_id", "cust_id", "rating", "date"],
        )

        # Remove films rated below threshold
        source_df = source_df[source_df["rating"] > min_rating]

        # Remove date and rating information
        source_df = source_df.drop(["rating", "date"], axis=1)

        # Calculate max value for movie id
        num_movies = source_df["movie_id"].nunique()

        # Group by customer so that we get all the movies a customer liked
        source_df = (
            source_df.groupby("cust_id")["movie_id"]
            .apply(list)
            .reset_index(name="movies")
        )
        collection.append(source_df)
    source_df = pd.concat(collection)

    print(f"Number of filtered customer records: {len(source_df)}")
    print(f"Mean number of movies rated: {source_df['movies'].apply(len).mean()}")
    print(f"Median number of movies rated: {source_df['movies'].apply(len).median()}")
    print(f"Number of movies: {num_movies}")

    # Split data into groups for model development
    train_data, val_data = train_test_split(
        source_df,
        test_size=round(1.0 - train_size, 4),
        train_size=train_size,
        random_state=random_seed,
    )
    val_split_size = val_size / (1.0 - train_size)
    val_data, test_data = train_test_split(
        val_data,
        test_size=round(1.0 - val_split_size, 4),
        train_size=val_split_size,
        random_state=random_seed,
    )

    print(f"Size of training data: {len(train_data)}")
    print(f"Size of validation data: {len(val_data)}")
    print(f"Size of test data: {len(test_data)}")

    # Check to make sure no extreme class imbalance exists
    counts = dict()
    for row in train_data.iterrows():
        for movie in row[1]["movies"]:
            if movie in counts:
                counts[movie][0] += 1
            else:
                counts[movie] = [1, 0, 0]

    for row in val_data.iterrows():
        for movie in row[1]["movies"]:
            if movie in counts:
                counts[movie][1] += 1
            else:
                counts[movie] = [0, 1, 0]

    for row in test_data.iterrows():
        for movie in row[1]["movies"]:
            if movie in counts:
                counts[movie][2] += 1
            else:
                counts[movie] = [0, 0, 1]

    unpopular = 0
    no_train, no_val, no_test = 0, 0, 0
    for movie in counts:
        if sum(counts[movie]) < 100:
            unpopular += 1
        if counts[movie][0] == 0:
            no_train += 1
        if counts[movie][1] == 0:
            no_val += 1
        if counts[movie][2] == 0:
            no_test += 1

    print(f"Movies with less than 100 fans: {unpopular}")
    print(f"Movies with no training data: {no_train}")
    print(f"Movies with no validation data: {no_val}")
    print(f"Movies with no test data: {no_test}")

    print("\nSaving Training data to file...")
    train_data.to_csv("data/clean/train_data.csv", sep="\t", index=False)
    print("Saving Validation data...")
    val_data.to_csv("data/clean/val_data.csv", sep="\t", index=False)
    print("Saving Test data...")
    test_data.to_csv("data/clean/test_data.csv", sep="\t", index=False)
    print("Saving complete. Exiting now.")


if __name__ == "__main__":
    with open("params.yaml", "r") as raw_yaml:
        params = yaml.safe_load(raw_yaml)

    split_transform_file(sys.argv[1], params["prepare"])
