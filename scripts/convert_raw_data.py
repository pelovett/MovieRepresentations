import pandas as pd
from tqdm import tqdm
import os

filenames = [('combined_data_1.txt', 'data_1.csv'),
             ('combined_data_2.txt', 'data_2.csv'),
             ('combined_data_3.txt', 'data_3.csv'),
             ('combined_data_4.txt', 'data_4.csv')]

# Movie data is displayed
#  movie_id:
#   cust_id,rating,date
# We want
#  movie_id,cust_id,rating,date


def main():
    for filename in filenames:
        df_rows = []
        cur_movie = None
        print(f'Starting {filename[0]}...')
        for line in tqdm(open(os.path.join('data/raw/', filename[0]), 'r')):
            cur = line.strip()
            if cur[-1] == ':':
                cur_movie = int(cur[:-1])
            else:
                row = cur.split(',')
                df_rows.append([cur_movie, int(row[0]), int(
                    row[1]), pd.to_datetime(row[2])])
        df_movies = pd.DataFrame(df_rows)
        print(f'Finished {filename[0]}')
        df_movies.to_csv(os.path.join('data/clean/', filename[1]))


if __name__ == "__main__":
    main()
