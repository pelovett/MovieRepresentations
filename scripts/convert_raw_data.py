import pandas as pd
from tqdm import tqdm
import os
import sys


# Movie data is displayed
#  movie_id:
#   cust_id,rating,date
# We want
#  movie_id,cust_id,rating,date


def main(input_filename, output_filename):
    df_rows = []
    cur_movie = None
    print(f'Starting {input_filename} ...')
    with open(output_filename, 'w') as out_file:
        for line in tqdm(open(input_filename, 'r')):
            cur = line.strip()
            if cur[-1] == ':':
                cur_movie = cur[:-1]
            else:
                row = cur.split(',')
                df_rows.append([cur_movie, row[0], row[1],
                               str(pd.to_datetime(row[2]))])
                if len(df_rows) > 1e4:
                    for row in df_rows:
                        out_file.write('\t'.join(row) + '\n')
                    df_rows = []

        for row in df_rows:
            out_file.write('\t'.join(row) + '\n')
    print(f'Finished {input_filename}')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
