#!/bin/bash
python3 ./scripts/convert_data.py data/raw/combined_data_1.txt data/clean/data_1.csv
python3 ./scripts/convert_data.py data/raw/combined_data_2.txt data/clean/data_2.csv
python3 ./scripts/convert_data.py data/raw/combined_data_3.txt data/clean/data_3.csv
python3 ./scripts/convert_data.py data/raw/combined_data_4.txt data/clean/data_4.csv

