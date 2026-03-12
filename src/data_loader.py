import pandas as pd
import os

def load_data() ->pd.DataFrame:
    """
    Load heart.csv dataset from data/ directory.

    output:
    pd.DataFrame
    """
    # file path to the csv file
    file_path = "data/heart.csv"

    # check if the file exists
    if not os.path.exists(file_path):
        # if we are inside another directory we go back of 1
        file_path = "../data/heart.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file heart.csv not found in data/")

    return pd.read_csv(file_path)