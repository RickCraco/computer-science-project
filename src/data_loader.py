import pandas as pd
import kagglehub
import os
import shutil

def download_data():
    """
    Download the dataset from Kaggle.
    """
    print("Downloading dataset...")
    path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

    # join the path with the file name
    source_path = os.path.join(path, "heart.csv")
    dest_path = "data/heart.csv" # save the file in the data folder

    # check if the file exists
    if os.path.exists(source_path):
        shutil.copy(source_path, dest_path) # copy the file to the data folder
        print(f"Dataset downloaded successfully {dest_path}")
    else:
        print(f"Error: file heart.csv not found in {path}")

def load_data() -> pd.DataFrame:
    """
    Load the dataset from the data folder.

    Returns:
        pd.DataFrame: The dataset.
    """
    if not os.path.exists("data/heart.csv"):
        download_data()
    return pd.read_csv("data/heart.csv")