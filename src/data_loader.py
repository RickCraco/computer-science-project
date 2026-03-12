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
    dest_folder = "data" # save the file in the data folder

    dest_path = os.path.join(dest_folder, "heart.csv")

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
    file_path = "data/heart.csv"

    if not os.path.exists(file_path):
        download_data()
    return pd.read_csv(file_path)