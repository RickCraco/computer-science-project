import pandas as pd
from pathlib import Path

def load_data() ->pd.DataFrame:
    """
    Load heart.csv dataset from data/ directory.

    output:
    pd.DataFrame
    """
    # current file path
    current_file = Path(__file__).resolve()

    # root folder path
    project_root = current_file.parents[1]

    # dataset path
    file_path = project_root / "data" / "heart.csv"

    # check if the dataset exist
    if not file_path.exists():
        raise FileNotFoundError(f"File heart.csv not found: {file_path}")

    return pd.read_csv(file_path)