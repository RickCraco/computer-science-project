import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


def remove_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes outliers using the IQR method.

    Input:
    df: pd.DataFrame  dataset real/synthetic
    columns:  list    numerical columns

    Output:
    df:  pd.DataFrame  dataframe without outliers
    """
    # we make a copy of the dataset
    df_clean = df.copy()

    # for each column we calculate the IQR and filter the df
    for col in columns:
        Q1 = df_clean[col].quantile(0.25) # first quantile 25%
        Q3 = df_clean[col].quantile(0.75) # third quantile 75%
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # we filter the dataframe using the bounds
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

        print(f"Number of outliers removed: {len(df) - len(df_clean)}")

        return df_clean
