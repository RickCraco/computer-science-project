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



def get_preprocessor(numeric_features: list, categorical_features: list):
    """
    Build the preprocessor for the complete pipeline.

    Input:

    numeric_features: list   list of numerical features
    categorical_features: list   list of categorical features

    Return:
    preprocessor: obj   returns preprocessor obj from sklearn
    """
    # standard scaler for numerical features
    numeric_transformer = StandardScaler()

    # ordinal encoding for categorical features
    categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    # preprocessor for sklearn pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor