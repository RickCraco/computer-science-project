from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro, ttest_ind

def plot_hist(df: pd.DataFrame, column: str):
    """
    Shows the probability distribution of a column

    Args:
        df: pd.DataFrame
        column: str  column name of numeric feature
    Returns:
        None
    """
    sns.set_theme()
    plt.title(f"Distribution of column: {column}")

    # create the histogram of the column
    ax = sns.histplot(data=df, x=column, kde=True)

    # we create two vertical lines for the mean and median value
    ax.axvline(df[column].mean(), color="darkred", linestyle="--", label= "Mean")
    ax.axvline(df[column].median(), color="darkgreen", linestyle="--", label= "Median")
    ax.legend() # shows the label of the 2 lines