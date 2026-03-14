from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from scipy.stats.contingency import association

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


def plot_bar(df: pd.DataFrame, column: str):
    """
    Draws a bar chart of a categorical column

    input:
        df: pd.DataFrame  
        column: str   categorical column
    """
    sns.set_theme()
    plt.title(f"Distribution of column: {column}")

    # creates the countplot of the column
    ax = sns.countplot(data=df, x=column, hue=column, order=df[column].value_counts(ascending=True).index, legend='brief')

    # draws the count label on each bar
    for container in ax.containers:
        ax.bar_label(container)

    plt.tight_layout()
    plt.show()