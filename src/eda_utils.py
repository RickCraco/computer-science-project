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


def plot_box(df: pd.DataFrame, column1: str, column2: str | None = None):
    """
    Draws the box plot of a single column or,
    draws the box plot of a numerical column by
    a categorical column

    input:
        df: pd.DataFrame  dataset
        column1: str  column 
        column2: str  column
    """
    sns.set_theme()

    # check if column2 is None, if so draw a single boxplot
    if column2 is None:
        # creates the box plot
        ax = sns.boxplot(data=df, y=column1)
        plt.title(f"Distribution of column: {column1}")
    else:
        # creates the box plot of column by another
        ax = sns.boxplot(data=df, x=column2, y=column1, hue=column2, palette='viridis', legend=False)
        plt.title(f"Distribution of {column1} per {column2}")


def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
  """
  Returns a Dataframe containing all descriptive statistics
  for quantitative features.

  input: pd.DataFrame
  returns: pd.DataFrame
  """
  # measures of central tendency
  mean = df.mean()
  median = df.median()
  mode = df.mode().iloc[0]
  min = df.min()
  max = df.max()

  # measures of variability
  variance = df.var()
  std_dev = df.std()
  range_val = df.max() - df.min()

  # distribution shape
  skewness = df.skew()
  kurtosis = df.kurtosis()

  # df containing all statistics
  df_stats = pd.DataFrame({
      "Mean": mean,
      "Median": median,
      "Mode": mode,
      "Min": min,
      "Max": max,
      "Variance": variance,
      "Standard Deviation": std_dev,
      "Range": range_val,
      "Skewness": skewness,
      "Kurtosis": kurtosis
  })

  df_stats = df_stats.style.format("{:.2f}").background_gradient(cmap="viridis")
  return df_stats


