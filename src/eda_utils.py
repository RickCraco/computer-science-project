import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro, ttest_ind, chi2_contingency
from scipy.stats.contingency import association
from sklearn.preprocessing import LabelEncoder

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

    plt.show()


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
        plt.show()
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


def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
  """
  Calculates the frequency table of a
  qualitative column :

  input:
  df: pd.DataFrame
  column: str (name of a column)

  return: pd.DataFrame
  """
  # calculates absolute and relative frequencies
  freq_abs = df[column].value_counts()
  freq_rel = df[column].value_counts(normalize=True)

  # creates the dataframe for the frequency table
  freq_table = pd.DataFrame({
      "ABS Freq": freq_abs,
      "REL Freq (%)": (freq_rel *100).round(2)
  })

  return freq_table


def statistical_tests(df: pd.DataFrame, column: str, target: str = 'HeartDisease'):
  """
  Performs a Shapiro test (checks if data comes froma normal distribution),
  Mann-Whitney test (checks difference between two independent groups, non-normally distributed)
  t-test (checks difference between two independent groups normally distributed)

  input:
  df: pd.DataFrame
  column: str numeric feature
  target: str target class
  """
  # divide data into two groups (group_0 -> target = 0 ; group_1 -> target = 1)
  group_0 = df[df[target] == 0][column]
  group_1 = df[df[target] == 1][column]

  # Shapiro test
  print(f"Shapiro test per {column} (target = 0)", shapiro(group_0))
  print(f"Shapirot test per {column} (target = 1)", shapiro(group_1))

  # Mann-Whitney test
  u_stat, p_val = mannwhitneyu(group_0, group_1, alternative='two-sided')
  print("\nMann-Whitney U Test:")
  print(f"U-statistic: {u_stat:.4f}, p-value: {p_val:.4e}")

  # t-test :
  print("\nP-value T-test")
  t_stat,p_value = ttest_ind(group_0, group_1, equal_var=False)
  print(p_value)


def numeric_biv_analysis(df: pd.DataFrame, column: str, target: str = 'HeartDisease'):
  """
  Performs bivariate analysis between one numeric feature
  and target variable

  input:
  df: pd.DataFrame
  column: str Numeric feature
  target: str Target class, or any categorical feature
  """
  # calculate descriptive statistics by target
  print(df.groupby(target)[column].describe())

  # performs statistical tests (Shapiro Test, Mann-Whitney, t-test)
  print("\nStatistical tests:")
  statistical_tests(df, column, target)

  # box-plot
  plot_box(df, target, column)
  plt.show()



def qualitative_association(df: pd.DataFrame, column: str, target: str = "HeartDisease"):
  """
  Analyze the relationship between a categorical feature and a target variable
  using contingency tables, visualization, and statistical association tests.

  input:
  df: pd.DataFrame
  column: str qualitative feature
  target: str target class 'HeartDisease'
  """
  sns.set_theme()

  # we use the subplots function to draw multiple plots
  fig, axs = plt.subplots(2,2, figsize=(10,8))  # this creates a grid with 2 rows and 2 columns

  # create the contigency table of ABS Freq
  contingency_table = pd.crosstab(df[column], df[target])
  sns.heatmap(contingency_table, annot=True, fmt=".2f", cmap="magma", ax=axs[0,0])
  axs[0,0].set_title("Contingency table heatmap ABS Freq")

  # create the contigency table of REL Freq %
  contingency_table_rel = pd.crosstab(df[column], df[target], normalize=True) * 100
  sns.heatmap(contingency_table_rel, annot=True, fmt=".2f", cmap="magma", ax=axs[0,1])
  axs[0,1].set_title("Contingency table heatmap REL Freq %")

  # countplot to show the distribution by target
  sns.countplot(data=df, x=column, hue=target, palette="magma", ax=axs[1,0])

  fig.delaxes(axs[1,1])
  plt.tight_layout()
  plt.show()

  # calculate the chi-quadro for qualitative association
  chi2, p_val, dof, expected = chi2_contingency(contingency_table)
  print(f"P-value for Chi-Quadro Test: {p_val}")

  # Cramer's V to quantify the association
  cramer_v = association(contingency_table, method="cramer")
  print(f"Cramer's V : {cramer_v:.3f}")



def plot_corr_matrix(df: pd.DataFrame):
  """
  Plot the correlation matrix heatmap of the dataframe

  input:
  df: pd.DataFrame
  """
  # to plot the correlation matrix of all features
  # we first need to transform the qualitative features in numeric values

  df_copy = df.copy() # we create a copy of the orginal dataset
  le = LabelEncoder() # we instantiate a LabelEncoder obj to encode categorical features

  # we select only the qualitative features
  qualitative_cols = df.select_dtypes(include="object").columns

  # we use the label encoder to encode the columns
  for col in qualitative_cols:
    df_copy[col] = le.fit_transform(df_copy[col])

  # create the correlation matrix
  correlation_matrix = df_copy.corr()

  # heatmap
  plt.figure(figsize=(10,8))
  sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")