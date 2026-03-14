import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality, get_column_plot


def generate_syn_df(df: pd.DataFrame, n_epochs: int) -> pd.DataFrame:
  """
  Generates a synthetic dataset using the CTGAN network.

  input:
  df: pd.DataFrame original dataset
  n_epochs: int number of epochs to train the synthesizer

  output:
  syn_df: pd.DataFrame
  """
  # detects the metadata from the original df
  metadata = Metadata.detect_from_dataframe(df)

  # CTGAN Synthesizer
  synthesizer = CTGANSynthesizer(
      metadata=metadata,
      epochs=n_epochs,
      verbose=True
  )

  # we divide the original dataset into train and test to prevent data leakage
  train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['HeartDisease'])

  # we train the synthesizer using the train_df
  synthesizer.fit(train_df)

  # we generate the new df
  syn_df = synthesizer.sample(num_rows=10000)

  return syn_df