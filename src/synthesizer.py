import pandas as pd
import numpy as np
import os
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
  # path to models/ folder
  dir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

  # path to save the CTGAN model
  model_path = os.path.join(dir_path, f"ctgan_{n_epochs}_epochs.pkl")

  # check if the model already exists
  if os.path.exists(model_path):
    print(f"Loading existing model: {model_path}")
    synthesizer = CTGANSynthesizer.load(model_path) # load existing model
  else:
    print(f"Training new CTGAN model with {n_epochs}...")
    
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



def diagnostic_report(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
  """
  Prints diagnostic report for synthetic data

  input:
  real_data: pd.DataFrame   original dataset
  synthetic_data: pd.DataFrame   synthetica data generated using CTGAN
  """
  # detect metadata from the df
  metadata = Metadata.detect_from_dataframe(real_data)

  # run diagnostic report
  diagnostic = run_diagnostic(real_data, synthetic_data, metadata)



def quality_report(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> pd.DataFrame:
  """
  Shows the data quality report for synthetic data

  input:
  real_data: pd.DataFrame   original dataset
  synthetic_data: pd.DataFrame    synthetic data generated using CTGAN

  output:
  q_report: obj   quality report object from SDV
  """
  # detect metadata from the df
  metadata = Metadata.detect_from_dataframe(real_data)

  # run quality report
  q_report = evaluate_quality(real_data, synthetic_data, metadata)

  return q_report